"""Request-owned GPU session, reusing RVC get_f0/vc without editing core."""
import gc
import math
from types import SimpleNamespace

import numpy as np

from .audio import OutputProcessor, decode
from .errors import APIError
from .index_io import read_index
from .registry import fingerprint


class CheckedIndex:
    """The original vc divides by distances; reject invalid searches and bound zeros."""
    def __init__(self, index):
        self.index = index

    def search(self, features, k):
        scores, ids = self.index.search(features, k)
        if not np.isfinite(scores).all() or (ids < 0).any() or (ids >= self.index.ntotal).any():
            raise APIError("INCOMPATIBLE_INDEX", "索引检索没有返回有效候选")
        # A perfect match has zero L2 distance. Use a positive floor in this
        # API adapter, rather than changing the original inference function.
        return np.maximum(scores, 1e-6), ids


class InferenceEngine:
    def __init__(self, selection, settings, audio_path):
        self.selection, self.settings, self.audio_path = selection, settings, audio_path
        self.pipeline = self.net = self.hubert = self.speaker = None
        self.pitch = self.pitchf = self.index = self.vectors = None
        self.pending = None
        self.closed = False
        self.cleanup_result = None

    def prepare(self):
        import torch
        from scipy import signal
        from configs.config import infer_device, infer_dtype, infer_gpu_mem
        from infer.hubert import load_hubert_model
        from infer.module.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
                                         SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono)
        from infer.vc.pipeline import Pipeline, bh, ah

        self.device = infer_device
        self.sample_rate = self.selection.metadata["sample_rate"]
        self.params = self.selection.params
        self.half = infer_dtype == torch.float16
        self.context = 3 if self.half and infer_gpu_mem > 4 else 1
        cfg = SimpleNamespace(device=infer_device, is_half=self.half, x_pad=self.context,
                              x_query=10 if self.half else 6, x_center=60 if self.half else 38,
                              x_max=65 if self.half else 41)
        self.original = decode(self.audio_path, self.settings)
        self.valid_samples = len(self.original)
        audio = signal.filtfilt(bh, ah, self.original).astype("float32")
        self.step = max(160, round(self.params.chunk_seconds * 100) * 160)
        self.frame_samples = math.ceil(len(audio) / 160) * 160
        self.total = math.ceil(self.frame_samples / self.step)
        audio = np.pad(audio, (0, self.frame_samples - len(audio)), mode="reflect")
        self.pad = self.context * 16000
        self.audio = np.pad(audio, (self.pad, self.pad + 160), mode="reflect")
        self.original_pad = np.pad(self.original, (self.pad, self.frame_samples - len(self.original) + self.pad + 160), mode="reflect")

        with self.selection.weight_path.open("rb") as source:
            if fingerprint(self.selection.weight_path) != self.selection.weight_fingerprint:
                raise APIError("MODEL_LOAD_FAILED", "权重在选择后发生变化", 503)
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        config = list(checkpoint["config"])
        config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
        expected = self.selection.metadata
        if (checkpoint.get("version", "v1") != expected["version"] or
                bool(checkpoint.get("f0", 1)) != expected["supports_f0"] or config[-1] != self.sample_rate or
                config[-3] != expected["speaker_count"]):
            raise APIError("MODEL_LOAD_FAILED", "权重元信息发生变化", 503)
        classes = {("v1", True): SynthesizerTrnMs256NSFsid, ("v1", False): SynthesizerTrnMs256NSFsid_nono,
                   ("v2", True): SynthesizerTrnMs768NSFsid, ("v2", False): SynthesizerTrnMs768NSFsid_nono}
        self.net = classes[(expected["version"], expected["supports_f0"])](*config, is_half=self.half)
        del self.net.enc_q
        self.net.load_state_dict(checkpoint["weight"], strict=False)
        del checkpoint
        self.net = self.net.eval().to(infer_device)
        self.net = self.net.half() if self.half else self.net.float()
        self.pipeline = Pipeline(self.sample_rate, cfg)
        self.hubert = load_hubert_model(infer_device, self.half)
        self.speaker = torch.tensor([self.params.speaker_id], device=infer_device).long()

        if self.selection.index_path:
            if fingerprint(self.selection.index_path) != self.selection.index_fingerprint:
                raise APIError("INCOMPATIBLE_INDEX", "索引在选择后发生变化")
            index = read_index(self.selection.index_path)
            if index.d != expected["feature_dimension"] or index.ntotal < 8:
                raise APIError("INCOMPATIBLE_INDEX", "索引不兼容")
            self.vectors = index.reconstruct_n(0, index.ntotal)
            if not np.isfinite(self.vectors).all():
                raise APIError("INCOMPATIBLE_INDEX", "索引向量含非有限值")
            self.index = CheckedIndex(index)

        if expected["supports_f0"]:
            # Explicit local auxiliary path, no per-request environment mutation.
            if self.params.f0_method == "rmvpe":
                from infer.rmvpe import RMVPE
                self.pipeline.model_rmvpe = RMVPE(str(self.settings.project_root / "assets" / "rmvpe" / "rmvpe.pt"),
                                                 is_half=self.half, device=infer_device)
            length = self.audio.shape[0] // 160
            pitch, pitchf = self.pipeline.get_f0(self.audio, length, self.params.pitch_shift, self.params.f0_method)
            self.pitch = torch.tensor(pitch[:length], device=infer_device).unsqueeze(0).long()
            self.pitchf = torch.tensor(pitchf[:length].astype("float32"), device=infer_device).unsqueeze(0)
        self.output_sr = self.params.resample_sr or self.sample_rate
        self.processor = OutputProcessor(self.sample_rate, self.output_sr, self.params.rms_mix_rate)
        self.hold = round(self.sample_rate * 0.02)
        return {"total": self.total, "sample_rate": self.output_sr, "channels": 1, "codec": "pcm_s16le",
                "mode": "chunked_file", "holdback_seconds": 0.02, "current": 0, "status": "running",
                "parameters": self.params.dict(), "index_used": self.index is not None,
                "f0_applied": expected["supports_f0"], "device": str(infer_device)}

    def render(self, number):
        s = (number - 1) * self.step
        e = min(number * self.step, self.frame_samples)
        end = e + 2 * self.pad + 160
        local = self.audio[s:end]
        pitch = self.pitch[:, s // 160:end // 160] if self.pitch is not None else None
        pitchf = self.pitchf[:, s // 160:end // 160] if self.pitchf is not None else None
        generated = self.pipeline.vc(self.hubert, self.net, self.speaker, local, pitch, pitchf, [0, 0, 0],
                                     self.index, self.vectors, self.params.index_rate,
                                     self.selection.metadata["version"], self.params.protect)
        generated = self.processor.envelope(self.original_pad[s:end], generated)
        left = self.context * self.sample_rate
        end_valid = min(e, self.valid_samples)
        main_count = round(end_valid * self.sample_rate / 16000) - round(s * self.sample_rate / 16000)
        if len(generated) < left + main_count:
            raise APIError("INFERENCE_FAILED", "合成块长度不足", 500)
        main = generated[left:left + main_count].copy()
        if self.pending is not None:
            overlap = generated[left - len(self.pending):left]
            ramp = np.linspace(0, 1, len(self.pending), dtype="float32")
            prefix = self.pending * (1 - ramp) + overlap * ramp
            main = np.concatenate((prefix, main))
        last = number == self.total
        self.pending = None if last else main[-self.hold:].copy()
        stable = main if last else main[:-self.hold]
        offset, pcm = self.processor.pcm(stable, last=last)
        return {"index": number, "offset": offset, "samples": len(pcm) // 2, "pcm": pcm}

    def _drop_owners(self):
        from tools.cuda_graph import clear_cuda_graph_cache
        # Enumerate graph owners including nested FCPE wrappers and networks.
        seen = set()
        stack = [self.net, self.hubert, self.pipeline]
        while stack:
            owner = stack.pop()
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            clear_cuda_graph_cache(owner)
            for name in ("model_rmvpe", "model_fcpe", "model", "infer_model", "mel_extractor"):
                child = getattr(owner, name, None)
                if child is not None:
                    stack.append(child)
            if hasattr(owner, "children"):
                stack.extend(owner.children())
        self.net = self.hubert = self.pipeline = self.speaker = self.pitch = self.pitchf = None
        self.index = self.vectors = None

    def close(self):
        if self.closed:
            return self.cleanup_result
        import torch
        device = getattr(self, "device", None)
        cuda = device is not None and str(device).startswith("cuda")
        failure = None
        try:
            if cuda:
                torch.cuda.synchronize(device)
        except Exception as exc:
            failure = exc
        try:
            self._drop_owners()
        finally:
            gc.collect()
            if cuda:
                with torch.cuda.device(device):
                    # CUDA Graph warmup creates streams. PyTorch's cuBLAS
                    # workspaces are cached per stream/thread, outside Python
                    # tensor ownership, and empty_cache alone cannot free them.
                    clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
                    if clear_workspaces is None:
                        raise APIError("GPU_CLEANUP_FAILED", "当前 PyTorch 不支持释放 cuBLAS 工作区", 503)
                    clear_workspaces()
                    torch.cuda.empty_cache()
        if failure:
            raise APIError("GPU_CLEANUP_FAILED", "GPU 同步或清理失败", 503) from failure
        self.closed = True
        self.cleanup_result = {"gpu_cleanup_completed": True,
                               "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if cuda else 0,
                               "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if cuda else 0}
        return self.cleanup_result
