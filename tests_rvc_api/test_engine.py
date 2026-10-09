"""Real local CUDA/CPU inference using the bundled runtime and installed models."""
import json
from pathlib import Path
import time
import unittest
import wave

import numpy as np
import torch

from rvc_api.config import PROJECT_ROOT, Settings
from rvc_api.engine import InferenceEngine
from rvc_api.registry import ModelRegistry
from rvc_api.schemas import InferParams


def tiny_non_f0(path):
    from infer.module.models import SynthesizerTrnMs768NSFsid_nono
    config = [1025, 32, 32, 32, 64, 2, 2, 3, 0.0, "1", [3, 7, 11],
              [[1, 3, 5]] * 3, [10, 10, 2, 2], 128, [16, 16, 4, 4], 1, 32, 40000]
    torch.manual_seed(123)
    net = SynthesizerTrnMs768NSFsid_nono(*config, is_half=False)
    torch.save({"config": config, "weight": net.state_dict(), "f0": 0, "version": "v2"}, path)


def sample_audio(path, seconds=3.17):
    t = np.arange(round(16000 * seconds)) / 16000
    envelope = 0.2 * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    audio = envelope * (np.sin(2 * np.pi * (150 * t + 10 * t * t)) + 0.25 * np.sin(2 * np.pi * 300 * t))
    # Silence crosses a delivery seam and voiced energy resumes afterward.
    audio[(t > 0.92) & (t < 1.08)] = 0
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(np.rint(audio * 32767).astype("<i2").tobytes())


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.artifacts = Path(__file__).parent / "artifacts"
        cls.artifacts.mkdir(exist_ok=True)
        cls.source = cls.artifacts / "voiced-input.wav"
        sample_audio(cls.source)
        cls.test_folder = PROJECT_ROOT / "rvc_models" / "_api_test_nonf0"
        cls.test_folder.mkdir(exist_ok=False)
        tiny_non_f0(cls.test_folder / "fixture.pth")

    @classmethod
    def tearDownClass(cls):
        (cls.test_folder / "fixture.pth").unlink()
        cls.test_folder.rmdir()

    def execute(self, model, **params):
        registry = ModelRegistry()
        selected = registry.resolve(InferParams(model_id=model, chunk_seconds=1, **params))
        engine = InferenceEngine(selected, Settings(), self.source)
        baseline = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        timing, chunks, offsets, cleanup = [], [], [], None
        try:
            start = engine.prepare()
            for number in range(1, start["total"] + 1):
                chunk = engine.render(number)
                timing.append(time.monotonic())
                chunks.append(chunk["pcm"])
                offsets.append(chunk["offset"])
        finally:
            cleanup = engine.close()
        self.assertEqual(start["total"], 4)
        self.assertTrue(cleanup["gpu_cleanup_completed"])
        self.assertLessEqual(cleanup["cuda_allocated_bytes"], baseline + 1048576)
        self.assertEqual(offsets, list(np.cumsum([0] + [len(p) // 2 for p in chunks[:-1]])))
        pcm = b"".join(chunks)
        self.assertEqual(len(pcm) // 2, round(3.17 * start["sample_rate"]))
        value = np.frombuffer(pcm, dtype="<i2")
        self.assertGreater(np.abs(value.astype("int32")).max(), 0)
        self.assertLess(np.abs(value.astype("int32")).max(), 32767)
        with wave.open(str(self.artifacts / (model + ("-index" if start["index_used"] else "-plain") + ".wav")), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(start["sample_rate"])
            target.writeframes(pcm)
        report = {"model": model, "parameters": start["parameters"], "chunks": len(chunks), "samples": len(pcm) // 2,
                  "sample_rate": start["sample_rate"], "index_used": start["index_used"], "device": start["device"],
                  "chunk_complete_times": timing, "cleanup": cleanup}
        (self.artifacts / (model + "-result.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        return report

    def test_real_f0_index_and_resampling(self):
        report = self.execute("guanguanV1", resample_sr=24000, pitch_shift=2)
        self.assertTrue(report["index_used"])

    def test_real_f0_without_index(self):
        report = self.execute("youzhanv2-xi", index_rate=0, f0_method="fcpe", rms_mix_rate=1, protect=0.5)
        self.assertFalse(report["index_used"])

    def test_real_non_f0(self):
        report = self.execute("_api_test_nonf0", resample_sr=16000)
        self.assertFalse(report["index_used"])

    def test_real_chinese_model_and_weight_path(self):
        report = self.execute("芙宁娜", resample_sr=32000, f0_method="pm")
        self.assertTrue(report["index_used"])

    def test_cublas_workspace_cleanup_regression(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA workspace regression requires CUDA")
        baseline = torch.cuda.memory_allocated()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = torch.randn(256, 256, device="cuda")
            b = a @ a
        stream.synchronize()
        del a, b, stream
        engine = InferenceEngine(None, Settings(), self.source)
        engine.device = torch.device("cuda:0")
        cleanup = engine.close()
        self.assertLessEqual(cleanup["cuda_allocated_bytes"], baseline)


if __name__ == "__main__":
    unittest.main()
