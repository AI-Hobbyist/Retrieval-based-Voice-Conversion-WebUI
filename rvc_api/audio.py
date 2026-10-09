"""CPU decoding and bounded, stateful output processing."""
import json
import subprocess

import numpy as np
from scipy import ndimage
import soxr

from .errors import APIError


def decode(path, settings):
    probe = settings.project_root / "ffprobe.exe"
    ffmpeg = settings.project_root / "ffmpeg.exe"
    try:
        info = subprocess.run([str(probe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                              capture_output=True, timeout=settings.decode_timeout, check=True)
        meta = json.loads(info.stdout)
        stream = next(s for s in meta["streams"] if s.get("codec_type") == "audio")
        duration = stream.get("duration", meta.get("format", {}).get("duration"))
        if duration is not None and float(duration) > settings.max_audio_seconds:
            raise APIError("AUDIO_LIMIT_EXCEEDED", "音频时长超过限制", 413)
        # Decode at most the limit plus a small sentinel. Even unknown-duration
        # compressed inputs cannot inflate into an unbounded ndarray.
        result = subprocess.run([str(ffmpeg), "-v", "error", "-nostdin", "-i", str(path),
                                 "-map", "0:a:0", "-t", str(settings.max_audio_seconds + 0.1),
                                 "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                                capture_output=True, timeout=settings.decode_timeout, check=True)
        audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    except APIError:
        raise
    except (OSError, ValueError, StopIteration, KeyError, subprocess.SubprocessError) as exc:
        raise APIError("INVALID_AUDIO", "音频无法解码或解码超时") from exc
    if len(audio) > int(settings.max_audio_seconds * 16000):
        raise APIError("AUDIO_LIMIT_EXCEEDED", "解码后音频超过限制", 413)
    if len(audio) < 1600:
        raise APIError("AUDIO_TOO_SHORT", "音频至少需要 0.1 秒")
    if not np.isfinite(audio).all():
        raise APIError("INVALID_AUDIO", "音频含非有限值")
    peak = float(np.abs(audio).max()) / 0.95
    if peak > 1:
        audio /= peak
    return audio


class OutputProcessor:
    def __init__(self, source_sr, target_sr, rms_rate):
        self.source_sr, self.target_sr, self.rms_rate = source_sr, target_sr, rms_rate
        self.resampler = None if source_sr == target_sr else soxr.ResampleStream(source_sr, target_sr, 1, dtype="float32")
        self.gain = 1.0
        self.rms_gain = None
        self.offset = 0

    def envelope(self, original, generated):
        generated = np.asarray(generated, dtype="float32").copy()
        if not np.isfinite(generated).all():
            raise APIError("INFERENCE_FAILED", "合成产生非有限音频", 500)
        if self.rms_rate != 1:
            # Compute centered envelope over context, then crop; edge errors
            # do not fall at the delivery seam. Keep gain interpolation state.
            in_rms = np.sqrt(np.maximum(ndimage.uniform_filter1d(np.square(original, dtype="float64"), 16000), 0))
            out_rms = np.sqrt(np.maximum(ndimage.uniform_filter1d(np.square(generated, dtype="float64"), self.source_sr), 1e-12))
            ref = np.interp(np.arange(len(generated)) * 16000 / self.source_sr,
                            np.arange(len(original)), in_rms)
            gain = np.power(ref / out_rms, 1 - self.rms_rate)
            if not np.isfinite(gain).all():
                raise APIError("INFERENCE_FAILED", "包络处理产生非有限值", 500)
            generated *= gain.astype("float32")
        return generated

    def pcm(self, audio, last=False):
        audio = np.asarray(audio, dtype="float32")
        if self.resampler is not None:
            audio = self.resampler.resample_chunk(audio, last=last)
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise APIError("INFERENCE_FAILED", "输出块为空或含非有限值", 500)
        # 10 ms bounded lookahead, instantaneous attack, 50 ms release.
        peaks = ndimage.maximum_filter1d(np.abs(audio), max(1, int(self.target_sr * 0.02)))
        release = 1 - np.exp(-1 / (self.target_sr * 0.05))
        limited = np.empty_like(audio)
        for i, (sample, peak) in enumerate(zip(audio, peaks)):
            wanted = min(1.0, 0.99 / max(float(peak), 1e-12))
            self.gain = wanted if wanted < self.gain else self.gain + (wanted - self.gain) * release
            limited[i] = sample * self.gain
        pcm = np.rint(np.clip(limited, -0.999, 0.999) * 32767).astype("<i2").tobytes()
        offset = self.offset
        self.offset += len(pcm) // 2
        return offset, pcm
