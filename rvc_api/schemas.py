import math
import re
from typing import Literal, Optional

from pydantic import BaseModel, validator

PARAMETERS = {
    "speaker_id": {"type": "integer", "default": 0, "minimum": 0, "maximum": "speaker_count-1"},
    "pitch_shift": {"type": "integer", "default": 0, "requires": "supports_f0"},
    "f0_method": {"enum": ["pm", "rmvpe", "fcpe"], "default": "rmvpe", "requires": "supports_f0"},
    "index_rate": {"type": "number", "minimum": 0, "maximum": 1, "default": "0.75 if usable_index else 0", "requires": "usable_index when >0"},
    "resample_sr": {"type": "integer", "default": 0, "allowed": "0 or 16000..48000"},
    "rms_mix_rate": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.25},
    "protect": {"type": "number", "minimum": 0, "maximum": 0.5, "default": 0.33, "requires": "supports_f0"},
    "chunk_seconds": {"type": "number", "minimum": 1, "maximum": 30, "default": 5, "alignment_samples": 160},
    "mode": {"enum": ["chunked_file"], "default": "chunked_file"},
    "index_mode": {"enum": ["auto", "off", "required"], "default": "auto"},
}


class InferParams(BaseModel):
    model_id: str
    weight_id: Optional[str] = None
    index_id: Optional[str] = None
    speaker_id: int = 0
    pitch_shift: int = 0
    f0_method: Literal["pm", "rmvpe", "fcpe"] = "rmvpe"
    index_rate: Optional[float] = None
    resample_sr: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    chunk_seconds: float = 5
    mode: Literal["chunked_file"] = "chunked_file"
    index_mode: Literal["auto", "off", "required"] = "auto"

    class Config:
        extra = "forbid"

    @validator("model_id", "weight_id", "index_id")
    def safe_id(cls, value):
        if value is not None and (not value or value in (".", "..") or
                                  any(c in value for c in "/\\:\x00") or value != value.strip()):
            raise ValueError("ID must be a name returned by the model API, not a path")
        return value

    @validator("speaker_id", "pitch_shift", "resample_sr", pre=True)
    def exact_integer(cls, value):
        if isinstance(value, bool) or not re.fullmatch(r"[+-]?\d+", str(value)):
            raise ValueError("must be an integer without truncation")
        return int(value)

    @validator("speaker_id")
    def nonnegative_speaker(cls, value):
        if value < 0:
            raise ValueError("speaker_id must be nonnegative")
        return value

    @validator("resample_sr")
    def sample_rate(cls, value):
        if value != 0 and not 16000 <= value <= 48000:
            raise ValueError("resample_sr must be 0 or 16000..48000")
        return value

    @validator("index_rate", "rms_mix_rate", "protect", "chunk_seconds")
    def finite_range(cls, value, field):
        if value is None:
            return value
        low, high = {"index_rate": (0, 1), "rms_mix_rate": (0, 1),
                     "protect": (0, 0.5), "chunk_seconds": (1, 30)}[field.name]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"must be finite and within {low}..{high}")
        return value


def modes():
    return {
        "processing_modes": [{"id": "chunked_file", "upload_complete_before_inference": True,
                              "audio_emitted": "after_each_inference_chunk"}],
        "index_modes": [{"id": "auto"}, {"id": "off", "index_rate": 0},
                        {"id": "required", "requires": "usable_index and index_rate>0"}],
    }
