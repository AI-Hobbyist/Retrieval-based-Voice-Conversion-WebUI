import os
import math
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    bearer_token: str = field(default_factory=lambda: os.getenv("API_BEARER_TOKEN", ""))
    max_request_bytes: int = 100 * 1024 * 1024
    max_audio_seconds: float = 600.0
    upload_timeout: float = 120.0
    inference_timeout: float = 600.0
    decode_timeout: float = 60.0

    @property
    def model_root(self):
        return self.project_root / "rvc_models"

    @property
    def temp_root(self):
        return self.project_root / "rvc_api" / "tmp"

    @classmethod
    def from_env(cls):
        values = {}
        for name, kind in (("max_request_bytes", int), ("max_audio_seconds", float),
                           ("upload_timeout", float), ("inference_timeout", float),
                           ("decode_timeout", float)):
            key = "API_" + name.upper()
            if key in os.environ:
                value = kind(os.environ[key])
                if value <= 0 or not math.isfinite(value):
                    raise ValueError(f"{key} must be positive and finite")
                values[name] = value
        return cls(**values)
