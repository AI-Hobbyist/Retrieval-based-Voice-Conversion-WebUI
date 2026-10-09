"""CPU-only folder registry. Never search original global index directories."""
import copy
import importlib.util
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT
from .errors import APIError
from .schemas import InferParams
from .index_io import read_index


def is_link(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def fingerprint(path):
    info = path.stat()
    return (info.st_size, info.st_mtime_ns, info.st_ino)


@dataclass(frozen=True)
class Selection:
    weight_path: Path
    index_path: Path | None
    weight_fingerprint: tuple
    index_fingerprint: tuple | None
    metadata: dict
    params: InferParams


class ModelRegistry:
    def __init__(self, project_root=PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "rvc_models"
        self._cache = {}
        self._lock = threading.RLock()

    def checked(self, path):
        path = Path(path)
        try:
            path.resolve().relative_to(self.root.resolve())
            relative = path.relative_to(self.root)
            current = self.root
            for item in (None, *relative.parts):
                if item is not None:
                    current /= item
                if is_link(current):
                    raise ValueError("linked paths are not allowed")
            if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("not a regular file")
        except (OSError, ValueError) as exc:
            raise APIError("INVALID_PARAMETER", "模型文件路径不可用或越界") from exc
        return path

    def _weight(self, path):
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config, weights = checkpoint["config"], checkpoint["weight"]
        version, f0 = checkpoint.get("version", "v1"), checkpoint.get("f0", 1)
        speakers, sr = int(weights["emb_g.weight"].shape[0]), int(config[-1])
        if version not in ("v1", "v2") or f0 not in (0, 1) or speakers < 1 or sr not in (32000, 40000, 48000):
            raise ValueError("invalid inference checkpoint metadata")
        return {"version": version, "supports_f0": bool(f0), "speaker_count": speakers,
                "sample_rate": sr, "feature_dimension": 256 if version == "v1" else 768}

    def _index(self, path):
        import numpy as np
        index = read_index(path)
        if index.ntotal < 8:
            raise ValueError("index must contain at least 8 vectors")
        if index.d not in (256, 768):
            raise ValueError("index feature dimension must be 256 or 768")
        vectors = index.reconstruct_n(0, min(index.ntotal, 8))
        if not np.isfinite(vectors).all():
            raise ValueError("index vectors must be finite")
        return {"dimension": index.d, "vector_count": index.ntotal}

    def _inspect(self, path, kind):
        key = (str(path), fingerprint(path))
        with self._lock:
            if key not in self._cache:
                # Cache primitives only; never retain a checkpoint or GPU tensor.
                try:
                    value = {"usable": True, **getattr(self, "_" + kind)(path)}
                except Exception:
                    value = {"usable": False, "reason": "权重损坏或格式不支持" if kind == "weight" else "索引损坏、无向量或格式不支持"}
                self._cache = {k: v for k, v in self._cache.items() if k[0] != str(path)}
                self._cache[key] = value
            return copy.deepcopy(self._cache[key])

    def scan(self):
        if not self.root.exists():
            return {"state": "directory_missing", "models": [], "issues": []}
        if is_link(self.root) or not self.root.is_dir():
            return {"state": "invalid_directory", "models": [], "issues": ["模型根目录必须是普通目录"]}
        models, issues, names = [], [], set()
        for folder in sorted(self.root.iterdir(), key=lambda p: p.name.casefold()):
            if is_link(folder) or not folder.is_dir():
                continue
            if folder.name.casefold() in names:
                issues.append("模型 ID 大小写冲突")
                continue
            names.add(folder.name.casefold())
            weights, indexes, folder_issues = [], [], []
            for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
                if path.suffix.lower() not in (".pth", ".index"):
                    continue
                try:
                    self.checked(path)
                except APIError:
                    folder_issues.append("链接或非普通模型文件已排除")
                    continue
                kind = "weight" if path.suffix.lower() == ".pth" else "index"
                item = {kind + "_id": path.name, **self._inspect(path, kind)}
                (weights if kind == "weight" else indexes).append(item)
            for weight in weights:
                compatible = [i["index_id"] for i in indexes if i["usable"] and
                              i["dimension"] == weight.get("feature_dimension")]
                weight.update(index_rate_enabled=bool(compatible), default_index_rate=0.75 if compatible else 0,
                              compatible_indexes=compatible)
            if weights:
                models.append({"model_id": folder.name, "display_name": folder.name,
                               "has_index": bool(indexes), "weights": weights,
                               "indexes": indexes, "issues": folder_issues})
        return {"state": "ready" if models else "empty", "models": models, "issues": issues}

    def model(self, model_id):
        InferParams(model_id=model_id)
        for model in self.scan()["models"]:
            if model["model_id"] == model_id:
                return model
        raise APIError("MODEL_NOT_FOUND", "模型不存在", 404)

    def resolve(self, params):
        model = self.model(params.model_id)
        weights = model["weights"]
        if params.weight_id is None:
            if len(weights) != 1:
                raise APIError("AMBIGUOUS_SELECTION", "多个权重需要 weight_id")
            weight = weights[0]
        else:
            weight = next((w for w in weights if w["weight_id"] == params.weight_id), None)
            if weight is None:
                raise APIError("WEIGHT_NOT_FOUND", "所选模型文件夹内没有此权重", 404)
        if not weight["usable"]:
            raise APIError("MODEL_LOAD_FAILED", weight["reason"], 503)
        if params.speaker_id >= weight["speaker_count"]:
            raise APIError("INVALID_PARAMETER", "speaker_id 超出模型说话人范围")
        if not weight["supports_f0"] and (params.pitch_shift != 0 or params.f0_method != "rmvpe" or params.protect != 0.33):
            raise APIError("INVALID_PARAMETER", "此权重不支持 F0 参数调整")
        indexes = [i for i in model["indexes"] if i["index_id"] in weight["compatible_indexes"]]
        rate = params.index_rate
        if params.index_mode == "off":
            if (rate is not None and rate != 0) or params.index_id is not None:
                raise APIError("INVALID_PARAMETER", "off 模式不接受索引或非零 Index Rate")
            rate = 0
        elif rate is None:
            rate = 0.75 if indexes else 0
        if params.index_mode == "required" and rate == 0:
            raise APIError("INDEX_NOT_AVAILABLE", "required 模式需要可用索引及非零 Index Rate")
        selected = None
        if rate > 0 or params.index_id is not None:
            if not indexes:
                raise APIError("INDEX_NOT_AVAILABLE", "所选权重没有可用索引")
            if params.index_id is not None:
                selected = next((i for i in indexes if i["index_id"] == params.index_id), None)
                if selected is None:
                    raise APIError("INCOMPATIBLE_INDEX", "所选索引不存在、不可用或维度不匹配")
            elif len(indexes) == 1:
                selected = indexes[0]
            else:
                raise APIError("AMBIGUOUS_SELECTION", "多个索引需要 index_id")
        effective = params.copy(update={"weight_id": weight["weight_id"], "index_rate": rate,
                                        "index_id": selected["index_id"] if selected and rate > 0 else None})
        base = self.root / params.model_id
        weight_path = self.checked(base / weight["weight_id"])
        index_path = self.checked(base / effective.index_id) if effective.index_id else None
        return Selection(weight_path, index_path, fingerprint(weight_path),
                         fingerprint(index_path) if index_path else None, weight, effective)

    def f0_methods(self):
        rmvpe = self.project_root / "assets" / "rmvpe" / "rmvpe.pt"
        result = []
        for method, package in (("pm", "parselmouth"), ("rmvpe", "torch"), ("fcpe", "torchfcpe")):
            available = importlib.util.find_spec(package) is not None and (method != "rmvpe" or rmvpe.is_file())
            result.append({"id": method, "available": available, "verification": "dependency_check",
                           "reason": None if available else "依赖或辅助权重未就绪", "default": method == "rmvpe"})
        return result
