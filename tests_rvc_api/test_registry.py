import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np
import torch
from pydantic import ValidationError

from rvc_api.errors import APIError
from rvc_api.registry import ModelRegistry
from rvc_api.schemas import InferParams


def weight(path, version="v2", f0=1, speakers=1):
    torch.save({"config": [speakers, 256, 40000], "weight": {"emb_g.weight": torch.zeros(speakers, 256)},
                "version": version, "f0": f0}, path)


def index(path, dimension=768, count=8):
    value = faiss.IndexFlatL2(dimension)
    value.add(np.ones((count, dimension), dtype="float32"))
    with path.open("wb") as target:
        faiss.write_index(value, faiss.PyCallbackIOWriter(target.write))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "rvc_models" / "音色 A"
        self.folder.mkdir(parents=True)
        weight(self.folder / "voice.pth")
        self.registry = ModelRegistry(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def resolve(self, **kwargs):
        return self.registry.resolve(InferParams(model_id="音色 A", **kwargs))

    def test_no_index_and_cross_model_index(self):
        other = self.root / "rvc_models" / "other"
        other.mkdir()
        index(other / "only.index")
        self.assertEqual(self.resolve().params.index_rate, 0)
        for kwargs in ({"index_rate": 0.1}, {"index_id": "only.index"}, {"index_mode": "required"}):
            with self.assertRaises(APIError):
                self.resolve(**kwargs)

    def test_valid_index_and_modes(self):
        index(self.folder / "feature.index")
        self.assertEqual(self.resolve().params.index_rate, 0.75)
        self.assertEqual(self.resolve(index_mode="off").params.index_rate, 0)
        with self.assertRaises(APIError):
            self.resolve(index_mode="off", index_rate=0.5)
        with self.assertRaises(APIError):
            self.resolve(index_mode="required", index_rate=0)
        self.assertIsNone(self.resolve(index_rate=0).index_path)

    def test_bad_small_or_mismatched_index(self):
        (self.folder / "bad.index").write_bytes(b"bad")
        index(self.folder / "small.index", count=7)
        index(self.folder / "wrong.index", dimension=256)
        model = self.registry.model("音色 A")
        self.assertTrue(model["has_index"])
        self.assertFalse(model["weights"][0]["index_rate_enabled"])
        self.assertEqual(self.resolve().params.index_rate, 0)
        with self.assertRaises(APIError):
            self.resolve(index_rate=1)

    def test_multiple_candidates_require_selection(self):
        weight(self.folder / "second.pth")
        with self.assertRaises(APIError):
            self.resolve()
        index(self.folder / "a.index")
        index(self.folder / "b.index")
        with self.assertRaises(APIError):
            self.resolve(weight_id="voice.pth")
        selected = self.resolve(weight_id="voice.pth", index_id="b.index")
        self.assertEqual(selected.index_path.name, "b.index")

    def test_speaker_and_non_f0(self):
        with self.assertRaises(APIError):
            self.resolve(speaker_id=1)
        weight(self.folder / "voice.pth", f0=0)
        self.assertFalse(self.resolve().metadata["supports_f0"])
        for kwargs in ({"pitch_shift": 1}, {"f0_method": "pm"}, {"protect": 0.5}):
            with self.assertRaises(APIError):
                self.resolve(**kwargs)

    def test_paths_and_cache_invalidation(self):
        for name in ("../other", "a/b", "C:\\secret", "..", "a\x00b"):
            with self.assertRaises(ValidationError):
                InferParams(model_id=name)
        outside = self.root / "secret.pth"
        weight(outside)
        with self.assertRaises(APIError):
            self.registry.checked(outside)
        (self.folder / "voice.pth").write_bytes(b"broken")
        with self.assertRaises(APIError):
            self.resolve()

    def test_symlink_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        weight(outside / "secret.pth")
        link = self.folder / "link.pth"
        try:
            link.symlink_to(outside / "secret.pth")
        except OSError:
            import _winapi
            junction = self.root / "rvc_models" / "linked-folder"
            _winapi.CreateJunction(str(outside), str(junction))
            try:
                self.assertNotIn("linked-folder", [m["model_id"] for m in self.registry.scan()["models"]])
                with self.assertRaises(APIError):
                    self.registry.checked(junction / "secret.pth")
            finally:
                junction.rmdir()
            return
        self.assertNotIn("link.pth", [w["weight_id"] for w in self.registry.model("音色 A")["weights"]])

    def test_parameter_validation(self):
        for kwargs in ({"pitch_shift": "1.2"}, {"speaker_id": True}, {"index_rate": "nan"},
                       {"rms_mix_rate": "inf"}, {"protect": 0.6}, {"resample_sr": 8000},
                       {"chunk_seconds": 0.5}, {"mode": "realtime"}, {"unknown": 1}):
            with self.assertRaises(ValidationError):
                InferParams(model_id="音色 A", **kwargs)
        self.assertEqual(InferParams(model_id="音色 A", pitch_shift="-12").pitch_shift, -12)


if __name__ == "__main__":
    unittest.main()
