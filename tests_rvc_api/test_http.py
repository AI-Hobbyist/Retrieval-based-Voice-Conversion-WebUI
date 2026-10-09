import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import soxr
from fastapi.testclient import TestClient

from rvc_api.app import create_app
from rvc_api.audio import OutputProcessor
from rvc_api.config import Settings, PROJECT_ROOT
from rvc_api.errors import APIError
from rvc_api.protocol import part, closing
from helpers import MultipartReader, parse_response
from test_registry import weight


class FakeEngine:
    instances = []
    fail_chunk = 0
    fail_cleanup = False

    def __init__(self, selection, settings, path):
        self.selection, self.path = selection, path
        self.cleaned = False
        self.rendered = []
        self.instances.append(self)

    def prepare(self):
        if self.path.read_bytes() == b"bad":
            raise APIError("INVALID_AUDIO", "bad")
        return {"total": 3, "codec": "pcm_s16le", "sample_rate": 16000, "channels": 1}

    def render(self, number):
        time.sleep(0.01)
        if number == self.fail_chunk:
            raise APIError("GPU_OUT_OF_MEMORY", "fixture OOM", 503)
        self.rendered.append(number)
        pcm = np.full(16000, number, dtype="<i2").tobytes()
        return {"index": number, "offset": (number - 1) * 16000, "samples": 16000, "pcm": pcm}

    def close(self):
        if self.fail_cleanup:
            raise APIError("GPU_CLEANUP_FAILED", "fixture cleanup", 503)
        self.cleaned = True
        return {"gpu_cleanup_completed": True, "cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0}


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        folder = self.root / "rvc_models" / "voice"
        folder.mkdir(parents=True)
        weight(folder / "voice.pth", f0=0)
        FakeEngine.instances, FakeEngine.fail_chunk, FakeEngine.fail_cleanup = [], 0, False

    def tearDown(self):
        self.temp.cleanup()

    def client(self, **kwargs):
        return TestClient(create_app(Settings(project_root=self.root, **kwargs), FakeEngine))

    def send(self, client, body=b"audio", **kwargs):
        return client.post("/api/v1/infer?model_id=voice", content=body,
                           headers={"Content-Type": "audio/wav"}, **kwargs)

    def test_init_models_modes_f0_and_cpu_policy(self):
        with self.client() as client:
            value = client.get("/api/v1/init").json()
            self.assertEqual(value["models"][0]["model_id"], "voice")
            self.assertEqual(value["gpu_policy"], "unload_after_each_request")
            self.assertEqual(value["models"][0]["weights"][0]["default_index_rate"], 0)
            for path in ("models", "models/voice", "f0-methods", "modes"):
                self.assertEqual(client.get("/api/v1/" + path).status_code, 200)
            self.assertTrue(client.get("/health").json()["alive"])

    def test_progress_audio_offsets_done_and_cleanup(self):
        with self.client() as client:
            response = self.send(client)
            self.assertEqual(response.status_code, 200)
            parts = parse_response(response)
            self.assertEqual([p[0] for p in parts], ["start", "audio", "progress", "audio", "progress", "audio", "progress", "done"])
            self.assertEqual([p[2]["current"] for p in parts if p[0] == "progress"], [1, 2, 3])
            self.assertEqual([int(p[1]["x-sample-offset"]) for p in parts if p[0] == "audio"], [0, 16000, 32000])
            self.assertTrue(parts[-1][2]["gpu_cleanup_completed"])
            self.assertTrue(FakeEngine.instances[0].cleaned)
            self.assertFalse(list((self.root / "rvc_api" / "tmp").iterdir()))
            self.assertFalse(client.app.state.service.gpu_gate.locked())

    def test_auth_all_business_routes_and_schema(self):
        with self.client(bearer_token="secret") as client:
            for path in ("init", "models", "models/voice", "modes", "f0-methods"):
                self.assertEqual(client.get("/api/v1/" + path).status_code, 401)
                self.assertEqual(client.get("/api/v1/" + path, headers={"Authorization": "Bearer secret"}).status_code, 200)
            response = self.send(client)
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers["www-authenticate"], "Bearer")
            self.assertEqual(client.get("/api/v1/init", headers={"Authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(client.get("/api/v1/init", headers={"Authorization": "Basic secret"}).status_code, 401)
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/openapi.json").status_code, 404)
            self.assertEqual(client.post("/api/v1/infer?model_id=voice", content=b"audio",
                                        headers={"Authorization": "Bearer secret", "Content-Type": "audio/wav"}).status_code, 200)

    def test_limits_bad_input_and_cleanup(self):
        with self.client(max_request_bytes=4) as client:
            self.assertEqual(self.send(client, b"large").status_code, 413)
            self.assertEqual(self.send(client, b"").status_code, 422)
            self.assertEqual(self.send(client, b"bad").status_code, 422)
            self.assertFalse(client.app.state.service.gpu_gate.locked())
            self.assertTrue(FakeEngine.instances[-1].cleaned)
        with self.client() as client:
            for query in ("pitch_shift=1.5", "index_rate=1", "resample_sr=8000", "unknown=1", "model_id=voice"):
                self.assertEqual(client.post("/api/v1/infer?model_id=voice&" + query, content=b"a",
                                            headers={"Content-Type": "audio/wav"}).status_code, 422)

    def test_stream_failure_has_no_done_and_releases(self):
        FakeEngine.fail_chunk = 2
        with self.client() as client:
            parts = parse_response(self.send(client))
            self.assertEqual(parts[-1][0], "error")
            self.assertEqual(parts[-1][2]["current"], 1)
            self.assertEqual(parts[-1][2]["status"], "failed")
            self.assertNotIn("done", [p[0] for p in parts])
            self.assertTrue(FakeEngine.instances[-1].cleaned)
            self.assertFalse(client.app.state.service.gpu_gate.locked())

    def test_cleanup_failure_marks_unready(self):
        FakeEngine.fail_cleanup = True
        with self.client() as client:
            parts = parse_response(self.send(client))
            self.assertEqual(parts[-1][0], "error")
            self.assertFalse(parts[-1][2]["gpu_cleanup_completed"])
            self.assertFalse(client.get("/health").json()["ready"])
            self.assertEqual(self.send(client).status_code, 503)

    def test_multipart_random_fragmentation_and_embedded_boundary(self):
        boundary = "example"
        payload = b"\x00\xff\r\n--example\r\nnot a header\x00"
        encoded = b"".join(part(boundary, "audio", payload)) + closing(boundary)
        reader = MultipartReader(boundary)
        found = []
        for byte in encoded:
            found.extend(reader.feed(bytes([byte])))
        self.assertTrue(reader.closed)
        self.assertEqual(found[0][2], payload)

    def test_stateful_resampling_offsets_and_limiter(self):
        samples = np.sin(np.arange(44100 * 3) * 0.03).astype("float32") * 0.1
        processor = OutputProcessor(44100, 24000, 1)
        result = []
        offsets = []
        for number, segment in enumerate(np.array_split(samples, 3)):
            offset, pcm = processor.pcm(segment, last=number == 2)
            offsets.append(offset)
            result.append(np.frombuffer(pcm, dtype="<i2"))
        combined = np.concatenate(result)
        expected = np.rint(soxr.resample(samples, 44100, 24000) * 32767).astype("int16")
        self.assertEqual(len(combined), 72000)
        self.assertLessEqual(np.max(np.abs(combined.astype("int32") - expected)), 1)
        self.assertEqual(offsets, [0, len(result[0]), len(result[0]) + len(result[1])])
        _, pcm = OutputProcessor(16000, 16000, 1).pcm(np.full(16000, 5.0, dtype="float32"), last=True)
        self.assertLessEqual(np.abs(np.frombuffer(pcm, dtype="<i2")).max(), 32766)

    def test_original_files_unchanged(self):
        manifest = json.loads((Path(__file__).parent / "original_files.json").read_text(encoding="utf-8-sig"))
        for relative, expected in manifest.items():
            self.assertEqual(hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest().upper(), expected, relative)


if __name__ == "__main__":
    unittest.main()
