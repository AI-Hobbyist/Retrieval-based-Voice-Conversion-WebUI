"""Real TCP tests. Uvicorn lives inside this foreground test process."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import re
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import wave

import httpx
import numpy as np
import torch
import uvicorn

from rvc_api.app import create_app
from rvc_api.client import MultipartReader, infer as client_infer
from rvc_api.config import PROJECT_ROOT, Settings
from rvc_api.engine import InferenceEngine
from rvc_api.errors import APIError
from test_engine import sample_audio
from test_registry import weight


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for observed server condition")


class LiveServer:
    def __init__(self, app):
        self.app = app
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.port = self.socket.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="info"))
        self.server.install_signal_handlers = lambda: None
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True)
        self.thread.start()
        wait_for(lambda: self.server.started)
        self.url = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.should_exit = True
        self.thread.join(20)
        if self.thread.is_alive():
            raise AssertionError("Test server failed to shut down")
        self.socket.close()


class ProbeEngine:
    instances = []
    total = 6
    bytes_per_chunk = 32000
    delay = 0.08
    fail_chunk = 0

    def __init__(self, selection, settings, path):
        self.selection, self.path = selection, path
        self.rendered = []
        self.cleaned = False
        self.instances.append(self)

    def prepare(self):
        return {"total": self.total, "codec": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                "parameters": self.selection.params.dict()}

    def render(self, number):
        time.sleep(self.delay)
        if number == self.fail_chunk:
            raise APIError("GPU_OUT_OF_MEMORY", "injected failure", 503)
        self.rendered.append((number, time.monotonic()))
        value = 1 if self.selection.params.model_id == "A" else 2
        count = self.bytes_per_chunk // 2
        return {"index": number, "offset": (number - 1) * count, "samples": count,
                "pcm": np.full(count, value, dtype="<i2").tobytes()}

    def close(self):
        self.cleaned = True
        return {"gpu_cleanup_completed": True}


def events(response):
    reader = MultipartReader(re.search(r"boundary=([^;]+)", response.headers["content-type"]).group(1))
    for data in response.iter_bytes(chunk_size=4096):
        for event, headers, body in reader.feed(data):
            yield event, headers, body if event == "audio" else json.loads(body)
    if not reader.closed:
        raise AssertionError("Multipart termination missing")


class LiveResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("A", "B"):
            folder = self.root / "rvc_models" / name
            folder.mkdir(parents=True)
            weight(folder / "voice.pth", f0=0)
        ProbeEngine.instances, ProbeEngine.total = [], 6
        ProbeEngine.bytes_per_chunk, ProbeEngine.delay, ProbeEngine.fail_chunk = 32000, 0.08, 0
        self.settings = Settings(project_root=self.root, bearer_token="test-only-token", max_request_bytes=64,
                                 upload_timeout=0.5, inference_timeout=10)
        self.app = create_app(self.settings, ProbeEngine)
        self.live = LiveServer(self.app)
        self.client = httpx.Client(base_url=self.live.url, timeout=20,
                                   headers={"Authorization": "Bearer test-only-token"})

    def tearDown(self):
        self.client.close()
        self.live.close()
        self.temp.cleanup()

    def stream(self, model="A", body=None):
        return self.client.stream("POST", "/api/v1/infer", params={"model_id": model},
                                  content=body or iter([b"one", b"two"]), headers={"Content-Type": "audio/wav"})

    def clean(self):
        wait_for(lambda: not self.app.state.service.gpu_gate.locked() and not self.app.state.service.jobs)
        self.assertFalse(list(self.settings.temp_root.iterdir()))

    def test_first_chunk_before_last_and_serial_model_isolation(self):
        with self.stream("A") as response:
            iterator = events(response)
            self.assertEqual(next(iterator)[0], "start")
            event, _, pcm = next(iterator)
            self.assertEqual(event, "audio")
            first_received = time.monotonic()
            busy = self.client.post("/api/v1/infer?model_id=B", content=b"audio", headers={"Content-Type": "audio/wav"})
            self.assertEqual(busy.status_code, 429)
            health_start = time.monotonic()
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertLess(time.monotonic() - health_start, 1)
            self.assertTrue(np.all(np.frombuffer(pcm, dtype="<i2") == 1))
            remaining = list(iterator)
        self.assertLess(first_received, ProbeEngine.instances[0].rendered[-1][1])
        self.assertEqual([p[2]["current"] for p in remaining if p[0] == "progress"], list(range(1, 7)))
        self.clean()
        with self.stream("B") as response:
            parts = list(events(response))
        self.assertTrue(all(np.all(np.frombuffer(p[2], dtype="<i2") == 2) for p in parts if p[0] == "audio"))
        self.assertTrue(all(e.cleaned for e in ProbeEngine.instances))
        self.clean()

    def test_backpressure_and_disconnect(self):
        ProbeEngine.total, ProbeEngine.bytes_per_chunk, ProbeEngine.delay = 80, 512 * 1024, 0.01
        peer = socket.socket()
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        peer.connect(("127.0.0.1", self.live.port))
        request = ("POST /api/v1/infer?model_id=A HTTP/1.1\r\nHost: localhost\r\n"
                   "Authorization: Bearer test-only-token\r\nContent-Type: audio/wav\r\nContent-Length: 1\r\n\r\na")
        peer.sendall(request.encode("ascii"))
        wait_for(lambda: ProbeEngine.instances and len(ProbeEngine.instances[0].rendered) >= 3)
        job = next(iter(self.app.state.service.jobs))
        time.sleep(0.4)
        count = len(ProbeEngine.instances[0].rendered)
        self.assertLess(count, 80)
        self.assertLessEqual(job.max_queue_depth, 2)
        print(json.dumps({"backpressure_rendered": count, "planned": 80, "max_queue_depth": job.max_queue_depth}), flush=True)
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0))
        peer.close()
        wait_for(lambda: not self.app.state.service.jobs, timeout=3)
        self.clean()
        self.assertTrue(ProbeEngine.instances[0].cleaned)

    def test_upload_limits_timeout_and_disconnect_before_ready(self):
        response = self.client.post("/api/v1/infer?model_id=A", content=iter([b"x" * 32, b"x" * 33]),
                                    headers={"Content-Type": "audio/wav"})
        self.assertEqual(response.status_code, 413)
        peer = socket.create_connection(("127.0.0.1", self.live.port))
        peer.settimeout(4)
        peer.sendall(b"POST /api/v1/infer?model_id=A HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-only-token\r\nContent-Type: audio/wav\r\nContent-Length: 20\r\n\r\na")
        received = peer.recv(4096)
        self.assertIn(b"408", received.split(b"\r\n", 1)[0])
        peer.close()
        wait_for(lambda: not self.app.state.service.gpu_gate.locked())
        peer = socket.create_connection(("127.0.0.1", self.live.port))
        peer.sendall(b"POST /api/v1/infer?model_id=A HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-only-token\r\nContent-Type: audio/wav\r\nContent-Length: 20\r\n\r\na")
        time.sleep(0.05)
        peer.close()
        wait_for(lambda: not self.app.state.service.gpu_gate.locked())
        self.assertFalse(list(self.settings.temp_root.iterdir()))

    def test_reference_client_chunked_upload_and_wav_finalization(self):
        source, dest = self.root / "input.wav", self.root / "output.wav"
        source.write_bytes(b"audio")
        progress = []
        done = client_infer(self.live.url, source, dest, {"model_id": "B"}, token="test-only-token", on_progress=progress.append)
        self.assertEqual([p["current"] for p in progress], list(range(1, 7)))
        with wave.open(str(dest), "rb") as value:
            self.assertEqual(value.getnframes(), done["samples"])
        ProbeEngine.fail_chunk = 2
        with self.assertRaises(RuntimeError):
            client_infer(self.live.url, source, self.root / "failed.wav", {"model_id": "A"}, token="test-only-token")
        self.assertFalse((self.root / "failed.wav.partial").exists())
        self.assertFalse((self.root / "failed.wav").exists())
        self.clean()

    def test_inference_timeout_has_failure_terminal_and_cleanup(self):
        self.app.state.service.settings = replace(self.settings, inference_timeout=0.15)
        with self.stream() as response:
            parts = list(events(response))
        self.assertEqual(parts[-1][0], "error")
        self.assertEqual(parts[-1][2]["error"]["code"], "REQUEST_TIMEOUT")
        self.assertNotIn("done", [p[0] for p in parts])
        self.assertLess(parts[-1][2]["current"], parts[-1][2]["total"])
        self.clean()


class ObservedEngine(InferenceEngine):
    instances = []
    fail_chunk = 0

    def __init__(self, *args):
        super().__init__(*args)
        self.times = []
        self.instances.append(self)

    def render(self, number):
        if number == self.fail_chunk:
            # A real CUDA allocation failure, while this request owns models.
            torch.empty(10**12, dtype=torch.float32, device=self.device)
        block = super().render(number)
        self.times.append((number, time.monotonic()))
        return block


class RealNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.artifacts = Path(__file__).parent / "artifacts"
        cls.artifacts.mkdir(exist_ok=True)
        cls.source = cls.artifacts / "live-input.wav"
        sample_audio(cls.source, seconds=6.17)
        cls.settings = Settings(bearer_token="test-only-token")
        cls.app = create_app(cls.settings, ObservedEngine)
        cls.live = LiveServer(cls.app)
        cls.client = httpx.Client(base_url=cls.live.url, timeout=120,
                                  headers={"Authorization": "Bearer test-only-token"})
        cls.reports = []

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.live.close()
        (cls.artifacts / "live-results.json").write_text(json.dumps(cls.reports, ensure_ascii=False, indent=2), encoding="utf-8")

    def setUp(self):
        ObservedEngine.instances, ObservedEngine.fail_chunk = [], 0

    def upload(self):
        with self.source.open("rb") as source:
            while block := source.read(1007):
                yield block

    def clean(self):
        wait_for(lambda: not self.app.state.service.gpu_gate.locked() and not self.app.state.service.jobs, 30)
        self.assertEqual(torch.cuda.memory_allocated(), 0)
        self.assertEqual(torch.cuda.memory_reserved(), 0)
        self.assertFalse(list(self.settings.temp_root.iterdir()))

    def test_real_chinese_stream_progress_and_repeated_cleanup(self):
        for model in ("芙宁娜", "guanguanV1", "芙宁娜"):
            submitted = time.monotonic()
            baseline = torch.cuda.memory_allocated()
            self.assertEqual(self.client.get("/api/v1/init").status_code, 200)
            self.assertEqual(torch.cuda.memory_allocated(), baseline)
            pcm, progress, first = [], [], None
            with self.client.stream("POST", "/api/v1/infer", params={"model_id": model, "chunk_seconds": 1,
                                     "f0_method": "rmvpe", "resample_sr": 24000}, content=self.upload(),
                                    headers={"Content-Type": "audio/wav"}) as response:
                self.assertEqual(response.status_code, 200, response.read().decode("utf-8") if response.status_code != 200 else "")
                for event, headers, value in events(response):
                    if event == "start":
                        start = value
                    elif event == "audio":
                        first = first or time.monotonic()
                        self.assertEqual(int(headers["x-sample-offset"]), sum(len(b) // 2 for b in pcm))
                        pcm.append(value)
                    elif event == "progress":
                        progress.append(value["current"])
                    elif event == "done":
                        done = value
                    elif event == "error":
                        self.fail(str(value))
            self.assertEqual(progress, list(range(1, 8)))
            self.assertEqual(done["samples"], round(6.17 * 24000))
            self.assertEqual(start["parameters"]["model_id"], model)
            self.assertTrue(start["index_used"])
            last_completed = ObservedEngine.instances[-1].times[-1][1]
            self.assertLess(first, last_completed)
            self.assertTrue(done["gpu_cleanup_completed"])
            self.clean()
            with wave.open(str(self.artifacts / (model + "-live.wav")), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(24000)
                destination.writeframes(b"".join(pcm))
            values = np.frombuffer(b"".join(pcm), dtype="<i2").astype("int32")
            difference = np.abs(np.diff(values))
            boundaries = np.cumsum([len(b) // 2 for b in pcm])[:-1]
            seam_step = int(max(difference[max(0, b - 1)] for b in boundaries))
            report = {"model": model, "first_chunk_seconds": first - submitted, "first_received": first,
                      "last_chunk_completed": last_completed, "chunks": len(pcm), "samples": done["samples"],
                      "progress": progress, "cleanup_allocated": done["cuda_allocated_bytes"],
                      "cleanup_reserved": done["cuda_reserved_bytes"], "max_seam_step": seam_step,
                      "signal_p99_step": float(np.percentile(difference, 99))}
            self.reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)

    def test_real_cuda_oom_midstream_releases_all_refs(self):
        ObservedEngine.fail_chunk = 2
        with self.client.stream("POST", "/api/v1/infer", params={"model_id": "芙宁娜", "chunk_seconds": 1},
                                content=self.upload(), headers={"Content-Type": "audio/wav"}) as response:
            parts = list(events(response))
        self.assertEqual(parts[-1][0], "error")
        self.assertEqual(parts[-1][2]["error"]["code"], "GPU_OUT_OF_MEMORY")
        self.assertEqual(parts[-1][2]["current"], 1)
        self.assertNotIn("done", [p[0] for p in parts])
        self.clean()

    def test_real_disconnect_stops_next_chunks_and_cleans(self):
        with self.client.stream("POST", "/api/v1/infer", params={"model_id": "芙宁娜", "chunk_seconds": 1},
                                content=self.upload(), headers={"Content-Type": "audio/wav"}) as response:
            for event, _, _ in events(response):
                if event == "audio":
                    break
        self.clean()
        self.assertTrue(ObservedEngine.instances[-1].closed)
        self.assertLess(len(ObservedEngine.instances[-1].times), 7)


if __name__ == "__main__":
    unittest.main()
