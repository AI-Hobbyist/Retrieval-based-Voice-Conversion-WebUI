import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rvc_api.client import infer
from rvc_api.protocol import part


class ClientTests(unittest.TestCase):
    def test_done_without_closing_boundary_removes_partial(self):
        class Response(io.BytesIO):
            headers = {"content-type": "multipart/mixed; boundary=test"}

        data = b"".join(part("test", "start", {"codec": "pcm_s16le", "channels": 1,
                                              "sample_rate": 16000, "total": 1}))
        data += b"".join(part("test", "audio", b"\0\0" * 1600,
                              {"X-Chunk-Index": "1", "X-Sample-Offset": "0", "X-Sample-Count": "1600"}))
        data += b"".join(part("test", "done", {"current": 1, "samples": 1600, "gpu_cleanup_completed": True}))
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder) / "in.wav", Path(folder) / "out.wav"
            source.write_bytes(b"audio")
            with patch("urllib.request.urlopen", return_value=Response(data)):
                with self.assertRaises(ValueError):
                    infer("http://127.0.0.1", source, destination, {"model_id": "A"})
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("out.wav.partial").exists())
