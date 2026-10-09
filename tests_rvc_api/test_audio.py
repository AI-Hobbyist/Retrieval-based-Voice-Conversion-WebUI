from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from rvc_api.audio import decode
from rvc_api.config import Settings
from rvc_api.errors import APIError


class DecodeTests(unittest.TestCase):
    def test_real_decoder_limits_short_invalid_and_stereo(self):
        settings = Settings()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "中文音频.wav"
            signal = np.sin(np.arange(16000) * 0.03).astype("float32") * 0.1
            sf.write(path, np.column_stack([signal, signal * 0.5]), 16000)
            decoded = decode(path, settings)
            self.assertEqual(len(decoded), 16000)
            self.assertTrue(np.isfinite(decoded).all())
            self.assertGreater(np.abs(decoded).max(), 0.01)
            with self.assertRaises(APIError) as caught:
                decode(path, replace(settings, max_audio_seconds=0.2))
            self.assertEqual(caught.exception.code, "AUDIO_LIMIT_EXCEEDED")
            sf.write(path, signal[:800], 16000)
            with self.assertRaises(APIError) as caught:
                decode(path, settings)
            self.assertEqual(caught.exception.code, "AUDIO_TOO_SHORT")
            path.write_bytes(b"invalid audio")
            with self.assertRaises(APIError) as caught:
                decode(path, settings)
            self.assertEqual(caught.exception.code, "INVALID_AUDIO")
