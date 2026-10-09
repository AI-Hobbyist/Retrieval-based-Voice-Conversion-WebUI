"""Streaming reference client: raw upload, MIME PCM/progress, atomic WAV save."""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import wave


class MultipartReader:
    def __init__(self, boundary):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,70}", boundary):
            raise ValueError("Invalid response boundary")
        self.marker = b"--" + boundary.encode("ascii")
        self.buffer = bytearray()
        self.headers = None
        self.closed = False

    def feed(self, data):
        self.buffer.extend(data)
        events = []
        while not self.closed:
            if self.headers is None:
                if self.buffer.startswith(b"\r\n"):
                    del self.buffer[:2]
                if self.buffer.startswith(self.marker + b"--\r\n"):
                    del self.buffer[:len(self.marker) + 4]
                    self.closed = True
                    break
                end = self.buffer.find(b"\r\n\r\n")
                if end < 0:
                    if len(self.buffer) > 16384:
                        raise ValueError("Response headers too large")
                    break
                lines = bytes(self.buffer[:end]).decode("ascii").split("\r\n")
                if lines[0].encode("ascii") != self.marker:
                    raise ValueError("Invalid part boundary")
                self.headers = {key.lower(): value.strip() for key, value in (line.split(":", 1) for line in lines[1:])}
                self.length = int(self.headers["content-length"])
                if not 0 <= self.length <= 8 * 1024 * 1024:
                    raise ValueError("Invalid part length")
                del self.buffer[:end + 4]
            if len(self.buffer) < self.length + 2:
                break
            payload = bytes(self.buffer[:self.length])
            if self.buffer[self.length:self.length + 2] != b"\r\n":
                raise ValueError("Invalid part termination")
            del self.buffer[:self.length + 2]
            events.append((self.headers["x-event-type"], self.headers, payload))
            self.headers = None
        return events


def infer(url, source, destination, parameters, token="", on_progress=None):
    source, destination = Path(source), Path(destination)
    partial = destination.with_name(destination.name + ".partial")

    def upload():
        with source.open("rb") as audio:
            while data := audio.read(65536):
                yield data

    query = urllib.parse.urlencode(parameters)
    headers = {"Content-Type": "application/octet-stream"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url.rstrip("/") + "/api/v1/infer?" + query, data=upload(), headers=headers, method="POST")
    writer = None
    saved = False
    done, start, samples, chunks = None, None, 0, 0
    try:
        with urllib.request.urlopen(request, timeout=700) as response:
            boundary = re.search(r"boundary=([^;]+)", response.headers.get("content-type", ""))
            if boundary is None:
                raise ValueError("Response is not multipart audio/progress")
            reader = MultipartReader(boundary.group(1))
            while data := response.read1(65536):
                for event, part_headers, body in reader.feed(data):
                    if event == "audio":
                        if writer is None or int(part_headers["x-chunk-index"]) != chunks + 1:
                            raise ValueError("Unexpected audio chunk sequence")
                        count = int(part_headers["x-sample-count"])
                        if int(part_headers["x-sample-offset"]) != samples or len(body) != count * 2:
                            raise ValueError("Audio offsets or length are inconsistent")
                        writer.writeframesraw(body)
                        samples += count
                        chunks += 1
                    else:
                        value = json.loads(body)
                        if event == "start":
                            if writer is not None or value["codec"] != "pcm_s16le" or value["channels"] != 1:
                                raise ValueError("Unsupported audio format")
                            start = value
                            writer = wave.open(str(partial), "wb")
                            writer.setnchannels(1)
                            writer.setsampwidth(2)
                            writer.setframerate(value["sample_rate"])
                        elif event == "progress":
                            if value["current"] != chunks or value["total"] != start["total"]:
                                raise ValueError("Progress does not match audio")
                            if on_progress:
                                on_progress(value)
                        elif event == "error":
                            raise RuntimeError(f"{value['error']['code']}: {value['error']['message']}")
                        elif event == "done":
                            if value["current"] != start["total"] or chunks != start["total"] or value["samples"] != samples or not value["gpu_cleanup_completed"]:
                                raise ValueError("Incomplete inference result")
                            done = value
                        else:
                            raise ValueError("Unknown event")
            if not reader.closed or done is None or reader.buffer:
                raise ValueError("Stream ended without a complete done event")
        writer.close()
        writer = None
        partial.replace(destination)
        saved = True
        return done
    finally:
        if writer:
            writer.close()
        if not saved and partial.exists():
            partial.unlink()


def main():
    parser = argparse.ArgumentParser(description="RVC streaming API reference client")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--weight")
    parser.add_argument("--index")
    parser.add_argument("--index-rate", type=float)
    parser.add_argument("--index-mode", choices=["auto", "off", "required"], default="auto")
    parser.add_argument("--f0-method", choices=["pm", "rmvpe", "fcpe"], default="rmvpe")
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--chunk-seconds", type=float, default=5)
    args = parser.parse_args()
    parameters = {"model_id": args.model, "weight_id": args.weight, "index_id": args.index,
                  "index_rate": args.index_rate, "index_mode": args.index_mode, "f0_method": args.f0_method,
                  "pitch_shift": args.pitch_shift, "speaker_id": args.speaker_id, "resample_sr": args.resample_sr,
                  "rms_mix_rate": args.rms_mix_rate, "protect": args.protect, "chunk_seconds": args.chunk_seconds}
    result = infer(args.url, args.input, args.output, {k: v for k, v in parameters.items() if v is not None},
                   token=os.getenv("API_BEARER_TOKEN", ""), on_progress=lambda p: print(f"{p['current']}/{p['total']}", flush=True))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
