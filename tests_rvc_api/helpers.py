import json
import re


class MultipartReader:
    """Incremental length-framed MIME parser shared by protocol/live tests."""
    def __init__(self, boundary):
        self.marker = b"--" + boundary.encode("ascii")
        self.buffer = bytearray()
        self.headers = None
        self.closed = False

    def feed(self, data):
        self.buffer.extend(data)
        result = []
        while True:
            if self.headers is None:
                if self.buffer.startswith(b"\r\n"):
                    del self.buffer[:2]
                if self.buffer.startswith(self.marker + b"--\r\n"):
                    del self.buffer[:len(self.marker) + 4]
                    self.closed = True
                    break
                end = self.buffer.find(b"\r\n\r\n")
                if end < 0:
                    break
                raw = bytes(self.buffer[:end]).decode("ascii")
                lines = raw.split("\r\n")
                if lines[0].encode("ascii") != self.marker:
                    raise ValueError("bad multipart boundary")
                self.headers = {key.lower(): value.strip() for key, value in
                                (line.split(":", 1) for line in lines[1:])}
                self.length = int(self.headers["content-length"])
                if self.length < 0 or self.length > 8 * 1024 * 1024:
                    raise ValueError("invalid part length")
                del self.buffer[:end + 4]
            if len(self.buffer) < self.length + 2:
                break
            body = bytes(self.buffer[:self.length])
            if self.buffer[self.length:self.length + 2] != b"\r\n":
                raise ValueError("missing part terminator")
            del self.buffer[:self.length + 2]
            result.append((self.headers["x-event-type"], self.headers, body))
            self.headers = None
        return result


def parse_response(response):
    content_type = response.headers["content-type"]
    boundary = re.search(r"boundary=([^;]+)", content_type).group(1)
    reader = MultipartReader(boundary)
    parts = reader.feed(response.content)
    if not reader.closed:
        raise ValueError("incomplete response")
    return [(event, headers, body if event == "audio" else json.loads(body)) for event, headers, body in parts]
