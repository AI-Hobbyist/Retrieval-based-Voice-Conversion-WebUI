import json


def part(boundary, event, payload, headers=None):
    binary = isinstance(payload, bytes)
    body = payload if binary else json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    values = {"Content-Type": "application/octet-stream" if binary else "application/json; charset=utf-8",
              "Content-Length": str(len(body)), "X-Event-Type": event, **(headers or {})}
    yield ("--" + boundary + "\r\n" + "".join(f"{k}: {v}\r\n" for k, v in values.items()) + "\r\n").encode("ascii")
    for offset in range(0, len(body), 65536):
        yield body[offset:offset + 65536]
    yield b"\r\n"


def closing(boundary):
    return ("--" + boundary + "--\r\n").encode("ascii")
