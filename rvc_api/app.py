import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import queue
import tempfile
import time
import uuid
from pathlib import Path
import anyio

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from .config import Settings
from .errors import APIError
from .protocol import closing, part
from .registry import ModelRegistry, is_link
from .schemas import InferParams, PARAMETERS, modes
from .service import InferenceService

logger = logging.getLogger("rvc_api")


def create_app(settings=None, engine_factory=None):
    settings = settings or Settings.from_env()
    registry = ModelRegistry(settings.project_root)
    service = InferenceService(settings, **({"engine_factory": engine_factory} if engine_factory else {}))

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="RVC Streaming Inference", lifespan=lifespan,
                  docs_url=None if settings.bearer_token else "/docs",
                  redoc_url=None if settings.bearer_token else "/redoc",
                  openapi_url=None if settings.bearer_token else "/openapi.json")
    app.state.registry, app.state.service, app.state.settings = registry, service, settings
    bearer = HTTPBearer(auto_error=False)

    async def authorize(request: Request, credentials=Depends(bearer)):
        if settings.bearer_token and (credentials is None or not hmac.compare_digest(
                credentials.credentials.encode("utf-8"), settings.bearer_token.encode("utf-8"))):
            raise APIError("UNAUTHORIZED", "Bearer Token 缺失或不正确", 401)

    @app.exception_handler(APIError)
    async def error_response(request, error):
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        return JSONResponse(error.payload(request_id), status_code=error.status,
                            headers={"WWW-Authenticate": "Bearer"} if error.status == 401 else None)

    @app.get("/health")
    async def health():
        return {"alive": True, "ready": service.ready}

    protected = [Depends(authorize)]

    @app.get("/api/v1/init", dependencies=protected)
    async def initialize():
        catalog = await asyncio.to_thread(registry.scan)
        return {"models": catalog["models"], "models_state": catalog["state"], "issues": catalog["issues"],
                "f0_methods": registry.f0_methods(), "modes": modes(), "parameters": PARAMETERS,
                "auth_required": bool(settings.bearer_token),
                "audio_transport": {"upload": "raw_body_stream", "download": "multipart_mixed_audio_progress",
                                    "codec": "pcm_s16le", "inference_starts": "after_upload_and_preprocess",
                                    "audio_emitted": "after_each_inference_chunk", "progress": "current/total"},
                "limits": {"max_request_bytes": settings.max_request_bytes, "max_audio_seconds": settings.max_audio_seconds,
                           "upload_timeout": settings.upload_timeout, "inference_timeout": settings.inference_timeout,
                           "max_audio_queue": 2, "max_active_responses": 4},
                "gpu_policy": "unload_after_each_request"}

    @app.get("/api/v1/models", dependencies=protected)
    async def models_endpoint():
        return await asyncio.to_thread(registry.scan)

    @app.get("/api/v1/models/{model_id}", dependencies=protected)
    async def model_endpoint(model_id: str):
        try:
            return {**(await asyncio.to_thread(registry.model, model_id)), "parameters": PARAMETERS}
        except ValidationError as exc:
            raise APIError("INVALID_PARAMETER", "模型 ID 无效") from exc

    @app.get("/api/v1/f0-methods", dependencies=protected)
    async def f0_endpoint():
        return {"f0_methods": registry.f0_methods()}

    @app.get("/api/v1/modes", dependencies=protected)
    async def modes_endpoint():
        return modes()

    @app.post("/api/v1/infer", dependencies=protected)
    async def infer(request: Request):
        request.state.request_id = request_id = uuid.uuid4().hex
        try:
            # Explicitly reject repeated and unknown keys, avoiding ambiguous
            # effective parameters when proxies preserve duplicate queries.
            if len(request.query_params.multi_items()) != len(request.query_params):
                raise APIError("INVALID_PARAMETER", "请求参数不能重复")
            params = InferParams(**dict(request.query_params))
        except ValidationError as exc:
            raise APIError("INVALID_PARAMETER", "参数类型、范围或字段不正确") from exc
        media = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if not (media.startswith("audio/") or media == "application/octet-stream"):
            raise APIError("INVALID_AUDIO", "请求体必须是原始音频数据")
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > settings.max_request_bytes:
                    raise APIError("AUDIO_LIMIT_EXCEEDED", "上传超过大小限制", 413)
            except ValueError as exc:
                raise APIError("INVALID_PARAMETER", "Content-Length 无效") from exc
        service.reserve()
        job, temp, transferred = None, None, False
        try:
            selection = await asyncio.to_thread(registry.resolve, params)
            if selection.metadata["supports_f0"]:
                methods = {m["id"]: m for m in registry.f0_methods()}
                if not methods[params.f0_method]["available"]:
                    raise APIError("DEPENDENCY_UNAVAILABLE", "所选 F0 工具不可用", 503)
            settings.temp_root.mkdir(parents=True, exist_ok=True)
            if is_link(settings.temp_root) or settings.temp_root.resolve().parent != (settings.project_root / "rvc_api").resolve():
                raise APIError("INVALID_PARAMETER", "服务临时目录无效")
            temp = tempfile.TemporaryDirectory(prefix="request-", dir=settings.temp_root)
            path = Path(temp.name) / "input.audio"
            size = 0
            async def upload():
                nonlocal size
                with path.open("wb") as target:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > settings.max_request_bytes:
                            raise APIError("AUDIO_LIMIT_EXCEEDED", "上传超过大小限制", 413)
                        await asyncio.to_thread(target.write, chunk)
            try:
                await asyncio.wait_for(upload(), settings.upload_timeout)
            except asyncio.TimeoutError as exc:
                raise APIError("REQUEST_TIMEOUT", "上传超时", 408) from exc
            if size == 0:
                raise APIError("INVALID_AUDIO", "上传音频为空")
            job = service.submit(request_id, selection, path)
            start = await asyncio.wait_for(asyncio.shield(job.ready), settings.inference_timeout)
            boundary = "rvc_" + request_id

            async def response():
                try:
                    for item in part(boundary, "start", start):
                        yield item
                    while True:
                        try:
                            event, payload = await asyncio.to_thread(job.queue.get, True, 0.2)
                        except queue.Empty:
                            if job.complete.is_set():
                                for item in part(boundary, "error", {"status": "failed", "current": job.sent,
                                                                      "total": start["total"], "request_id": request_id,
                                                                      "error": {"code": "STREAM_INTERRUPTED", "message": "响应未完成"}}):
                                    yield item
                                break
                            continue
                        if event == "audio":
                            headers = {"X-Chunk-Index": str(payload["index"]), "X-Sample-Offset": str(payload["offset"]),
                                       "X-Sample-Count": str(payload["samples"])}
                            for item in part(boundary, "audio", payload["pcm"], headers):
                                yield item
                            job.sent += 1
                            progress = {"request_id": request_id, "stage": "infer", "current": job.sent,
                                        "total": start["total"], "status": "running"}
                            for item in part(boundary, "progress", progress):
                                yield item
                        else:
                            payload.update(request_id=request_id, current=job.sent, total=start["total"])
                            for item in part(boundary, event, payload):
                                yield item
                            break
                    yield closing(boundary)
                finally:
                    with anyio.CancelScope(shield=True):
                        try:
                            await service.finish(job)
                        finally:
                            service.stream_slots.release()
                            temp.cleanup()

            transferred = True
            return StreamingResponse(response(), media_type="multipart/mixed; boundary=" + boundary,
                                     headers={"X-Request-ID": request_id, "Cache-Control": "no-store",
                                              "X-Accel-Buffering": "no"})
        except ClientDisconnect as exc:
            raise APIError("REQUEST_CANCELLED", "客户端中断上传", 408) from exc
        except asyncio.TimeoutError as exc:
            raise APIError("REQUEST_TIMEOUT", "推理初始化超时", 408) from exc
        finally:
            if not transferred:
                with anyio.CancelScope(shield=True):
                    try:
                        if job:
                            await service.finish(job)
                        else:
                            service.gpu_gate.release()
                    finally:
                        service.stream_slots.release()
                        if temp:
                            temp.cleanup()

    return app
