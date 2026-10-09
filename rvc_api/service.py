import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import queue
import threading
import time
import traceback

from .engine import InferenceEngine
from .errors import APIError

logger = logging.getLogger("rvc_api")


def classified(exc):
    if isinstance(exc, APIError):
        return exc
    if "out of memory" in str(exc).lower():
        return APIError("GPU_OUT_OF_MEMORY", "显存不足", 503)
    if isinstance(exc, (ImportError, FileNotFoundError)):
        return APIError("DEPENDENCY_UNAVAILABLE", "推理依赖或辅助资源缺失", 503)
    return APIError("INFERENCE_FAILED", "推理失败；请按 request_id 检查服务日志", 500)


class Job:
    def __init__(self, service, request_id, selection, path, loop):
        self.service, self.request_id, self.selection, self.path = service, request_id, selection, path
        self.loop = loop
        self.ready = loop.create_future()
        self.queue = queue.Queue(maxsize=2)
        self.cancel = threading.Event()
        self.complete = threading.Event()
        self.started = time.monotonic()
        self.sent = 0
        self.max_queue_depth = 0
        self.future = service.executor.submit(self.run)

    def publish_ready(self, result=None, error=None):
        def publish():
            if not self.ready.done():
                if error:
                    self.ready.set_exception(error)
                else:
                    self.ready.set_result(result)
        self.loop.call_soon_threadsafe(publish)

    def stopped(self):
        return self.cancel.is_set() or time.monotonic() - self.started > self.service.settings.inference_timeout

    def put(self, value):
        while not self.cancel.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
                return
            except queue.Full:
                if self.stopped():
                    raise APIError("REQUEST_TIMEOUT", "推理或响应超时", 408)

    def run(self):
        engine = None
        error = None
        cleanup = None
        last_block = None
        prepared = False
        completed_chunks = 0
        try:
            engine = self.service.engine_factory(self.selection, self.service.settings, self.path)
            metadata = engine.prepare()
            metadata["request_id"] = self.request_id
            self.publish_ready(metadata)
            prepared = True
            for number in range(1, metadata["total"] + 1):
                if self.stopped():
                    raise APIError("REQUEST_TIMEOUT", "请求已取消或超时", 408)
                block = engine.render(number)
                completed_chunks = number
                logger.info("%s chunk=%s/%s completed=%.6f", self.request_id, number, metadata["total"], time.monotonic())
                if number == metadata["total"]:
                    last_block = block
                else:
                    self.put(("audio", block))
        except Exception as exc:
            logger.exception("%s inference failed", self.request_id)
            error = classified(exc)
            # Tracebacks may own models/tensors from failed vc/get_f0 frames.
            traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None
            error.__traceback__ = None
            error.__cause__ = None
            error.__context__ = None
        finally:
            try:
                if engine:
                    cleanup = engine.close()
            except Exception as exc:
                logger.exception("%s GPU cleanup failed", self.request_id)
                self.service.ready = False
                error = APIError("GPU_CLEANUP_FAILED", "GPU 清理失败，服务暂不可用", 503)
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            # All queued payloads are CPU-only. Release GPU session before
            # potentially blocking last audio/terminal sends to a slow peer.
            engine = None
            self.service.gpu_gate.release()
        try:
            if not prepared:
                self.publish_ready(error=error or APIError("INFERENCE_FAILED", "推理初始化失败", 500))
            elif error:
                self.put(("error", {"error": {"code": error.code, "message": error.message},
                                    "status": "failed", **(cleanup or {"gpu_cleanup_completed": False})}))
            else:
                self.put(("audio", last_block))
                self.put(("done", {"status": "completed", "samples": last_block["offset"] + last_block["samples"],
                                   "parameters": self.selection.params.dict(), **cleanup}))
        except Exception:
            self.cancel.set()
        finally:
            logger.info("%s finished chunks=%s max_queue=%s cleanup=%s", self.request_id, completed_chunks,
                        self.max_queue_depth, cleanup)
            self.complete.set()


class InferenceService:
    def __init__(self, settings, engine_factory=InferenceEngine):
        self.settings, self.engine_factory = settings, engine_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rvc-infer")
        self.gpu_gate = threading.Lock()
        self.stream_slots = threading.BoundedSemaphore(4)
        self.ready = True
        self.jobs = set()

    def reserve(self):
        if not self.ready:
            raise APIError("DEPENDENCY_UNAVAILABLE", "服务清理失败或正在关闭", 503)
        if not self.stream_slots.acquire(blocking=False):
            raise APIError("INFERENCE_BUSY", "响应容量已满", 429)
        if not self.gpu_gate.acquire(blocking=False):
            self.stream_slots.release()
            raise APIError("INFERENCE_BUSY", "已有推理请求运行中", 429)

    def submit(self, request_id, selection, path):
        job = Job(self, request_id, selection, path, asyncio.get_running_loop())
        self.jobs.add(job)
        return job

    async def finish(self, job):
        job.cancel.set()
        await asyncio.shield(asyncio.wrap_future(job.future))
        self.jobs.discard(job)

    async def close(self):
        self.ready = False
        for job in tuple(self.jobs):
            job.cancel.set()
        for job in tuple(self.jobs):
            await asyncio.shield(asyncio.wrap_future(job.future))
        self.executor.shutdown(wait=True)
