"""ChatBridge (WS client into chat-svc's /ws/internal/llm) + JobQueue + process_job —
ported from services/llm-svc/app/main.py, extended with an internal retrieval step
(app/retrieval/orchestrator.py) that llm-svc never had: previously a separate rag-svc
called llm-svc's /generate with `rag.chunks` already populated, but rag-svc is gone and
this service now does its own retrieval before generation (see graph-rag-svc README,
"Онлайн-пайплайн"). The /ws/internal/llm event contract itself (generation_started /
stream_chunk / message_completed / message_error) is unchanged, so chat-svc needs no
changes to consume it.
"""

import asyncio
import json
import logging
from contextlib import suppress
from typing import Any

import websockets

from app.analytics import retrieval_log
from app.api.schemas import GenerateRequest
from app.core.config import settings
from app.core.internal_auth import get_internal_auth_headers
from app.db.session import SessionLocal
from app.generation.llm_backends import stream_model
from app.generation.prompt_builder import build_messages
from app.generation.sinks import Sink
from app.mtls import ssl_context
from app.retrieval.orchestrator import retrieve

logger = logging.getLogger("graph-rag-svc.generation.job_queue")



def _tls_kwargs() -> dict[str, Any]:
    """Клиентский сертификат для WebSocket в chat-svc (app/mtls.py); {} без mTLS.

    Контекст собирается при каждом подключении, а подключение переустанавливается в цикле
    _run при обрыве, так что обновлённый суточный сертификат подхватывается без перезапуска.
    """
    ctx = ssl_context()
    return {"ssl": ctx} if ctx is not None else {}


class ChatBridge:
    def __init__(self) -> None:
        self.ws = None
        self.lock = asyncio.Lock()
        self.connected = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.worker_task and not self.worker_task.done():
            return
        self.worker_task = asyncio.create_task(self._run())

        # Диагностика при запуске. Недоступность chat-svc намеренно НЕ мешает стартовать:
        # зависимость взаимная (chat-svc зовёт наш HTTP, мы подключаемся к его WS), и
        # жёсткий отказ стартовать создал бы цикл при подъёме стека. Гарантию от зависания
        # даёт таймаут в send_json(), а не отказ старта — но знать о проблеме нужно сразу,
        # а не в момент первого вопроса пользователя.
        try:
            await asyncio.wait_for(self.connected.wait(), timeout=settings.bridge_startup_wait_sec)
            logger.info("Мост с chat-svc установлен")
        except asyncio.TimeoutError:
            logger.warning(
                "Мост с chat-svc (%s) не поднялся за %s с. Сервис стартует, переподключение идёт "
                "в фоне; доставка ответов через мост будет падать, пока соединения нет.",
                settings.chat_ws_url,
                settings.bridge_startup_wait_sec,
            )

    async def stop(self) -> None:
        self.stop_event.set()
        if self.worker_task:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker_task
        await self._close_ws()

    async def _close_ws(self) -> None:
        self.connected.clear()
        if self.ws is not None:
            with suppress(Exception):
                await self.ws.close()
        self.ws = None

    async def _run(self) -> None:
        headers = get_internal_auth_headers()
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(settings.chat_ws_url, additional_headers=headers, ping_interval=20, ping_timeout=20, **_tls_kwargs()) as ws:
                    self.ws = ws
                    self.connected.set()
                    logger.info("Connected to chat-svc internal websocket")
                    async for raw in ws:
                        logger.debug("chat-svc internal event: %s", raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected.clear()
                logger.warning("Internal websocket disconnected: %s", exc)
                await asyncio.sleep(2)
            finally:
                await self._close_ws()

    async def send_json(self, payload: dict[str, Any]) -> None:
        # Ограниченное ожидание вместо бесконечного: раньше при недоступном chat-svc воркер
        # вставал здесь навсегда — уже на generation_started, то есть до всякой полезной
        # работы, — и при LLM_HTTP_CONCURRENCY=1 останавливалась вся очередь.
        try:
            await asyncio.wait_for(self.connected.wait(), timeout=settings.bridge_send_timeout_sec)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Мост с chat-svc недоступен дольше {settings.bridge_send_timeout_sec} с"
            ) from exc
        async with self.lock:
            if self.ws is None:
                raise RuntimeError("chat bridge is not connected")
            await self.ws.send(json.dumps(payload, ensure_ascii=False))


class JobQueue:
    def __init__(self) -> None:
        # Очередь хранит пару «задание + приёмник результата»: параллелизм ограничен одинаково
        # для синхронных и потоковых запросов, иначе очередь ничего не гарантирует.
        self.queue: asyncio.Queue[tuple[GenerateRequest, Sink]] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.workers: list[asyncio.Task] = []

    async def start(self) -> None:
        if self.workers:
            return
        for idx in range(settings.llm_http_concurrency):
            self.workers.append(asyncio.create_task(self._worker(idx)))

    async def stop(self) -> None:
        for worker in self.workers:
            worker.cancel()
        for worker in self.workers:
            with suppress(asyncio.CancelledError):
                await worker
        self.workers.clear()

    async def enqueue(self, item: GenerateRequest, sink: Sink) -> int:
        cancellations.register(item.request_id)
        await self.queue.put((item, sink))
        return self.queue.qsize()

    async def _worker(self, idx: int) -> None:
        logger.info("Worker %s started", idx)
        while True:
            job, sink = await self.queue.get()
            cancel_event = cancellations.get(job.request_id)
            try:
                # Задание могли отменить, пока оно стояло в очереди — тогда не начинаем вовсе.
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("Задание %s отменено до начала обработки", job.request_id)
                    with suppress(Exception):
                        await sink.cancelled(job, {})
                else:
                    await process_job(job, sink, cancel_event)
            except Exception as exc:
                logger.exception("Job failed: %s", exc)
                with suppress(Exception):
                    await sink.error(job, str(exc))
            finally:
                cancellations.discard(job.request_id)
                self.queue.task_done()


class CancellationRegistry:
    """Отмена активных и ещё не начатых заданий.

    Отмена кооперативная: событие проверяется между фрагментами генерации, поэтому задание
    завершается штатно (с уже накопленным текстом), а не обрывается посреди записи в БД.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    def register(self, request_id: str) -> asyncio.Event:
        event = asyncio.Event()
        self._events[request_id] = event
        return event

    def get(self, request_id: str) -> asyncio.Event | None:
        return self._events.get(request_id)

    def cancel(self, request_id: str) -> bool:
        event = self._events.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def discard(self, request_id: str) -> None:
        self._events.pop(request_id, None)

    @property
    def active(self) -> int:
        return len(self._events)


bridge = ChatBridge()
cancellations = CancellationRegistry()
job_queue = JobQueue()


async def process_job(job: GenerateRequest, sink: Sink, cancel_event: asyncio.Event | None = None) -> None:
    """Один и тот же пайплайн для всех режимов доставки — различается только приёмник."""
    await sink.started(job)

    # Провал ретривала — ошибка всей генерации, а не повод ответить без контекста: ответ на
    # официальный вопрос без источников хуже честной ошибки, и раньше пользователь никак не
    # мог отличить одно от другого.
    try:
        job.rag = await retrieve(SessionLocal, job.question, job.profile or {}, job.history)
    except Exception as exc:
        logger.exception("Retrieval failed for request %s", job.request_id)
        await retrieval_log.record(job, status="error", error=f"retrieval: {exc}")
        raise RuntimeError("Не удалось получить контекст для ответа") from exc

    usage: dict[str, Any] = {}
    messages = build_messages(job)
    async for event in stream_model(messages, usage=usage):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Генерация %s отменена пользователем", job.request_id)
            await sink.cancelled(job, usage)
            await retrieval_log.record(job, usage, status="cancelled")
            return
        if event["type"] == "chunk":
            await sink.chunk(job, event["content"])
        elif event["type"] == "error":
            await sink.error(job, event["content"])
            await retrieval_log.record(job, usage, status="error", error=f"generation: {event['content']}")
            return

    await sink.completed(job, usage)
    await retrieval_log.record(job, usage)
