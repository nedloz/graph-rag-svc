"""ChatBridge (WS client into chat-svc's /ws/internal/llm) + JobQueue + process_job —
ported from services/llm-svc/app/main.py, extended with an internal retrieval step that
llm-svc never had: previously a separate rag-svc called llm-svc's /generate with
`rag.chunks` already populated, but rag-svc is gone and this service now does its own
retrieval before generation. Ретривал — пайплайн ноутбука (app/pipeline), модель — через
llm-proxy (app/generation/llm_client.py). The /ws/internal/llm event contract itself
(generation_started / stream_chunk / message_completed / message_error) is unchanged, so
chat-svc needs no changes to consume it.
"""

import asyncio
import json
import logging
import time
from contextlib import aclosing, suppress
from typing import Any

import websockets

from app.metrics import (
    CONTEXT_CHUNKS,
    EMPTY_CONTEXT,
    JUDGE_DECISIONS,
    LLM_ERRORS,
    RETRIEVAL_FAILURES,
    STAGE_DURATION,
)
from app.analytics import retrieval_log
from app.api.schemas import GenerateRequest
from app.core.config import settings
from app.db.session import SessionLocal
from app.generation.injection_defense import find_prompt_injection_signals
from app.generation.llm_client import LlmClient
from app.generation.sinks import Sink
from app.mtls import ssl_context
from app.pipeline.index import KnowledgeIndex
from app.pipeline.prompts import ANSWER_MAX_TOKENS, ANSWER_TEMPERATURE
from app.pipeline.rag_payload import build_rag_payload
from app.pipeline.service import RetrievalResult, run_retrieval

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
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(settings.chat_ws_url, ping_interval=20, ping_timeout=20, **_tls_kwargs()) as ws:
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
# Индекс базы знаний на весь процесс: загружается и обновляется из app/main.py
# (app/pipeline/runtime.py), читается каждым заданием через snapshot().
knowledge_index = KnowledgeIndex()
# Один объект на процесс, но HTTP-клиент (и контекст mTLS) он создаёт на каждый вызов.
llm = LlmClient(settings.llm_base_url, settings.llm_model)


class InjectionRefused(RuntimeError):
    """FAIL_CLOSED_ON_INJECTION: в выбранном контексте есть признаки prompt injection."""


def _injection_signals(job: GenerateRequest, result: RetrievalResult) -> list[str]:
    """Проверяет выбранные чанки эвристиками injection_defense и пишет предупреждение по каждому.

    Ответ не меняется: промпт ноутбука сам велит опираться только на материалы. Сигналы нужны,
    чтобы заметить подозрительный документ в базе, а при FAIL_CLOSED_ON_INJECTION — не отвечать.
    """
    found: list[str] = []
    for item in result.selected:
        signals = find_prompt_injection_signals(item.text)
        if not signals:
            continue
        found.extend(signals)
        logger.warning(
            "Признаки prompt injection (%s) в чанке %s документа «%s», запрос %s",
            ", ".join(signals),
            item.chunk_id,
            item.title,
            job.request_id,
        )
    return sorted(set(found))


def _observe_retrieval(result: RetrievalResult, route: str, seconds: float) -> None:
    STAGE_DURATION.labels(stage="retrieval").observe(seconds)
    # Судьи больше нет; метка route — маршрут пайплайна (quick_answer | hybrid), имя метрики
    # прежнее, чтобы не ломать дашборд.
    JUDGE_DECISIONS.labels(route=route).inc()
    # Быстрому ответу контекст не нужен — в статистику контекста он не попадает.
    if result.quick_answer is None:
        CONTEXT_CHUNKS.observe(len(result.selected))
        if not result.selected:
            EMPTY_CONTEXT.inc()


async def process_job(job: GenerateRequest, sink: Sink, cancel_event: asyncio.Event | None = None) -> None:
    """Один и тот же пайплайн для всех режимов доставки — различается только приёмник."""
    await sink.started(job)

    # Провал ретривала — ошибка всей генерации, а не повод ответить без контекста: ответ на
    # официальный вопрос без источников хуже честной ошибки. Сюда же попадают ошибки вызовов
    # LLM в реранке и переписывании вопроса и ошибки embedding-svc.
    retrieval_started = time.perf_counter()
    try:
        result = await run_retrieval(job, llm, knowledge_index, SessionLocal)
    except Exception as exc:
        RETRIEVAL_FAILURES.inc()
        logger.exception("Retrieval failed for request %s", job.request_id)
        await retrieval_log.record(job, status="error", error=f"retrieval: {exc}")
        raise RuntimeError("Не удалось получить контекст для ответа") from exc

    job.rag = build_rag_payload(result)
    _observe_retrieval(result, job.rag["route_used"], time.perf_counter() - retrieval_started)

    # Ретривал теперь сам зовёт LLM и длится секунды: отмену, пришедшую за это время,
    # учитываем до генерации, а не после первого фрагмента ответа.
    if cancel_event is not None and cancel_event.is_set():
        logger.info("Задание %s отменено до генерации", job.request_id)
        await sink.cancelled(job, {})
        await retrieval_log.record(job, {}, status="cancelled")
        return

    # Быстрый ответ — готовый текст из базы, модель не вызывается.
    if result.quick_answer is not None:
        await sink.chunk(job, result.quick_answer.answer)
        await sink.completed(job, {})
        await retrieval_log.record(job, {})
        return

    signals = _injection_signals(job, result)
    if signals and settings.fail_closed_on_injection:
        error = f"injection: {', '.join(signals)}"
        await retrieval_log.record(job, status="error", error=error)
        raise InjectionRefused("Найденный контекст отклонён: признаки prompt injection")

    usage: dict[str, Any] = {}
    generation_started = time.perf_counter()
    try:
        # aclosing: при отмене поток закрывается сразу, вместе с соединением к llm-proxy.
        async with aclosing(
            llm.stream(result.messages, usage=usage, temperature=ANSWER_TEMPERATURE, max_tokens=ANSWER_MAX_TOKENS)
        ) as events:
            async for event in events:
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("Генерация %s отменена пользователем", job.request_id)
                    await sink.cancelled(job, usage)
                    await retrieval_log.record(job, usage, status="cancelled")
                    return
                if event["type"] == "chunk":
                    await sink.chunk(job, event["content"])
                elif event["type"] == "error":
                    LLM_ERRORS.inc()
                    await sink.error(job, event["content"])
                    await retrieval_log.record(job, usage, status="error", error=f"generation: {event['content']}")
                    return
    finally:
        STAGE_DURATION.labels(stage="generation").observe(time.perf_counter() - generation_started)

    await sink.completed(job, usage)
    await retrieval_log.record(job, usage)
