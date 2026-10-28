import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.metrics import GENERATE_REQUESTS
from app.api.schemas import CancelRequest, GenerateRequest
from app.core.config import settings
from app.core.internal_auth import require_internal_service
from app.generation.job_queue import bridge, cancellations, job_queue
from app.generation.sinks import BufferSink, ChatBridgeSink, SinkError

logger = logging.getLogger("graph-rag-svc.api.http")

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/ping")
async def ping():
    return {
        "status": "ok",
        "service": settings.service_name,
        "llm_base_url": settings.llm_base_url,
        "chat_bridge_connected": bridge.connected.is_set(),
        "queue_size": job_queue.queue.qsize(),
        "workers": settings.llm_http_concurrency,
        "active_jobs": cancellations.active,
    }


@router.post("/generate")
async def generate(req: GenerateRequest, caller: str = Depends(require_internal_service)):
    """Генерация ответа в одном из двух режимов доставки.

    `sync` — ответ возвращается прямо здесь, в HTTP-ответе (потребители без chat-svc).
    `ws_bridge` — ответ уходит потоком в chat-svc, здесь возвращается только подтверждение
    приёма. Оба режима проходят через одну очередь, поэтому ограничение параллелизма общее.
    """
    if job_queue.queue.full():
        raise HTTPException(status_code=429, detail="queue is full")

    GENERATE_REQUESTS.labels(mode=req.response_mode).inc()

    # Чей это трафик — берём из проверенного заголовка, а не из тела запроса.
    req.consumer = caller

    if req.response_mode == "ws_bridge":
        position = await job_queue.enqueue(req, ChatBridgeSink(bridge))
        return {
            "status": "accepted",
            "request_id": req.request_id,
            "assistant_message_id": req.delivery.assistant_message_id,
            "queue_size": position,
        }

    sink = BufferSink()
    await job_queue.enqueue(req, sink)
    try:
        return await sink.wait(settings.sync_response_timeout_sec)
    except SinkError as exc:
        # Ошибка генерации или ретривала. Ответ без контекста здесь не отдаётся сознательно.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"генерация не уложилась в {settings.sync_response_timeout_sec} с",
        ) from exc


@router.post("/cancel")
async def cancel(req: CancelRequest, _: str = Depends(require_internal_service)):
    """Отмена задания по `request_id`.

    Работает и для уже идущей генерации, и для стоящей в очереди. Отмена кооперативная:
    активное задание останавливается между фрагментами, поэтому накопленный текст остаётся
    сохранённым, а не теряется посреди записи.
    """
    cancelled = cancellations.cancel(req.request_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="задание не найдено (уже завершено?)")
    return {"status": "cancelling", "request_id": req.request_id}
