"""Приёмники результата генерации.

Разделяют «что сервис вычисляет» и «куда он отдаёт результат». Пайплайн (ретривал → сборка
промпта → генерация) один и тот же для всех потребителей; меняется только приёмник. Именно это
позволяет одному сервису обслуживать и наш веб-чат (события потоком в chat-svc через WS-мост),
и Telegram-бот соседней команды (готовый ответ в HTTP-ответе), не заводя второй сервис и не
заставляя бота имитировать chat-svc. См. README, §10-11.
"""

import asyncio
from typing import Any, Protocol

from app.api.schemas import GenerateRequest


class Sink(Protocol):
    """Контракт приёмника. Порядок вызовов: started → (chunk* → completed | error)."""

    async def started(self, job: GenerateRequest) -> None: ...

    async def chunk(self, job: GenerateRequest, delta: str) -> None: ...

    async def completed(self, job: GenerateRequest, usage: dict[str, Any]) -> None: ...

    async def error(self, job: GenerateRequest, message: str) -> None: ...

    async def cancelled(self, job: GenerateRequest, usage: dict[str, Any]) -> None: ...


def _retrieval_meta(job: GenerateRequest) -> dict[str, Any]:
    rag = job.rag or {}
    return {
        "query_embedding_model": rag.get("query_embedding_model"),
        "filters_json": rag.get("filters_json"),
        "retrieved_chunk_ids": rag.get("retrieved_chunk_ids"),
        "retrieved_doc_ids": rag.get("retrieved_doc_ids"),
        "sources": rag.get("sources"),
        "scores_json": rag.get("scores_json"),
        "retrieval_ms": rag.get("retrieval_ms"),
        "route_used": rag.get("route_used"),
    }


class ChatBridgeSink:
    """Доставка в chat-svc по WS-мосту — поведение, которое было до появления режимов.

    Контракт событий (`generation_started` / `stream_chunk` / `message_completed` /
    `message_error`) не менялся, поэтому chat-svc не пришлось переписывать под приёмники.
    """

    def __init__(self, bridge) -> None:
        self.bridge = bridge

    def _envelope(self, job: GenerateRequest) -> dict[str, Any]:
        delivery = job.delivery
        return {
            "request_id": job.request_id,
            "session_id": job.session_id,
            "assistant_message_id": delivery.assistant_message_id if delivery else None,
        }

    async def started(self, job: GenerateRequest) -> None:
        payload = self._envelope(job) | {"type": "generation_started"}
        if job.delivery is not None:
            payload["message_id"] = job.delivery.message_id
        await self.bridge.send_json(payload)

    async def chunk(self, job: GenerateRequest, delta: str) -> None:
        await self.bridge.send_json(self._envelope(job) | {"type": "stream_chunk", "delta": delta})

    async def completed(self, job: GenerateRequest, usage: dict[str, Any]) -> None:
        await self.bridge.send_json(
            self._envelope(job)
            | {"type": "message_completed"}
            | _retrieval_meta(job)
            | {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )

    async def error(self, job: GenerateRequest, message: str) -> None:
        await self.bridge.send_json(self._envelope(job) | {"type": "message_error", "message": message})

    async def cancelled(self, job: GenerateRequest, usage: dict[str, Any]) -> None:
        # Отдельное событие, а не message_error: отмена — это не сбой, и показывать её
        # пользователю нужно иначе. Накопленный текст к этому моменту уже сохранён
        # chat-svc'ом из stream_chunk, поэтому здесь передаётся только факт остановки.
        await self.bridge.send_json(
            self._envelope(job)
            | {"type": "message_cancelled"}
            | _retrieval_meta(job)
            | {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )


class SinkError(Exception):
    """Генерация завершилась ошибкой — для синхронного режима это ответ вызывающей стороне."""


class BufferSink:
    """Копит ответ целиком и отдаёт его в HTTP-ответе — синхронный режим.

    Ожидание реализовано через future, а не обходом очереди: у синхронного и потокового
    режимов должно быть общее ограничение параллелизма, иначе очередь перестаёт что-либо
    гарантировать.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._future: asyncio.Future = asyncio.get_running_loop().create_future()

    async def started(self, job: GenerateRequest) -> None:
        return

    async def chunk(self, job: GenerateRequest, delta: str) -> None:
        self._parts.append(delta)

    async def completed(self, job: GenerateRequest, usage: dict[str, Any]) -> None:
        if self._future.done():
            return
        meta = _retrieval_meta(job)
        self._future.set_result(
            {
                "request_id": job.request_id,
                "answer": "".join(self._parts),
                "route_used": meta.get("route_used"),
                "sources": meta.get("sources") or [],
                "retrieval": {
                    "retrieval_ms": meta.get("retrieval_ms"),
                    "chunks_used": len(meta.get("retrieved_chunk_ids") or []),
                    "retrieved_chunk_ids": meta.get("retrieved_chunk_ids") or [],
                    "retrieved_doc_ids": meta.get("retrieved_doc_ids") or [],
                    "query_embedding_model": meta.get("query_embedding_model"),
                },
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
            }
        )

    async def error(self, job: GenerateRequest, message: str) -> None:
        if not self._future.done():
            self._future.set_exception(SinkError(message))

    async def cancelled(self, job: GenerateRequest, usage: dict[str, Any]) -> None:
        # Синхронному вызывающему отдаём то, что успели сгенерировать, с явным статусом:
        # отмена не ошибка, и обрывать HTTP-запрос исключением было бы неверно.
        if self._future.done():
            return
        await self.completed(job, usage)
        result = self._future.result()
        result["status"] = "cancelled"

    async def wait(self, timeout: float) -> dict[str, Any]:
        return await asyncio.wait_for(asyncio.shield(self._future), timeout=timeout)
