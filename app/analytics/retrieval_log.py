"""Запись аналитики ретривала в нейтральную схему `rag`.

Зачем отдельно от `chat.rag_runs`: ту таблицу пишет chat-svc, а у соседнего проекта
chat-svc нет вовсе. Обе команды пишут сюда, колонка `consumer` разделяет трафик.
`chat.rag_runs` при этом сохраняется — там есть связь с `message_id`, которой у бота нет
(README, §12.3).

Запись аналитики никогда не должна ломать ответ пользователю: любая ошибка здесь только
логируется.
"""

import logging
import uuid
from typing import Any

from app.api.schemas import GenerateRequest
from app.db.models import RetrievalRun
from app.db.session import SessionLocal

logger = logging.getLogger("graph-rag-svc.analytics.retrieval_log")


def _uuid_list(values: Any) -> list[uuid.UUID] | None:
    if not values:
        return None
    result: list[uuid.UUID] = []
    for value in values:
        try:
            result.append(uuid.UUID(str(value)))
        except (ValueError, AttributeError, TypeError):
            continue
    return result or None


async def record(
    job: GenerateRequest,
    usage: dict[str, Any] | None = None,
    *,
    status: str = "completed",
    error: str | None = None,
) -> None:
    rag = job.rag or {}
    usage = usage or {}
    try:
        async with SessionLocal() as session:
            session.add(
                RetrievalRun(
                    request_id=job.request_id,
                    consumer=job.consumer,
                    session_id=job.session_id,
                    user_id=job.user_id,
                    question=job.question,
                    route_used=rag.get("route_used"),
                    query_embedding_model=rag.get("query_embedding_model"),
                    retrieved_chunk_ids=_uuid_list(rag.get("retrieved_chunk_ids")),
                    retrieved_doc_ids=_uuid_list(rag.get("retrieved_doc_ids")),
                    scores_json={"scores": rag.get("scores_json")} if rag.get("scores_json") else None,
                    retrieval_ms=rag.get("retrieval_ms"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    status=status,
                    error=error,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("Не удалось записать аналитику ретривала для запроса %s", job.request_id)
