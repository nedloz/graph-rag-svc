"""`job.rag` из результата пайплайна — в той форме, которую читают потребители.

Ключи прежние (их собирал сборщик контекста старого ретривала): `app/analytics/retrieval_log.record`
пишет их в `rag.retrieval_runs`, `app/generation/sinks` отдаёт в `message_completed` (chat-svc
сохраняет в `chat.rag_runs`) и в синхронный ответ боту соседней команды. Менять ключи — значит
менять контракт с обоими.
"""

from app.pipeline.index import Candidate
from app.pipeline.service import RetrievalResult

ROUTE_QUICK_ANSWER = "quick_answer"
ROUTE_HYBRID = "hybrid"
CHUNK_SOURCE = "hybrid"


def _score(item: Candidate) -> float:
    # Оценка чанка — rerank-оценка LLM; у соседних чанков, добавленных к попаданиям, она 0.
    return round(float(item.rerank_score or 0.0), 4)


def _chunk(item: Candidate) -> dict:
    return {
        "chunk_id": str(item.chunk_id),
        "doc_id": str(item.document_id),
        "text": item.text,
        "score": _score(item),
        "source": CHUNK_SOURCE,
        "doc_title": item.title,
    }


def _sources(selected: list[Candidate]) -> list[dict]:
    """Документы-источники с названиями, в порядке первого появления.

    UUID сохраняются обязательно: по ним следующий этап (фронтенд или бот) строит ссылки на
    документы; название — для человека, его знает только этот сервис.
    """
    by_doc: dict[str, dict] = {}
    for item in selected:
        doc_id = str(item.document_id)
        entry = by_doc.get(doc_id)
        if entry is None:
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "title": item.title,
                "chunk_ids": [str(item.chunk_id)],
                "score": _score(item),
            }
            continue
        entry["chunk_ids"].append(str(item.chunk_id))
        entry["score"] = max(entry["score"], _score(item))
    return list(by_doc.values())


def build_rag_payload(result: RetrievalResult) -> dict:
    """`job.rag` для обоих путей: быстрый ответ (чанков нет) и гибридный поиск."""
    selected = result.selected
    return {
        "chunks": [_chunk(item) for item in selected],
        "query_embedding_model": result.query_model,
        "filters_json": None,
        "retrieved_chunk_ids": [str(item.chunk_id) for item in selected],
        "retrieved_doc_ids": list(dict.fromkeys(str(item.document_id) for item in selected)),
        "scores_json": [_score(item) for item in selected],
        "retrieval_ms": result.retrieval_ms,
        "sources": _sources(selected),
        "route_used": ROUTE_QUICK_ANSWER if result.quick_answer is not None else ROUTE_HYBRID,
    }
