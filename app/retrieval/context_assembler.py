"""Turns a list of RetrievedChunk into the `rag` dict shape process_job hands to
app/generation/prompt_builder.py (`{"chunks": [...]}`, method-agnostic — build_messages
doesn't care whether a chunk came from graph traversal or vector search) and, via
message_completed, back to chat-svc for persistence in chat.rag_runs.
"""

from app.retrieval.types import RetrievedChunk


def assemble(
    chunks: list[RetrievedChunk],
    *,
    retrieval_ms: float,
    query_embedding_model: str | None = None,
    route_used: str | None = None,
) -> dict:
    retrieved_chunk_ids = [str(c.chunk_id) for c in chunks]
    retrieved_doc_ids = list(dict.fromkeys(str(c.doc_id) for c in chunks))  # dedup, keep order
    scores = [c.score for c in chunks]

    return {
        "chunks": [c.to_dict() for c in chunks],
        "query_embedding_model": query_embedding_model,
        "filters_json": None,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "scores_json": scores,
        "retrieval_ms": retrieval_ms,
        "sources": _build_sources(chunks),
        # "graph" | "vector" | "graph->vector" (сработал защитный откат)
        "route_used": route_used,
    }


def _build_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """Сворачивает чанки в список документов-источников с их названиями.

    Названия собираются здесь, последним этапом перед отправкой, потому что только этот
    сервис имеет доступ к library.documents. UUID при этом сохраняются обязательно: по ним
    следующий этап обработки (фронтенд или бот) сам решает, как строить ссылки на документы.
    """
    by_doc: dict[str, dict] = {}
    for chunk in chunks:
        doc_id = str(chunk.doc_id)
        entry = by_doc.get(doc_id)
        if entry is None:
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "title": chunk.doc_title,
                "chunk_ids": [str(chunk.chunk_id)],
                "score": chunk.score,
            }
            continue
        entry["chunk_ids"].append(str(chunk.chunk_id))
        entry["score"] = max(entry["score"], chunk.score)
    return list(by_doc.values())
