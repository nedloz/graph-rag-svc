"""build_rag_payload: `job.rag` из RetrievalResult — ключи, которые читают аналитика и приёмники."""

import uuid

from app.pipeline.index import Candidate
from app.pipeline.quick_answers import QuickAnswer
from app.pipeline.rag_payload import build_rag_payload
from app.pipeline.service import RetrievalResult

DOC_A = uuid.UUID(int=1)
DOC_B = uuid.UUID(int=2)

# Ключи, которые читают app/analytics/retrieval_log.record и app/generation/sinks._retrieval_meta.
CONSUMED_KEYS = {
    "chunks",
    "query_embedding_model",
    "filters_json",
    "retrieved_chunk_ids",
    "retrieved_doc_ids",
    "scores_json",
    "retrieval_ms",
    "sources",
    "route_used",
}


def _cand(idx: int, doc: uuid.UUID, title: str, score: float, text: str = "текст") -> Candidate:
    return Candidate(
        idx=idx, chunk_id=uuid.UUID(int=100 + idx), document_id=doc, chunk_index=idx,
        title=title, category="Общие документы", section="Раздел", text=text, rerank_score=score,
    )


def _result(selected, quick=None) -> RetrievalResult:
    return RetrievalResult(
        quick_answer=quick, messages=[], selected=selected, links=[], rerank_tokens=None,
        query_model="e5-base", retrieval_ms=12.5, search_query="q",
    )


def test_hybrid_payload_maps_selected_chunks():
    selected = [
        _cand(0, DOC_A, "Положение о пересдачах", 0.91234, "Пересдача не более двух раз."),
        _cand(1, DOC_A, "Положение о пересдачах", 0.0),  # сосед: rerank-оценки нет
        _cand(2, DOC_B, "Стипендии", 0.5),
    ]

    rag = build_rag_payload(_result(selected))

    assert CONSUMED_KEYS <= set(rag)
    assert rag["route_used"] == "hybrid"
    assert rag["query_embedding_model"] == "e5-base"
    assert rag["filters_json"] is None
    assert rag["retrieval_ms"] == 12.5
    assert rag["retrieved_chunk_ids"] == [str(uuid.UUID(int=100 + i)) for i in range(3)]
    assert rag["retrieved_doc_ids"] == [str(DOC_A), str(DOC_B)]
    assert rag["scores_json"] == [0.9123, 0.0, 0.5]
    assert rag["chunks"][0] == {
        "chunk_id": str(uuid.UUID(int=100)),
        "doc_id": str(DOC_A),
        "text": "Пересдача не более двух раз.",
        "score": 0.9123,
        "source": "hybrid",
        "doc_title": "Положение о пересдачах",
    }
    assert rag["sources"] == [
        {
            "doc_id": str(DOC_A),
            "title": "Положение о пересдачах",
            "chunk_ids": [str(uuid.UUID(int=100)), str(uuid.UUID(int=101))],
            "score": 0.9123,
        },
        {"doc_id": str(DOC_B), "title": "Стипендии", "chunk_ids": [str(uuid.UUID(int=102))], "score": 0.5},
    ]


def test_hybrid_payload_with_nothing_selected():
    rag = build_rag_payload(_result([]))
    assert rag["route_used"] == "hybrid"
    assert rag["chunks"] == [] and rag["sources"] == []
    assert rag["retrieved_chunk_ids"] == [] and rag["retrieved_doc_ids"] == [] and rag["scores_json"] == []


def test_quick_answer_payload():
    quick = QuickAnswer(answer="Закажите в ЛК.", question="Как получить справку?", rrf_score=0.041)

    rag = build_rag_payload(_result([], quick=quick))

    assert CONSUMED_KEYS <= set(rag)
    assert rag["route_used"] == "quick_answer"
    assert rag["query_embedding_model"] == "e5-base"
    assert rag["retrieval_ms"] == 12.5
    assert rag["chunks"] == [] and rag["sources"] == []
