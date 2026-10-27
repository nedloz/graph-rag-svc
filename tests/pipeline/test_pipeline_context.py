"""Агрегация по документам, выбор контекста с соседями и сборка текста (ячейка 25)."""

import uuid

import pytest

from app.pipeline.context import aggregate_documents, build_context, select_context
from app.pipeline.index import Candidate

DOC_A = uuid.UUID(int=1)
DOC_B = uuid.UUID(int=2)


def _cand(idx: int, doc: uuid.UUID, chunk_index: int, rerank: float = 0.0, text: str = "текст",
          title: str | None = None) -> Candidate:
    return Candidate(
        idx=idx, chunk_id=uuid.UUID(int=1000 + idx), document_id=doc, chunk_index=chunk_index,
        title=title or f"Док {doc.int}", category="Общие документы", section="Раздел", text=text,
        rerank_score=rerank,
    )


def _no_neighbours(document_id, chunk_index):
    return None


def test_aggregate_documents_weights_four_best_hits():
    ranked = [
        _cand(1, DOC_A, 1, 0.7),
        _cand(2, DOC_A, 2, 0.9),
        _cand(3, DOC_A, 3, 0.5),  # пятый по силе — не учитывается
        _cand(4, DOC_A, 4, 0.8),
        _cand(5, DOC_A, 5, 0.6),
        _cand(6, DOC_B, 1, 0.95),
    ]

    docs = aggregate_documents(ranked)

    assert [d["document_id"] for d in docs] == [DOC_A, DOC_B]
    assert docs[0]["score"] == pytest.approx(0.9 * 1.0 + 0.8 * 0.55 + 0.7 * 0.30 + 0.6 * 0.15)
    assert docs[1]["score"] == pytest.approx(0.95)
    assert [h.idx for h in docs[0]["hits"]] == [2, 4, 1, 5, 3]  # по убыванию rerank_score
    assert docs[0]["title"] == "Док 1"
    assert aggregate_documents([]) == []


@pytest.mark.parametrize(
    ("first", "second", "takes_second"),
    [
        (1.0, 0.6, True),  # ровно 0.6 первого и ≥ 0.35
        (1.0, 0.59, False),  # меньше 0.6 первого
        (0.5, 0.34, False),  # ≥ 0.6 первого, но < 0.35
        (0.5, 0.35, True),
    ],
)
def test_select_context_second_document_thresholds(first, second, takes_second):
    ranked = [_cand(1, DOC_A, 0, first), _cand(2, DOC_B, 0, second)]

    selected, docs = select_context(ranked, _no_neighbours)

    assert [d["document_id"] for d in docs] == [DOC_A, DOC_B]
    assert [c.idx for c in selected] == ([1, 2] if takes_second else [1])


def test_select_context_adds_neighbours_in_notebook_order():
    index = {
        (DOC_A, 4): _cand(40, DOC_A, 4),
        (DOC_A, 6): _cand(60, DOC_A, 6),
        (DOC_A, 7): _cand(70, DOC_A, 7),
    }
    ranked = [_cand(50, DOC_A, 5, 0.9), _cand(70, DOC_A, 7, 0.8)]

    selected, _ = select_context(ranked, lambda d, i: index.get((d, i)))

    # попадание, его соседи −1/+1, следующее попадание (уже добавлено как сосед — без дубля), его соседи
    assert [c.idx for c in selected] == [50, 40, 60, 70]
    assert selected[0].rerank_score == 0.9


def test_select_context_hit_limits_and_max_chunks():
    ranked = [_cand(i, DOC_A, i * 10, 1.0 - i * 0.01) for i in range(8)]
    ranked += [_cand(100 + i, DOC_B, i * 10, 0.99 - i * 0.01) for i in range(4)]

    def neighbours(doc, chunk_index):
        base = 1000 if doc == DOC_A else 2000
        return _cand(base + chunk_index, doc, chunk_index)

    selected, _ = select_context(ranked, neighbours)
    assert len(selected) == 9

    # без соседей: 5 попаданий первого документа + 2 второго
    selected, _ = select_context(ranked, _no_neighbours)
    assert [c.idx for c in selected] == [0, 1, 2, 3, 4, 100, 101]

    selected, _ = select_context(ranked, _no_neighbours, max_chunks=3)
    assert [c.idx for c in selected] == [0, 1, 2]


def test_select_context_empty():
    assert select_context([], _no_neighbours) == ([], [])


def test_build_context_format():
    item = _cand(1, DOC_A, 0, text="Пересдача разрешена дважды.", title="Положение")
    assert build_context([item]) == (
        "Источник: Положение\nКатегория: Общие документы\nРаздел: Раздел\n\nПересдача разрешена дважды."
    )
    assert build_context([]) == ""


def test_build_context_never_exceeds_max_chars():
    # Девять блоков ровно по 2000 символов: без учёта разделителей вышло бы 18000 + 8 разделителей.
    header_len = len(build_context([_cand(0, DOC_A, 0, text="")]))
    selected = [_cand(i, DOC_A, i, text="я" * (2000 - header_len)) for i in range(12)]

    context = build_context(selected, max_chars=18000)

    assert 0 < len(context) <= 18000
    assert len(build_context(selected)) <= 18000
    assert context.count("\n\n============================\n\n") == context.count("Источник:") - 1 == 7


def test_build_context_skips_too_long_block_but_keeps_later_ones():
    big = _cand(1, DOC_A, 0, text="я" * 500)
    small = _cand(2, DOC_A, 1, text="ok")

    context = build_context([big, small], max_chars=200)

    assert "я" * 500 not in context
    assert context.endswith("ok")
