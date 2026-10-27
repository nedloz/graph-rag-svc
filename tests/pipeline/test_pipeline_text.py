"""Токенизация для BM25 (ячейка 13), нормализация вариантов быстрых ответов и RRF (ячейки 21, 33)."""

import re

import pytest

from app.pipeline.hybrid import (
    CHUNK_DENSE_WEIGHT,
    CHUNK_LEXICAL_WEIGHT,
    CHUNK_TOP_K,
    QUICK_DENSE_WEIGHT,
    QUICK_LEXICAL_WEIGHT,
    RRF_K,
    hybrid_candidates,
    rrf_merge,
)
from app.pipeline.text import chunk_retrieval_text, normalize_variant
from app.pipeline.tokenize import tokenize


def test_tokenize_stems_and_drops_short_words():
    assert tokenize("Пересдачи экзаменов") == ["пересдач", "экзамен"]
    # слова длиной ≤ 2 отбрасываются до стемминга, регистр не важен
    assert tokenize("Я и ОН: пересдачи в 2024") == ["пересдач", "2024"]
    assert tokenize("ИУП, retake!") == ["иуп", "retake"]
    assert tokenize("") == []


def test_tokenize_keeps_yo_as_word_character():
    # [A-Za-zА-Яа-яЁё0-9]: «ещё» — одно слово, а не «ещ» + «ё»
    assert len(tokenize("ещё сдача")) == 2


_NOTEBOOK_RE = r"[^a-zA-Zа-яА-Я0-9\s]"


@pytest.mark.parametrize(
    "raw",
    [
        "Как получить справку об обучении?",
        "  СОП — это что?!  ",
        "Где взять ещё одну Ёлку; 2024 г.",
        "How to get a certificate of enrollment?",
        "чтотакое соп",
        "",
    ],
)
def test_normalize_variant_is_exactly_the_notebook_rule(raw):
    assert normalize_variant(raw) == re.sub(_NOTEBOOK_RE, "", raw).lower().strip()


def test_normalize_variant_cuts_yo():
    # Диапазон [а-яА-Я] не включает Ё/ё: буква вырезается так же, как при импорте вариантов.
    assert normalize_variant("Ещё ЁЛКА") == "ещ лка"
    assert normalize_variant("Справка?") == "справка"


def test_chunk_retrieval_text_matches_ingest_contract():
    assert chunk_retrieval_text("ПУД", "Положения", "Раздел 1", "Текст чанка") == (
        "Название документа: ПУД\nКатегория: Положения\nРаздел: Раздел 1\nТекст: Текст чанка"
    )


def test_rrf_merge_sums_weighted_ranks_and_cuts_top_k():
    dense = [(1, 0.9), (2, 0.8)]
    lexical = [(2, 5.0), (3, 4.0)]

    merged = rrf_merge(dense, lexical, dense_weight=1.0, lexical_weight=0.9, k=60)

    assert [idx for idx, _ in merged] == [2, 1, 3]
    scores = dict(merged)
    assert scores[2] == pytest.approx(1.0 / 62 + 0.9 / 61)  # одинаковый индекс суммируется
    assert scores[1] == pytest.approx(1.0 / 61)
    assert scores[3] == pytest.approx(0.9 / 62)

    assert rrf_merge(dense, lexical, 1.0, 0.9, k=60, top_k=2) == merged[:2]


def test_rrf_merge_weights_and_k_change_the_scores():
    merged = rrf_merge([(7, 0.5)], [(7, 1.0)], dense_weight=1.5, lexical_weight=1.0, k=10)
    assert merged == [(7, pytest.approx(1.5 / 11 + 1.0 / 11))]
    assert rrf_merge([], [], 1.0, 1.0) == []


def test_notebook_constants():
    assert (CHUNK_DENSE_WEIGHT, CHUNK_LEXICAL_WEIGHT, CHUNK_TOP_K, RRF_K) == (1.0, 0.9, 40, 60)
    assert (QUICK_DENSE_WEIGHT, QUICK_LEXICAL_WEIGHT) == (1.5, 1.0)


def test_hybrid_candidates_uses_chunk_weights_and_top_40():
    dense = [(i, 1.0) for i in range(60)]
    lexical = [(i, 1.0) for i in range(100, 160)]
    merged = hybrid_candidates(dense, lexical)
    assert len(merged) == 40
    assert merged == rrf_merge(dense, lexical, 1.0, 0.9, k=60, top_k=40)
