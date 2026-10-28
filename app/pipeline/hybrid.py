"""Слияние плотного и лексического поиска по RRF (ячейки 21 и 33)."""

from collections import defaultdict

RRF_K = 60

# Поиск чанков (ячейка 21): dense top-60 + BM25 top-60 → top-40.
CHUNK_DENSE_WEIGHT = 1.0
CHUNK_LEXICAL_WEIGHT = 0.9
CHUNK_DENSE_K = 60
# hnsw.ef_search для dense-запроса: не меньше CHUNK_DENSE_K (см. index.tune_hnsw_scan), с
# запасом на строки, которые отбросят фильтры по модели и статусу документа.
CHUNK_DENSE_EF_SEARCH = 200
CHUNK_LEXICAL_K = 60
CHUNK_TOP_K = 40

# Быстрые ответы (ячейка 33).
QUICK_DENSE_WEIGHT = 1.5
QUICK_LEXICAL_WEIGHT = 1.0


def rrf_merge(
    dense: list[tuple[int, float]],
    lexical: list[tuple[int, float]],
    dense_weight: float,
    lexical_weight: float,
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Взвешенный Reciprocal Rank Fusion: `вес / (k + ранг)`, ранги с единицы.

    Оценки исходных списков не используются — важен только порядок. Одинаковый индекс в обоих
    списках получает сумму вкладов. При равенстве впереди тот, кто раньше встретился (dense
    раньше lexical), как в ноутбуке.
    """
    scores: dict[int, float] = defaultdict(float)
    for rank, (idx, _) in enumerate(dense, start=1):
        scores[idx] += dense_weight / (k + rank)
    for rank, (idx, _) in enumerate(lexical, start=1):
        scores[idx] += lexical_weight / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [(idx, float(score)) for idx, score in ranked]


def hybrid_candidates(dense: list[tuple[int, float]], lexical: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Кандидаты для реранка (ячейка 21, `hybrid_candidates`)."""
    return rrf_merge(dense, lexical, CHUNK_DENSE_WEIGHT, CHUNK_LEXICAL_WEIGHT, k=RRF_K, top_k=CHUNK_TOP_K)
