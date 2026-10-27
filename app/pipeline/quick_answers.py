"""Быстрые ответы: гибридный поиск по вариантам вопросов (ячейки 32–34).

Найденный ответ отдаётся пользователю без вызова LLM. BM25 здесь свой, по вариантам быстрых
ответов, и токенизирует простым `.split()` нормализованного текста — не путать с BM25 чанков
(в ноутбуке второй перезаписывал первый, в сервисе они раздельны).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.db.models import LibraryQuickAnswer, LibraryQuickAnswerVariant
from app.pipeline.hybrid import QUICK_DENSE_WEIGHT, QUICK_LEXICAL_WEIGHT, RRF_K, rrf_merge
from app.pipeline.text import normalize_variant

QUICK_RRF_THRESHOLD = 0.04
QUICK_COSINE_THRESHOLD = 0.83


@dataclass(frozen=True)
class QuickAnswer:
    answer: str
    question: str
    rrf_score: float


async def _load_variants(session, query_vector: list[float], model: str) -> list[Any]:
    """Все варианты активных быстрых ответов с косинусной близостью к вопросу, по её убыванию.

    Вариантов немного (сотни), поэтому берутся все: плотный ранг в RRF считается по всему
    набору, как в ноутбуке, а тексты нужны для BM25 по тому же набору.
    """
    distance = LibraryQuickAnswerVariant.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            LibraryQuickAnswerVariant.text,
            (1 - distance).label("cosine"),
            LibraryQuickAnswer.answer,
            LibraryQuickAnswer.question,
        )
        .join(LibraryQuickAnswer, LibraryQuickAnswer.id == LibraryQuickAnswerVariant.quick_answer_id)
        .where(LibraryQuickAnswer.is_active.is_(True))
        .where(LibraryQuickAnswerVariant.embedding.is_not(None))
        # Векторы другой модели несравнимы с вектором вопроса (см. index.dense_search).
        .where(LibraryQuickAnswerVariant.embedding_model == model)
        .order_by(distance, LibraryQuickAnswerVariant.id)
    )
    return list((await session.execute(stmt)).all())


async def find_quick_answer(
    session,
    question: str,
    query_vector: list[float],
    model: str,
    rrf_threshold: float = QUICK_RRF_THRESHOLD,
    cosine_threshold: float = QUICK_COSINE_THRESHOLD,
) -> QuickAnswer | None:
    """Готовый ответ, если лучший по RRF вариант прошёл оба порога, иначе None.

    `query_vector` — эмбеддинг (kind=query) нормализованного вопроса `normalize_variant(question)`:
    в ноутбуке (ячейка 34) в `hybrid_search` попадает уже очищенный текст. Косинусный порог,
    как в ноутбуке, сравнивается с лучшей плотной близостью среди всех вариантов.
    """
    query = normalize_variant(question)
    if not query:
        return None

    variants = await _load_variants(session, query_vector, model)
    if not variants:
        return None

    dense = [(i, float(row.cosine)) for i, row in enumerate(variants)]
    best_dense_score = dense[0][1]

    bm25 = BM25Okapi([row.text.split() for row in variants])
    bm25_scores = bm25.get_scores(query.lower().split())
    lexical = [(int(i), float(bm25_scores[i])) for i in np.argsort(bm25_scores)[::-1] if bm25_scores[i] > 0]

    merged = rrf_merge(dense, lexical, QUICK_DENSE_WEIGHT, QUICK_LEXICAL_WEIGHT, k=RRF_K)
    best_idx, best_rrf_score = merged[0]

    if best_rrf_score >= rrf_threshold and best_dense_score >= cosine_threshold:
        best = variants[best_idx]
        return QuickAnswer(answer=best.answer, question=best.question, rrf_score=best_rrf_score)
    return None
