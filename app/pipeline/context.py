"""Агрегация кандидатов по документам, выбор контекста с соседями и сборка текста (ячейка 25)."""

import uuid
from collections import defaultdict
from collections.abc import Callable

from app.pipeline.index import Candidate

DOCUMENT_HIT_WEIGHTS = (1.0, 0.55, 0.30, 0.15)
SECOND_DOC_RELATIVE = 0.60
SECOND_DOC_ABSOLUTE = 0.35
FIRST_DOC_HITS = 5
SECOND_DOC_HITS = 2
MAX_CONTEXT_CHUNKS = 9
MAX_CONTEXT_CHARS = 18000
CONTEXT_SEPARATOR = "\n\n============================\n\n"


def aggregate_documents(ranked: list[Candidate]) -> list[dict]:
    """Оценка документа — взвешенная сумма четырёх лучших rerank-оценок его фрагментов.

    Возвращает `[{"document_id", "title", "score", "hits"}]` по убыванию оценки; `hits` —
    фрагменты документа по убыванию rerank_score.
    """
    grouped: dict[uuid.UUID, list[Candidate]] = defaultdict(list)
    for item in ranked:
        grouped[item.document_id].append(item)

    docs = []
    for doc_id, items in grouped.items():
        items = sorted(items, key=lambda x: x.rerank_score or 0, reverse=True)
        scores = [item.rerank_score or 0 for item in items[:4]]
        doc_score = sum(score * weight for score, weight in zip(scores, DOCUMENT_HIT_WEIGHTS))
        docs.append({
            "document_id": doc_id,
            "title": items[0].title,
            "score": float(doc_score),
            "hits": items,
        })

    return sorted(docs, key=lambda x: x["score"], reverse=True)


def select_context(
    ranked: list[Candidate],
    neighbours: Callable[[uuid.UUID, int], Candidate | None],
    max_chunks: int = MAX_CONTEXT_CHUNKS,
) -> tuple[list[Candidate], list[dict]]:
    """Лучший документ (до 5 попаданий) и, если он близок, второй (до 2 попаданий).

    Второй документ берётся при оценке ≥ 0.6 первого и ≥ 0.35. К каждому попаданию
    добавляются соседние чанки `±1` того же документа (`neighbours(document_id, chunk_index)`),
    повторы отбрасываются, итог обрезается до `max_chunks`. Возвращает (выбранные чанки,
    документы из `aggregate_documents`).
    """
    docs = aggregate_documents(ranked)
    if not docs:
        return [], []

    chosen_docs = [docs[0]]
    if len(docs) > 1:
        first = docs[0]["score"]
        second = docs[1]["score"]
        if second >= SECOND_DOC_RELATIVE * first and second >= SECOND_DOC_ABSOLUTE:
            chosen_docs.append(docs[1])

    selected: list[Candidate] = []
    seen: set[int] = set()

    def add(item: Candidate) -> None:
        if item.idx in seen:
            return
        seen.add(item.idx)
        selected.append(item)

    for doc_pos, doc_info in enumerate(chosen_docs):
        hit_limit = FIRST_DOC_HITS if doc_pos == 0 else SECOND_DOC_HITS
        for hit in doc_info["hits"][:hit_limit]:
            add(hit)
            for chunk_index in (hit.chunk_index - 1, hit.chunk_index + 1):
                neighbour = neighbours(hit.document_id, chunk_index)
                if neighbour is not None:
                    add(neighbour)

    return selected[:max_chunks], docs


def build_context(selected: list[Candidate], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Блоки «Источник / Категория / Раздел / текст» через разделитель, не длиннее `max_chars`.

    Блок, который не помещается, пропускается, а следующие короткие ещё могут войти (как в
    ноутбуке). В отличие от ноутбука, в бюджет засчитываются и разделители: иначе девять
    блоков давали до 18000 + 8·32 символов.
    """
    blocks: list[str] = []
    used_chars = 0

    for item in selected:
        block = (
            f"Источник: {item.title}\n"
            f"Категория: {item.category}\n"
            f"Раздел: {item.section}\n\n"
            f"{item.text}"
        )
        cost = len(block) + (len(CONTEXT_SEPARATOR) if blocks else 0)
        if used_chars + cost > max_chars:
            continue
        blocks.append(block)
        used_chars += cost

    return CONTEXT_SEPARATOR.join(blocks)
