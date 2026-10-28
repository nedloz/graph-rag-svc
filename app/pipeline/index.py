"""Индекс базы знаний в памяти: чанки проиндексированных документов и BM25 по ним (ячейки 13, 21, 25).

Плотный поиск идёт в pgvector, но его результаты переводятся в позиции этого же снимка, чтобы
RRF, реранк и соседние чанки работали с одной нумерацией, как `idx` в ноутбуке.

Снимок (`IndexSnapshot`) неизменяем. `KnowledgeIndex` подменяет его одним присваиванием, когда
новый снимок полностью построен, поэтому читатели никогда не видят индекс наполовину. Запросу
стоит взять `index.snapshot()` один раз и работать с ним: тогда позиции из лексического и
плотного поиска гарантированно относятся к одному и тому же набору чанков.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select, text

from app.db.models import LibraryChunk, LibraryChunkEmbedding, LibraryDocument
from app.pipeline.hybrid import CHUNK_DENSE_EF_SEARCH
from app.pipeline.text import chunk_retrieval_text
from app.pipeline.tokenize import tokenize

logger = logging.getLogger(__name__)

Signature = tuple[int, str | None]


@dataclass(frozen=True)
class Candidate:
    """Чанк базы знаний; `idx` — позиция в снимке индекса. Оценки меняются через `dataclasses.replace`."""

    idx: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    title: str
    category: str
    section: str
    text: str
    rrf_score: float = 0.0
    rerank_score: float = 0.0


async def fetch_signature(session) -> Signature:
    """Число чанков проиндексированных документов и последний `indexed_at` среди них.

    Переиндексация документа сдвигает `indexed_at`, снятие с индекса или удаление меняет число
    чанков — любое изменение набора чанков меняет сигнатуру.
    """
    stmt = (
        select(
            func.count(LibraryChunk.id).label("chunks"),
            func.max(LibraryDocument.indexed_at).label("last_indexed_at"),
        )
        .select_from(LibraryChunk)
        .join(LibraryDocument, LibraryDocument.id == LibraryChunk.document_id)
        .where(LibraryDocument.ingest_status == "indexed")
    )
    row = (await session.execute(stmt)).one()
    last = row.last_indexed_at
    return int(row.chunks or 0), (last.isoformat() if hasattr(last, "isoformat") else last)


async def fetch_chunks(session) -> list[Any]:
    """Чанки проиндексированных документов вместе с названием, категорией и разделом."""
    stmt = (
        select(
            LibraryChunk.id,
            LibraryChunk.document_id,
            LibraryChunk.chunk_index,
            LibraryChunk.section_path,
            LibraryChunk.text,
            LibraryDocument.title,
            LibraryDocument.category,
        )
        .join(LibraryDocument, LibraryDocument.id == LibraryChunk.document_id)
        .where(LibraryDocument.ingest_status == "indexed")
        .order_by(LibraryChunk.document_id, LibraryChunk.chunk_index)
    )
    return list((await session.execute(stmt)).all())


# HNSW-индекс library.chunk_embeddings отдаёт не больше hnsw.ef_search строк (по умолчанию 40),
# а фильтры по модели и ingest_status применяются уже после обхода индекса. Без настройки
# dense-выдача короче k=60 и не совпадает с точным top-60 ноутбука (FAISS IndexFlatIP).
_SET_EF_SEARCH = text("SELECT set_config('hnsw.ef_search', :value, true)")
# Итеративный обход (pgvector >= 0.8) добирает строки, если фильтры отбросили слишком много;
# strict_order сохраняет точный порядок по расстоянию. На старых версиях параметра нет, а
# незнакомое имя с префиксом hnsw. после загрузки библиотеки — ошибка, поэтому set_config
# выполняется только при версии расширения >= 0.8.
_SET_ITERATIVE_SCAN = text(
    "SELECT set_config('hnsw.iterative_scan', 'strict_order', true) "
    "FROM pg_extension WHERE extname = 'vector' "
    "AND ARRAY[split_part(extversion, '.', 1)::int, split_part(extversion, '.', 2)::int] >= ARRAY[0, 8]"
)


async def tune_hnsw_scan(session, k: int) -> None:
    """Настройки обхода HNSW на текущую транзакцию (is_local): ef_search не меньше k."""
    ef_search = max(CHUNK_DENSE_EF_SEARCH, k)
    await session.execute(_SET_EF_SEARCH, {"value": str(ef_search)})
    await session.execute(_SET_ITERATIVE_SCAN)


class IndexSnapshot:
    """Неизменяемый набор чанков с BM25 и картой соседей."""

    def __init__(self, candidates: list[Candidate], signature: Signature | None) -> None:
        self.candidates = candidates
        self.signature = signature
        self._by_chunk_id = {c.chunk_id: c.idx for c in candidates}
        self._positions = {(c.document_id, c.chunk_index): c.idx for c in candidates}
        # BM25Okapi падает на пустом корпусе: пустой индекс просто ничего не находит.
        self._bm25 = (
            BM25Okapi([tokenize(chunk_retrieval_text(c.title, c.category, c.section, c.text)) for c in candidates])
            if candidates
            else None
        )

    @classmethod
    def build(cls, rows: list[Any], signature: Signature | None) -> "IndexSnapshot":
        candidates = [
            Candidate(
                idx=idx,
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_index=int(row.chunk_index),
                title=row.title or "",
                category=row.category or "",
                section=row.section_path or "",
                text=row.text or "",
            )
            for idx, row in enumerate(rows)
        ]
        return cls(candidates, signature)

    @property
    def size(self) -> int:
        return len(self.candidates)

    def candidate(self, idx: int) -> Candidate:
        return self.candidates[idx]

    def lexical_search(self, question: str, k: int) -> list[tuple[int, float]]:
        """BM25 top-k (ячейка 21, `bm25_search`): нулевые оценки тоже попадают в выдачу, как в ноутбуке."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(question))
        order = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in order]

    async def dense_search(self, session, query_vector: list[float], model: str, k: int) -> list[tuple[int, float]]:
        """pgvector top-k по косинусу → [(позиция в снимке, косинусная близость)].

        Чанки, которых нет в снимке (проиндексированы после его построения), пропускаются: их
        не знают ни BM25, ни карта соседей. Следующее обновление индекса их подхватит.
        """
        if not self.candidates:
            return []
        await tune_hnsw_scan(session, k)
        distance = LibraryChunkEmbedding.embedding.cosine_distance(query_vector)
        stmt = (
            select(LibraryChunkEmbedding.chunk_id, distance.label("distance"))
            .join(LibraryChunk, LibraryChunk.id == LibraryChunkEmbedding.chunk_id)
            .join(LibraryDocument, LibraryDocument.id == LibraryChunk.document_id)
            # Сравнивать можно только векторы одной модели: у разных моделей разные векторные
            # пространства, и косинусная близость между ними бессмысленна. Во время смены
            # модели это даёт меньше результатов, но никогда — неверные (README, §13.1).
            .where(LibraryChunkEmbedding.embedding_model == model)
            # Чанки документов, снятых с публикации или не доиндексированных, в выдачу
            # попадать не должны.
            .where(LibraryDocument.ingest_status == "indexed")
            .order_by(distance)
            .limit(k)
        )
        rows = (await session.execute(stmt)).all()
        return [
            (self._by_chunk_id[row.chunk_id], 1.0 - float(row.distance))
            for row in rows
            if row.chunk_id in self._by_chunk_id
        ]

    def neighbour(self, document_id: uuid.UUID, chunk_index: int) -> Candidate | None:
        idx = self._positions.get((document_id, chunk_index))
        return None if idx is None else self.candidates[idx]


SessionFactory = Callable[[], Any]


class KnowledgeIndex:
    """Держит текущий снимок и перестраивает его, когда меняется набор проиндексированных чанков."""

    def __init__(
        self,
        fetch_signature: Callable[[Any], Awaitable[Signature]] = fetch_signature,
        fetch_chunks: Callable[[Any], Awaitable[list[Any]]] = fetch_chunks,
    ) -> None:
        self._fetch_signature = fetch_signature
        self._fetch_chunks = fetch_chunks
        self._snapshot = IndexSnapshot([], None)
        self._lock = asyncio.Lock()

    def snapshot(self) -> IndexSnapshot:
        return self._snapshot

    @property
    def signature(self) -> Signature | None:
        return self._snapshot.signature

    @property
    def size(self) -> int:
        return self._snapshot.size

    def needs_refresh(self, signature: Signature) -> bool:
        current = self._snapshot.signature
        return current is None or tuple(current) != tuple(signature)

    async def load(self, session_factory: SessionFactory) -> None:
        async with self._lock:
            await self._load(session_factory)

    async def refresh_if_stale(self, session_factory: SessionFactory) -> bool:
        """Перестраивает индекс, если сигнатура в базе изменилась. True — индекс обновлён."""
        async with self._lock:
            async with session_factory() as session:
                signature = await self._fetch_signature(session)
            if not self.needs_refresh(signature):
                return False
            await self._load(session_factory)
            return True

    async def _load(self, session_factory: SessionFactory) -> None:
        # Сигнатура читается до чанков: если набор изменится между запросами, сохранённая
        # сигнатура окажется старше данных и следующая проверка просто перестроит индекс ещё раз.
        async with session_factory() as session:
            signature = await self._fetch_signature(session)
            rows = await self._fetch_chunks(session)
        # Токенизация и BM25 — CPU-работа на тысячи чанков: вне цикла событий.
        snapshot = await asyncio.to_thread(IndexSnapshot.build, rows, signature)
        self._snapshot = snapshot
        if snapshot.size == 0:
            logger.warning("Индекс базы знаний пуст: нет проиндексированных документов")
        else:
            logger.info("Индекс базы знаний: %d чанков, сигнатура %s", snapshot.size, signature)

    # Делегаты к текущему снимку — для простых вызовов; запросу лучше держать один snapshot().

    def candidate(self, idx: int) -> Candidate:
        return self._snapshot.candidate(idx)

    def lexical_search(self, question: str, k: int) -> list[tuple[int, float]]:
        return self._snapshot.lexical_search(question, k)

    async def dense_search(self, session, query_vector: list[float], model: str, k: int) -> list[tuple[int, float]]:
        return await self._snapshot.dense_search(session, query_vector, model, k)

    def neighbour(self, document_id: uuid.UUID, chunk_index: int) -> Candidate | None:
        return self._snapshot.neighbour(document_id, chunk_index)
