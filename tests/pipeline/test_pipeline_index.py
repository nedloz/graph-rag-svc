"""KnowledgeIndex: загрузка чанков, BM25, сигнатура обновления, атомарная подмена, dense-поиск."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from app.pipeline.index import Candidate, KnowledgeIndex
from pipeline_fakes import FakeSession, bound_values, chunk_row, session_factory, sql_of

DOC_A = uuid.UUID(int=1)
DOC_B = uuid.UUID(int=2)

ROWS = [
    chunk_row(DOC_A, 0, "Общие положения о порядке обучения студентов.", title="Положение об обучении"),
    chunk_row(DOC_A, 1, "Пересдача экзамена допускается не более двух раз.", title="Положение об обучении",
              section=None),
    chunk_row(DOC_B, 0, "Стипендия выплачивается ежемесячно.", title="Стипендии", category="Учебный процесс"),
]


class FakeSource:
    """Подделка чтения из БД: сигнатура и строки чанков задаются тестом."""

    def __init__(self, rows, signature=(3, "2026-10-27T10:00:00+00:00")):
        self.rows = rows
        self.signature = signature
        self.chunk_loads = 0
        self.gate: asyncio.Event | None = None
        self.entered: asyncio.Event | None = None

    async def fetch_signature(self, session):
        return self.signature

    async def fetch_chunks(self, session):
        self.chunk_loads += 1
        if self.gate is not None:
            self.entered.set()
            await self.gate.wait()
        return list(self.rows)


def _index(source: FakeSource) -> KnowledgeIndex:
    return KnowledgeIndex(fetch_signature=source.fetch_signature, fetch_chunks=source.fetch_chunks)


FACTORY = session_factory(FakeSession())


def _loaded(rows=ROWS, signature=(3, "2026-10-27T10:00:00+00:00")) -> tuple[KnowledgeIndex, FakeSource]:
    source = FakeSource(rows, signature)
    index = _index(source)
    asyncio.run(index.load(FACTORY))
    return index, source


def test_load_builds_candidates_in_order():
    index, _ = _loaded()

    assert index.size == 3
    first = index.candidate(1)
    assert isinstance(first, Candidate)
    assert (first.idx, first.document_id, first.chunk_index, first.title) == (1, DOC_A, 1, "Положение об обучении")
    assert first.section == ""  # NULL section_path → пустая строка, как у ingest-worker
    assert first.chunk_id == ROWS[1].id
    assert index.candidate(2).category == "Учебный процесс"


def test_lexical_search_ranks_matching_chunk_first():
    index, _ = _loaded()

    results = index.lexical_search("Сколько раз можно пересдать экзамен? Пересдача", k=2)

    assert results[0][0] == 1
    assert results[0][1] > 0
    assert len(results) == 2


def test_lexical_search_uses_title_and_category_too():
    # BM25 строится по тексту «Название документа / Категория / Раздел / Текст»
    index, _ = _loaded()
    assert index.lexical_search("стипендии учебный процесс", k=1)[0][0] == 2


def test_neighbour_lookup():
    index, _ = _loaded()
    assert index.neighbour(DOC_A, 0).idx == 0
    assert index.neighbour(DOC_A, 2) is None
    assert index.neighbour(DOC_B, -1) is None


def test_needs_refresh_compares_signature():
    index = _index(FakeSource(ROWS))
    assert index.needs_refresh((0, None)) is True  # ещё не загружен

    asyncio.run(index.load(FACTORY))

    assert index.signature == (3, "2026-10-27T10:00:00+00:00")
    assert index.needs_refresh((3, "2026-10-27T10:00:00+00:00")) is False
    assert index.needs_refresh((4, "2026-10-27T10:00:00+00:00")) is True
    assert index.needs_refresh((3, "2026-10-27T11:00:00+00:00")) is True


def test_refresh_if_stale_reloads_only_on_change():
    index, source = _loaded()

    assert asyncio.run(index.refresh_if_stale(FACTORY)) is False
    assert source.chunk_loads == 1

    source.rows = ROWS[:2]
    source.signature = (2, "2026-10-27T12:00:00+00:00")
    assert asyncio.run(index.refresh_if_stale(FACTORY)) is True
    assert source.chunk_loads == 2
    assert index.size == 2
    assert index.signature == (2, "2026-10-27T12:00:00+00:00")


def test_refresh_swaps_index_atomically():
    async def scenario():
        source = FakeSource(ROWS)
        index = _index(source)
        await index.load(FACTORY)
        old = index.snapshot()

        source.rows = [chunk_row(DOC_B, 0, "Общежитие: правила заселения.", title="Общежития")]
        source.signature = (1, "2026-10-28T00:00:00+00:00")
        source.gate = asyncio.Event()
        source.entered = asyncio.Event()

        refresh = asyncio.create_task(index.refresh_if_stale(FACTORY))
        await source.entered.wait()
        # Пока новый индекс строится, читатели видят старый целиком
        assert index.size == 3
        assert index.lexical_search("пересдача экзамена", k=1)[0][0] == 1
        assert index.signature == (3, "2026-10-27T10:00:00+00:00")

        source.gate.set()
        assert await refresh is True

        assert index.size == 1
        assert index.lexical_search("общежитие заселение", k=1)[0][0] == 0
        # Снимок, взятый до подмены, остаётся согласованным
        assert old.size == 3 and old.candidate(1).text == ROWS[1].text

    asyncio.run(scenario())


def test_empty_index_is_safe():
    index, _ = _loaded(rows=[], signature=(0, None))

    assert index.size == 0
    assert index.lexical_search("пересдача", k=60) == []
    assert index.neighbour(DOC_A, 0) is None
    assert index.needs_refresh((0, None)) is False

    unknown = SimpleNamespace(chunk_id=uuid.uuid4(), distance=0.1)
    assert asyncio.run(index.dense_search(FakeSession([unknown]), [0.1] * 4, "e5-base", k=60)) == []


def test_never_loaded_index_is_empty():
    index = _index(FakeSource(ROWS))
    assert index.size == 0
    assert index.lexical_search("пересдача", k=5) == []


def test_dense_search_maps_chunk_ids_and_filters_model():
    index, _ = _loaded()
    session = FakeSession([
        SimpleNamespace(chunk_id=ROWS[2].id, distance=0.1),
        SimpleNamespace(chunk_id=uuid.uuid4(), distance=0.2),  # чанк не из индекса — пропускается
        SimpleNamespace(chunk_id=ROWS[0].id, distance=0.3),
    ])

    results = asyncio.run(index.dense_search(session, [0.1] * 4, "e5-base", k=60))

    assert results == [(2, pytest.approx(0.9)), (0, pytest.approx(0.7))]
    values = bound_values(session.statements[0])
    assert "e5-base" in values and "indexed" in values and 60 in values
    assert "embedding_model" in sql_of(session.statements[0])


def test_load_sql_filters_indexed_documents():
    from app.pipeline.index import fetch_chunks, fetch_signature

    session = FakeSession([SimpleNamespace(chunks=0, last_indexed_at=None)], [])
    signature = asyncio.run(fetch_signature(session))
    asyncio.run(fetch_chunks(session))

    assert signature == (0, None)
    for stmt in session.statements:
        assert "indexed" in bound_values(stmt)


@pytest.mark.parametrize("k", [60, 500])
def test_dense_search_tunes_hnsw_before_select_in_same_session(k):
    # HNSW отдаёт не больше hnsw.ef_search строк, а фильтры по модели/статусу режут уже их:
    # настройка обязана уйти до SELECT в той же сессии (транзакции) и быть не меньше k.
    index, _ = _loaded()
    session = FakeSession([SimpleNamespace(chunk_id=ROWS[0].id, distance=0.1)])

    asyncio.run(index.dense_search(session, [0.1] * 4, "e5-base", k=k))

    (ef_stmt, ef_params), (iter_stmt, iter_params), (select_stmt, _) = session.executed
    assert "set_config('hnsw.ef_search', :value, true)" in str(ef_stmt)
    assert int(ef_params["value"]) >= k
    assert int(ef_params["value"]) >= 200
    # итеративный обход — только на pgvector >= 0.8, где такой параметр существует
    assert "hnsw.iterative_scan" in str(iter_stmt) and "extversion" in str(iter_stmt)
    assert iter_params is None
    assert select_stmt is session.statements[0]
    assert k in bound_values(select_stmt)


def test_dense_ef_search_constant_covers_dense_k():
    from app.pipeline.hybrid import CHUNK_DENSE_EF_SEARCH, CHUNK_DENSE_K

    assert CHUNK_DENSE_EF_SEARCH >= CHUNK_DENSE_K
