"""Загрузка индекса на старте и фоновое обновление (app/pipeline/runtime.py)."""

import asyncio
import logging

import pytest

from app.pipeline import runtime
from app.pipeline.index import KnowledgeIndex
from pipeline_fakes import FakeSession, chunk_row, session_factory


class FakeIndex:
    """Считает вызовы; refresh_if_stale по очереди отдаёт заготовленные исходы (исключение или bool)."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.loads = 0
        self.refreshes = 0
        self.refreshed = asyncio.Event()

    async def load(self, session_factory) -> None:
        self.loads += 1

    async def refresh_if_stale(self, session_factory) -> bool:
        self.refreshes += 1
        outcome = self.outcomes.pop(0) if self.outcomes else False
        if not self.outcomes:
            self.refreshed.set()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_start_loads_index_then_refreshes_in_background():
    async def scenario():
        index = FakeIndex(False, True)
        task = await runtime.start_index(index, session_factory(FakeSession()), interval_sec=0.001)
        assert index.loads == 1
        await asyncio.wait_for(index.refreshed.wait(), timeout=2)
        await runtime.stop_index_refresh(task)
        return index, task

    index, task = asyncio.run(scenario())
    assert index.refreshes >= 2
    assert task.done()


def test_refresh_errors_are_logged_and_loop_survives(caplog):
    async def scenario():
        index = FakeIndex(RuntimeError("db down"), ValueError("bad row"), True)
        task = await runtime.start_index(index, session_factory(FakeSession()), interval_sec=0.001)
        await asyncio.wait_for(index.refreshed.wait(), timeout=2)
        alive = not task.done()
        await runtime.stop_index_refresh(task)
        return index, alive

    with caplog.at_level(logging.ERROR):
        index, alive = asyncio.run(scenario())

    assert alive
    assert index.refreshes >= 3
    assert any("db down" in (r.exc_text or "") or "db down" in r.getMessage() for r in caplog.records)


def test_stop_cancels_refresh_task():
    async def scenario():
        index = FakeIndex()
        task = await runtime.start_index(index, session_factory(FakeSession()), interval_sec=3600)
        await runtime.stop_index_refresh(task)
        return task

    task = asyncio.run(scenario())
    assert task.cancelled()


def test_stop_without_task_is_noop():
    asyncio.run(runtime.stop_index_refresh(None))


def test_empty_index_only_warns(caplog):
    async def fetch_signature(session):
        return (0, None)

    async def fetch_chunks(session):
        return []

    async def scenario():
        index = KnowledgeIndex(fetch_signature=fetch_signature, fetch_chunks=fetch_chunks)
        task = await runtime.start_index(index, session_factory(FakeSession()), interval_sec=3600)
        await runtime.stop_index_refresh(task)
        return index

    with caplog.at_level(logging.WARNING):
        index = asyncio.run(scenario())

    assert index.size == 0
    assert any(r.levelno == logging.WARNING and "пуст" in r.getMessage() for r in caplog.records)


def test_load_error_fails_startup():
    class Broken(FakeIndex):
        async def load(self, session_factory) -> None:
            raise RuntimeError("no db")

    with pytest.raises(RuntimeError, match="no db"):
        asyncio.run(runtime.start_index(Broken(), session_factory(FakeSession()), interval_sec=1))


def test_real_index_is_loaded():
    import uuid

    rows = [chunk_row(uuid.UUID(int=1), 0, "Пересдача экзамена.")]

    async def fetch_signature(session):
        return (1, None)

    async def fetch_chunks(session):
        return rows

    async def scenario():
        index = KnowledgeIndex(fetch_signature=fetch_signature, fetch_chunks=fetch_chunks)
        task = await runtime.start_index(index, session_factory(FakeSession()), interval_sec=3600)
        await runtime.stop_index_refresh(task)
        return index

    assert asyncio.run(scenario()).size == 1
