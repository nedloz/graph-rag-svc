"""Индекс базы знаний в работающем сервисе: загрузка на старте и фоновое обновление.

Индекс строится один раз при запуске (до приёма запросов), а затем раз в
`INDEX_REFRESH_INTERVAL_SEC` сверяет сигнатуру набора чанков с базой и перестраивается, если
документы переиндексировали, добавили или сняли. Пустой индекс не мешает старту: KnowledgeIndex
сам пишет предупреждение, а поиск просто ничего не находит, пока документы не проиндексируют.
"""

import asyncio
import logging
from contextlib import suppress

from app.pipeline.index import KnowledgeIndex, SessionFactory

logger = logging.getLogger("graph-rag-svc.pipeline.runtime")


async def refresh_periodically(index: KnowledgeIndex, session_factory: SessionFactory, interval_sec: float) -> None:
    """Бесконечный цикл обновления. Сбой одной проверки (БД недоступна) только логируется."""
    while True:
        await asyncio.sleep(interval_sec)
        try:
            await index.refresh_if_stale(session_factory)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Не удалось обновить индекс базы знаний; повтор через %s с", interval_sec)


async def start_index(index: KnowledgeIndex, session_factory: SessionFactory, interval_sec: float) -> asyncio.Task:
    """Загружает индекс (ошибка загрузки останавливает старт) и запускает фоновое обновление."""
    await index.load(session_factory)
    return asyncio.create_task(refresh_periodically(index, session_factory, interval_sec))


async def stop_index_refresh(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
