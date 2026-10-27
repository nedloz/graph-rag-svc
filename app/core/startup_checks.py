"""Проверки согласованности конфигурации, выполняемые до начала обслуживания запросов.

Мотивация: размерность эмбеддингов раньше была объявлена в трёх местах независимо
(init.sql, ingest-worker, этот сервис), и расхождение обнаруживалось только в момент
вставки или поиска — то есть уже на реальных данных и в виде невнятной ошибки pgvector.
Теперь единственным источником является EMBEDDING_DIM, а сервис при запуске сверяет его
и с моделью в embedding-svc, и с фактическим типом колонок в БД, и отказывается работать
при расхождении. См. README, §13.1.
"""

import asyncio
import logging

import httpx
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from app.core.config import settings
from app.db.session import SessionLocal
from app.mtls import httpx_kwargs

logger = logging.getLogger("graph-rag-svc.core.startup_checks")

# Колонки, размерность которых обязана совпадать с моделью эмбеддингов.
_VECTOR_COLUMNS = (
    ("library.chunk_embeddings", "embedding"),
)
# graph.entities.name_embedding (VECTOR(1024)) намеренно не проверяется: GraphRAG отключён,
# а EMBEDDING_DIM для векторного поиска по чанкам больше не равен размерности графа.


def warn_if_development_only_provider() -> None:
    """`LLM_PROVIDER=local` — режим локальной разработки, не для прод-среды.

    Это единственный backend, который грузит модель в процесс самого сервиса, а не в
    отдельный контейнер: пересборка сервиса заново загружает модель, а память под неё
    конкурирует с памятью сервиса. Для локальной модели в проде есть `ollama`.
    """
    if settings.llm_provider == "local":
        logger.warning(
            "LLM_PROVIDER=local: модель грузится в процесс самого сервиса. Это режим локальной "
            "разработки — в прод-среде с остальными контейнерами используйте LLM_PROVIDER=ollama."
        )


async def _embedding_service_info() -> dict:
    """Опрашивает embedding-svc с повторами: модель грузится не мгновенно."""
    last_error: Exception | None = None
    for attempt in range(1, settings.embedding_startup_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0, **httpx_kwargs()) as client:
                response = await client.get(f"{settings.embedding_svc_url.rstrip('/')}/ping")
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            last_error = exc
            logger.info(
                "Ожидаем embedding-svc (%s/%s): %s",
                attempt,
                settings.embedding_startup_retries,
                exc,
            )
            await asyncio.sleep(settings.embedding_startup_retry_wait_sec)

    raise RuntimeError(
        f"embedding-svc недоступен по адресу {settings.embedding_svc_url}: {last_error}"
    )


async def _column_dimension(schema_table: str, column: str) -> int | None:
    """Фактическая размерность pgvector-колонки (None — если колонки нет либо тип без размера).

    Сам запрос живёт в БД (core.vector_column_dimension, migrations/007): ту же проверку делает
    ingest-worker, и две копии в разных репозиториях уже успели разойтись.
    """
    try:
        async with SessionLocal() as session:
            return (
                await session.execute(
                    text("SELECT core.vector_column_dimension(:rel, :col)"),
                    {"rel": schema_table, "col": column},
                )
            ).scalar()
    except ProgrammingError as exc:
        raise RuntimeError(
            "В БД нет функции core.vector_column_dimension(). Примените "
            "main/infra/db/migrations/007_vector_column_dimension.sql."
        ) from exc


async def verify_embedding_setup() -> None:
    """Падает с внятным сообщением, если модель и схема БД разошлись."""
    info = await _embedding_service_info()

    service_dim = info.get("dimension")
    if service_dim != settings.embedding_dim:
        raise RuntimeError(
            f"Модель '{info.get('model')}' в embedding-svc даёт {service_dim} измерений, "
            f"а EMBEDDING_DIM={settings.embedding_dim}. Смена модели эмбеддингов требует "
            f"пересчёта всех чанков — см. README, §13.1."
        )

    for schema_table, column in _VECTOR_COLUMNS:
        column_dim = await _column_dimension(schema_table, column)
        if column_dim is None:
            raise RuntimeError(
                f"Не удалось определить размерность {schema_table}.{column}: "
                f"колонка отсутствует или объявлена без размерности. "
                f"Примените main/infra/db/init/init.sql."
            )
        if column_dim != settings.embedding_dim:
            raise RuntimeError(
                f"Колонка {schema_table}.{column} объявлена как VECTOR({column_dim}), "
                f"а EMBEDDING_DIM={settings.embedding_dim}. Приведите схему БД и "
                f"конфигурацию к одному значению перед запуском."
            )

    logger.info(
        "Эмбеддинги: модель '%s', %s измерений — согласовано с %s",
        info.get("model"),
        settings.embedding_dim,
        " и ".join(f"{t}.{c}" for t, c in _VECTOR_COLUMNS),
    )
