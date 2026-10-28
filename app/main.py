import logging

from fastapi import FastAPI

from app.metrics import setup_metrics
from app.api.http import router as http_router
from app.core.config import settings
from app.core.internal_auth import validate_internal_auth_mode
from app.core.logging import configure_logging
from app.core.startup_checks import verify_embedding_setup
from app.db.session import SessionLocal
from app.generation.job_queue import bridge, job_queue, knowledge_index
from app.pipeline.runtime import start_index, stop_index_refresh

configure_logging()
logger = logging.getLogger("graph-rag-svc.main")

app = FastAPI(title=settings.app_name)
setup_metrics(app)
app.include_router(http_router)
app.state.index_refresh_task = None


@app.on_event("startup")
async def startup_event() -> None:
    validate_internal_auth_mode()
    # Сверка размерности эмбеддингов до запуска очереди: расхождение модели и схемы БД
    # должно останавливать сервис здесь, а не всплывать невнятной ошибкой pgvector на
    # первом же запросе пользователя. GraphRAG отключён, графовые колонки не проверяются.
    await verify_embedding_setup()

    # Индекс базы знаний (BM25 по чанкам) строится до приёма запросов и дальше обновляется
    # в фоне. Пустой индекс — только предупреждение: документы могут проиндексировать позже.
    app.state.index_refresh_task = await start_index(
        knowledge_index, SessionLocal, settings.index_refresh_interval_sec
    )

    await bridge.start()
    await job_queue.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await job_queue.stop()
    await bridge.stop()
    await stop_index_refresh(app.state.index_refresh_task)
    app.state.index_refresh_task = None
