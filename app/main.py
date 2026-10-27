import logging

from fastapi import FastAPI

from app.metrics import setup_metrics
from app.api.graph_admin import router as graph_admin_router
from app.api.http import router as http_router
from app.core.config import settings
from app.core.internal_auth import validate_internal_auth_mode
from app.core.logging import configure_logging
from app.core.startup_checks import verify_embedding_setup, warn_if_development_only_provider
from app.db.session import SessionLocal
from app.generation.job_queue import bridge, job_queue
from app.graph.store import release_orphaned_runs

configure_logging()
logger = logging.getLogger("graph-rag-svc.main")

app = FastAPI(title=settings.app_name)
setup_metrics(app)
app.include_router(http_router)
app.include_router(graph_admin_router)


@app.on_event("startup")
async def startup_event() -> None:
    validate_internal_auth_mode()
    # Сверка размерности эмбеддингов до запуска очереди: расхождение модели и схемы БД
    # должно останавливать сервис здесь, а не всплывать невнятной ошибкой pgvector на
    # первом же запросе пользователя.
    await verify_embedding_setup()
    warn_if_development_only_provider()

    # Сборка графа живёт как задача внутри процесса и перезапуск не переживает, поэтому
    # любой прогон, оставшийся в статусе running, на этом месте заведомо мёртв. Не закрыв
    # его, сервис навсегда отказывал бы в новом запуске: POST /internal/graph/build
    # отклоняет запрос при любом незавершённом прогоне.
    async with SessionLocal() as session:
        released = await release_orphaned_runs(session)
    if released:
        logger.warning(
            "Прогонов сборки графа помечено прерванными после перезапуска: %d", released
        )

    await bridge.start()
    await job_queue.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await job_queue.stop()
    await bridge.stop()
