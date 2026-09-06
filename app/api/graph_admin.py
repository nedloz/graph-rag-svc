"""Внутреннее API курирования графа (`/internal/graph/*`).

Живёт в этом сервисе, а не в db-svc, потому что все правила работы с графом уже реализованы
в `app/graph/` — дублировать их в админке на голом SQL значило бы разъехаться с билдером.
db-svc выступает клиентом этого API и проксирует запросы своему фронтенду. См. README, §13.3.
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.internal_auth import require_internal_service
from app.db.models import ExtractionRun
from app.db.session import SessionLocal, get_db_session
from app.graph import curation
from app.graph.builder import build_graph

logger = logging.getLogger("graph-rag-svc.api.graph_admin")

router = APIRouter(prefix="/internal/graph", dependencies=[Depends(require_internal_service)])


class EntityPatch(BaseModel):
    canonical_name: str | None = None
    entity_type: str | None = None
    attributes_json: dict | None = None


class MergeRequest(BaseModel):
    target_id: uuid.UUID


class AliasRequest(BaseModel):
    alias: str


class RelationPatch(BaseModel):
    relation: str | None = None
    confidence: float | None = None


class BuildRequest(BaseModel):
    document_ids: list[uuid.UUID] | None = None


def _handle(exc: curation.CurationError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get("/stats")
async def graph_stats(session: AsyncSession = Depends(get_db_session)):
    return await curation.stats(session)


@router.get("/entities")
async def list_entities(
    q: str | None = Query(default=None),
    entity_type: str | None = Query(default=None),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
):
    return await curation.search_entities(session, query=q, entity_type=entity_type, limit=limit, offset=offset)


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: uuid.UUID, session: AsyncSession = Depends(get_db_session)):
    try:
        return await curation.entity_card(session, entity_id)
    except curation.CurationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/entities/{entity_id}")
async def patch_entity(
    entity_id: uuid.UUID, payload: EntityPatch, session: AsyncSession = Depends(get_db_session)
):
    try:
        result = await curation.update_entity(
            session,
            entity_id,
            canonical_name=payload.canonical_name,
            entity_type=payload.entity_type,
            attributes_json=payload.attributes_json,
        )
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return result


@router.post("/entities/{entity_id}/merge")
async def merge_entity(
    entity_id: uuid.UUID, payload: MergeRequest, session: AsyncSession = Depends(get_db_session)
):
    try:
        result = await curation.merge_entities(session, entity_id, payload.target_id)
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return result


@router.post("/entities/{entity_id}/aliases")
async def create_alias(
    entity_id: uuid.UUID, payload: AliasRequest, session: AsyncSession = Depends(get_db_session)
):
    try:
        result = await curation.add_alias(session, entity_id, payload.alias)
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return result


@router.delete("/entities/{entity_id}/aliases/{alias_id}")
async def remove_alias(
    entity_id: uuid.UUID, alias_id: uuid.UUID, session: AsyncSession = Depends(get_db_session)
):
    try:
        await curation.delete_alias(session, entity_id, alias_id)
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return {"status": "deleted"}


@router.delete("/entities/{entity_id}")
async def remove_entity(
    entity_id: uuid.UUID,
    blocklist: bool = Query(default=False, description="Занести имена в стоп-лист, чтобы пересборка не создала заново"),
    reason: str | None = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    try:
        result = await curation.delete_entity(session, entity_id, blocklist=blocklist, reason=reason)
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return result


@router.patch("/relations/{relation_id}")
async def patch_relation(
    relation_id: uuid.UUID, payload: RelationPatch, session: AsyncSession = Depends(get_db_session)
):
    try:
        result = await curation.update_relation(
            session, relation_id, relation=payload.relation, confidence=payload.confidence
        )
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return result


@router.delete("/relations/{relation_id}")
async def remove_relation(relation_id: uuid.UUID, session: AsyncSession = Depends(get_db_session)):
    try:
        await curation.delete_relation(session, relation_id)
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return {"status": "deleted"}


@router.get("/orphans")
async def list_orphans(
    limit: int = Query(default=200, le=1000), session: AsyncSession = Depends(get_db_session)
):
    return await curation.orphans(session, limit=limit)


@router.get("/blocklist")
async def get_blocklist(
    limit: int = Query(default=200, le=1000), session: AsyncSession = Depends(get_db_session)
):
    return await curation.list_blocklist(session, limit=limit)


@router.delete("/blocklist/{entry_id}")
async def delete_blocklist_entry(entry_id: uuid.UUID, session: AsyncSession = Depends(get_db_session)):
    try:
        await curation.remove_from_blocklist(session, entry_id)
    except curation.CurationError as exc:
        raise _handle(exc) from exc
    await session.commit()
    return {"status": "deleted"}


@router.post("/build")
async def start_build(
    payload: BuildRequest = Body(default_factory=BuildRequest),
    session: AsyncSession = Depends(get_db_session),
):
    """Запуск пересборки графа. В проде отключается флагом GRAPH_BUILD_API_ENABLED=false.

    Сборка длинная, поэтому запускается фоновой задачей; за ходом следят через
    `GET /internal/graph/stats` → `recent_runs`, где виден статус и накопленная статистика.
    """
    if not settings.graph_build_api_enabled:
        raise HTTPException(status_code=403, detail="запуск пересборки через API отключён (GRAPH_BUILD_API_ENABLED)")

    running = (
        await session.execute(select(ExtractionRun.id).where(ExtractionRun.status == "running").limit(1))
    ).scalars().first()
    if running is not None:
        raise HTTPException(status_code=409, detail=f"пересборка уже идёт (run {running})")

    run_type = "incremental" if payload.document_ids else "full_rebuild"
    asyncio.create_task(_run_build(payload.document_ids, run_type))
    return {"status": "started", "run_type": run_type}


async def _run_build(document_ids: list[uuid.UUID] | None, run_type: str) -> None:
    try:
        stats_result = await build_graph(SessionLocal, document_ids=document_ids, run_type=run_type)
        logger.info("Пересборка графа завершена: %s", stats_result)
    except Exception:
        logger.exception("Пересборка графа упала")
