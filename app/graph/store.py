"""DISABLED: GraphRAG отключён (спецификация 2026-10-27-notebook-pipeline-design). Код не поддерживается и не вызывается.

Postgres-backed graph read/write API. Both the offline graph-builder and the online
retrieval pipeline go through this module rather than issuing raw queries of their own.

Writes are upserts keyed on the tables' natural-key UNIQUE constraints
(entity_type+canonical_name for entities, subject+object+relation for relations) —
re-running the builder over the same chunks is additive/idempotent rather than
destructive, so a full rebuild does not need to truncate-and-swap the graph (see
graph-rag-svc README, "Оффлайн-пайплайн построения графа").
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EntitySourceChunk, ExtractionRun, Relation


async def create_extraction_run(
    session: AsyncSession,
    *,
    run_type: str,
    extraction_model: str | None,
    ontology_version: str,
    document_ids: list[uuid.UUID] | None = None,
) -> ExtractionRun:
    run = ExtractionRun(
        run_type=run_type,
        extraction_model=extraction_model,
        ontology_version=ontology_version,
        document_ids=document_ids,
        status="running",
    )
    session.add(run)
    await session.flush()
    return run


async def finish_extraction_run(session: AsyncSession, run: ExtractionRun, *, status: str, stats: dict) -> None:
    run.status = status
    run.stats_json = stats
    run.finished_at = datetime.now(timezone.utc)
    await session.flush()


async def release_orphaned_runs(session: AsyncSession) -> int:
    """Помечает прогоны, оставшиеся в статусе `running`, как прерванные. Возвращает их число.

    Вызывается при старте сервиса. Сборка графа выполняется как задача внутри процесса
    (`asyncio.create_task`), поэтому пережить перезапуск контейнера она не может: любая
    строка со статусом `running` на момент старта — заведомо осиротевшая.

    Без этого один перезапуск во время сборки блокировал запуск навсегда: `POST /build`
    отказывает при любом незавершённом прогоне, и снять блокировку из интерфейса было
    нельзя — только правкой в базе.
    """
    runs = (
        await session.execute(select(ExtractionRun).where(ExtractionRun.status == "running"))
    ).scalars().all()

    for run in runs:
        stats = dict(run.stats_json or {})
        stats["interrupted_reason"] = "сервис перезапущен во время сборки"
        await finish_extraction_run(session, run, status="failed", stats=stats)

    if runs:
        await session.commit()
    return len(runs)


async def upsert_relation(
    session: AsyncSession,
    *,
    subject_entity_id: uuid.UUID,
    object_entity_id: uuid.UUID,
    relation: str,
    confidence: float | None,
    extraction_run_id: uuid.UUID | None,
) -> uuid.UUID:
    stmt = (
        pg_insert(Relation)
        .values(
            subject_entity_id=subject_entity_id,
            object_entity_id=object_entity_id,
            relation=relation,
            confidence=confidence,
            extraction_run_id=extraction_run_id,
        )
        .on_conflict_do_update(
            index_elements=[Relation.subject_entity_id, Relation.object_entity_id, Relation.relation],
            set_={"confidence": confidence, "extraction_run_id": extraction_run_id},
        )
        .returning(Relation.id)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def add_source_chunks(
    session: AsyncSession,
    chunk_ids: list[uuid.UUID],
    *,
    entity_id: uuid.UUID | None = None,
    relation_id: uuid.UUID | None = None,
) -> None:
    if not chunk_ids:
        return

    existing_stmt = select(EntitySourceChunk.chunk_id).where(EntitySourceChunk.chunk_id.in_(chunk_ids))
    if entity_id is not None:
        existing_stmt = existing_stmt.where(EntitySourceChunk.entity_id == entity_id)
    else:
        existing_stmt = existing_stmt.where(EntitySourceChunk.relation_id == relation_id)
    existing = set((await session.execute(existing_stmt)).scalars().all())

    for chunk_id in chunk_ids:
        if chunk_id in existing:
            continue
        session.add(EntitySourceChunk(entity_id=entity_id, relation_id=relation_id, chunk_id=chunk_id))
