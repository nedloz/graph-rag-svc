"""Offline graph-building pipeline: library.chunks -> shared ontology extractor ->
entity resolution -> graph storage. Invoked via cli/build_graph.py; also the module the
future automated-rebuild-on-document-update work (deferred, see README) would call into.
"""

import logging
import time
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.db.models import ExtractionRun, LibraryChunk, LibraryDocument
from app.graph.resolution import EntityBlocked, resolve_entity
from app.graph.store import add_source_chunks, create_extraction_run, finish_extraction_run, upsert_relation
from app.ontology.extractor import extract_triplets
from app.ontology.schema import ONTOLOGY_VERSION

logger = logging.getLogger("graph-rag-svc.graph.builder")


async def _indexed_document_ids(session: AsyncSession, document_ids: list[uuid.UUID] | None) -> list[uuid.UUID]:
    stmt = select(LibraryDocument.id).where(LibraryDocument.ingest_status == "indexed")
    if document_ids:
        stmt = stmt.where(LibraryDocument.id.in_(document_ids))
    return list((await session.execute(stmt)).scalars().all())


async def _document_chunks(session: AsyncSession, document_id: uuid.UUID) -> list[LibraryChunk]:
    stmt = (
        select(LibraryChunk)
        .where(LibraryChunk.document_id == document_id)
        .order_by(LibraryChunk.chunk_index)
    )
    return list((await session.execute(stmt)).scalars().all())


def _windows(chunks: list[LibraryChunk], window_size: int, step: int):
    if len(chunks) < window_size:
        if chunks:
            yield chunks
        return
    for i in range(0, len(chunks) - window_size + 1, step):
        yield chunks[i : i + window_size]


async def _process_window(session: AsyncSession, window: list[LibraryChunk], extraction_run_id: uuid.UUID, stats: dict) -> None:
    combined_text = "\n---\n".join(c.text for c in window)
    chunk_ids = [c.id for c in window]

    triplets = await extract_triplets(combined_text, mode="build", source_chunk_ids=[str(c) for c in chunk_ids])
    stats["windows_processed"] += 1
    stats["triplets_extracted"] += len(triplets)

    for triplet in triplets:
        try:
            subject = await resolve_entity(session, triplet.subject)
            obj = await resolve_entity(session, triplet.object)
        except EntityBlocked as blocked:
            # Сущность в стоп-листе курирования: триплет пропускаем целиком, иначе
            # пересборка вернула бы то, что куратор удалил вручную.
            logger.info("build_graph: пропущен триплет с заблокированной сущностью '%s'", blocked)
            stats["triplets_skipped_blocked"] += 1
            continue
        await session.flush()

        relation_id = await upsert_relation(
            session,
            subject_entity_id=subject.id,
            object_entity_id=obj.id,
            relation=triplet.relation,
            confidence=None,
            extraction_run_id=extraction_run_id,
        )
        await add_source_chunks(session, chunk_ids, relation_id=relation_id)
        await add_source_chunks(session, chunk_ids, entity_id=subject.id)
        await add_source_chunks(session, chunk_ids, entity_id=obj.id)
        stats["relations_upserted"] += 1


async def build_graph(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_ids: list[uuid.UUID] | None = None,
    run_type: str = "full_rebuild",
) -> dict:
    """Build/rebuild the graph from library.chunks. `document_ids=None` processes every
    `ingest_status='indexed'` document — a full rebuild is naturally idempotent (see
    app/graph/store.py), so this is safe to re-run as documents change.
    """
    stats = {
        "documents_processed": 0,
        "windows_processed": 0,
        "triplets_extracted": 0,
        "triplets_skipped_blocked": 0,
        "relations_upserted": 0,
        "errors": 0,
    }
    extraction_model = settings.graph_extraction_model or None

    async with session_factory() as session:
        run = await create_extraction_run(
            session,
            run_type=run_type,
            extraction_model=extraction_model,
            ontology_version=ONTOLOGY_VERSION,
            document_ids=document_ids,
        )
        await session.commit()
        run_id = run.id

    try:
        async with session_factory() as session:
            target_document_ids = await _indexed_document_ids(session, document_ids)

        logger.info("build_graph: %d document(s) in scope", len(target_document_ids))
        # Прогресс пишется по документам, а не только в начале и в конце. Сборка идёт
        # порядка секунд на окно, окон сотни, и без этих строк лог молчит десятками минут —
        # отличить работу от зависшего прогона по нему невозможно.
        total_documents = len(target_document_ids)
        started_at = time.monotonic()

        for document_id in target_document_ids:
            async with session_factory() as session:
                try:
                    chunks = await _document_chunks(session, document_id)
                    for window in _windows(chunks, settings.graph_window_size, settings.graph_window_step):
                        await _process_window(session, window, run_id, stats)

                    logger.info(
                        "build_graph: документ %d/%d — окон %d, триплетов %d, связей %d, %.0f с",
                        stats["documents_processed"] + 1,
                        total_documents,
                        stats["windows_processed"],
                        stats["triplets_extracted"],
                        stats["relations_upserted"],
                        time.monotonic() - started_at,
                    )
                    await session.commit()
                    stats["documents_processed"] += 1
                except Exception:
                    await session.rollback()
                    stats["errors"] += 1
                    logger.exception("build_graph: failed on document %s", document_id)

        async with session_factory() as session:
            run = await session.get(ExtractionRun, run_id)
            await finish_extraction_run(session, run, status="completed", stats=stats)
            await session.commit()

    except Exception:
        async with session_factory() as session:
            run = await session.get(ExtractionRun, run_id)
            if run is not None:
                await finish_extraction_run(session, run, status="failed", stats=stats)
                await session.commit()
        raise

    logger.info("build_graph: done — %s", stats)
    return stats
