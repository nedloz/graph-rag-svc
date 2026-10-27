"""DISABLED: GraphRAG отключён (спецификация 2026-10-27-notebook-pipeline-design). Код не поддерживается и не вызывается.

Entity resolution / aliasing (offline graph-builder + online query pipeline both
call this — same "resolve a name string to a graph.entities row" logic either way).

Resolution order, per the migration plan: exact/normalized alias match first (cheap,
precise), then pgvector similarity on Entity.name_embedding as a fallback for fuzzy
phrasing, else create a new entity. This is the Postgres-only stand-in for
SimpleGraphRAG's entity-FAISS index (research/SimpleRag.ipynb) — see graph-rag-svc's
README, decision 4.
"""

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import CurationBlocklist, Entity, EntityAlias
from app.graph.embeddings import embed_text, embed_text_with_model
from app.ontology.schema import BASE_ENTITY_TYPES

_BASE_TYPES_BY_NORMALIZED = {t.strip().casefold(): t for t in BASE_ENTITY_TYPES}
UNKNOWN_ENTITY_TYPE = "unknown"


class EntityBlocked(Exception):
    """Имя в стоп-листе курирования — сущность не должна создаваться заново."""


def normalize_alias(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip().casefold()


def classify_entity_type(name: str) -> str:
    """If `name` literally names one of the fixed ontology types (e.g. the LLM
    extracted "Студент" as an entity), use that as its type. Otherwise the entity is a
    context-specific concept the prompt asked for (e.g. "Библиотека", "Штраф") with no
    type of its own — bucket it as unknown for now; manual curation can refine later
    (see graph-rag-svc README, decision 3: iterative, not automated yet).
    """
    return _BASE_TYPES_BY_NORMALIZED.get(normalize_alias(name), UNKNOWN_ENTITY_TYPE)


async def is_blocked(session: AsyncSession, alias_normalized: str) -> bool:
    stmt = select(CurationBlocklist.id).where(CurationBlocklist.name_normalized == alias_normalized).limit(1)
    return (await session.execute(stmt)).scalars().first() is not None


async def _find_by_alias(session: AsyncSession, alias_normalized: str) -> Entity | None:
    stmt = (
        select(Entity)
        .join(EntityAlias, EntityAlias.entity_id == Entity.id)
        .where(EntityAlias.alias_normalized == alias_normalized)
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def _find_by_similarity(session: AsyncSession, entity_type: str, name: str) -> Entity | None:
    try:
        query_embedding = await embed_text(name)
    except Exception:
        return None

    max_distance = 1 - settings.entity_similarity_threshold
    distance_expr = Entity.name_embedding.cosine_distance(query_embedding)
    stmt = (
        select(Entity, distance_expr.label("distance"))
        .where(Entity.entity_type == entity_type, Entity.name_embedding.isnot(None))
        .order_by(distance_expr)
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None

    candidate, distance = row
    if distance is None or distance > max_distance:
        return None
    return candidate


async def find_entity(session: AsyncSession, name: str, *, entity_type: str | None = None) -> Entity | None:
    """Read-only lookup — same matching order as resolve_entity(), but never creates a
    new entity. Used by the online query pipeline (app/retrieval/graph_retriever.py):
    an unresolved question shouldn't pollute the graph with a new node, unlike the
    offline builder, which is meant to grow the graph.
    """
    resolved_type = entity_type or classify_entity_type(name)
    alias_normalized = normalize_alias(name)

    existing = await _find_by_alias(session, alias_normalized)
    if existing is not None:
        return existing

    return await _find_by_similarity(session, resolved_type, name)


async def resolve_entity(session: AsyncSession, name: str, *, entity_type: str | None = None) -> Entity:
    """Resolve `name` to a graph.entities row, creating one (plus its first alias) if
    nothing matches. Added to `session` but not committed — the caller controls the
    transaction boundary.
    """
    resolved_type = entity_type or classify_entity_type(name)
    alias_normalized = normalize_alias(name)

    existing = await _find_by_alias(session, alias_normalized)
    if existing is not None:
        return existing

    # Проверка стоп-листа делается здесь, а не в вызывающем коде: так ни один путь не может
    # создать сущность, которую куратор уже удалил как мусорную.
    if await is_blocked(session, alias_normalized):
        raise EntityBlocked(name)

    similar = await _find_by_similarity(session, resolved_type, name)
    if similar is not None:
        session.add(EntityAlias(entity_id=similar.id, alias=name.strip(), alias_normalized=alias_normalized, source="llm_merge"))
        return similar

    try:
        embedding, embedding_model = await embed_text_with_model(name)
    except Exception:
        embedding, embedding_model = None, None

    entity = Entity(
        entity_type=resolved_type,
        canonical_name=name.strip(),
        name_embedding=embedding,
        embedding_model=embedding_model,
    )
    session.add(entity)
    await session.flush()  # assign entity.id for the alias FK below

    session.add(EntityAlias(entity_id=entity.id, alias=name.strip(), alias_normalized=alias_normalized, source="extraction"))
    return entity
