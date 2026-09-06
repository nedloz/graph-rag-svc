"""Online graph-RAG retrieval: extract entities from the question (via the shared
ontology extractor, same module the offline builder uses), resolve them against
graph.entities, do a bounded breadth-first traversal of graph.relations, and map the
reached entities back to their source chunks. The Postgres-only stand-in for
SimpleGraphRAG's networkx traversal (research/SimpleRag.ipynb) — see graph-rag-svc
README, decision 4.
"""

import logging
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.graph.resolution import find_entity
from app.ontology.extractor import extract_triplets
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger("graph-rag-svc.retrieval.graph_retriever")

_TRAVERSE_AND_COLLECT_SQL = text(
    """
    WITH RECURSIVE reachable(entity_id, hop) AS (
        SELECT id AS entity_id, 0 AS hop
        FROM graph.entities
        WHERE id = ANY(:seed_ids)

        UNION

        SELECT
            CASE WHEN rel.subject_entity_id = reachable.entity_id
                 THEN rel.object_entity_id
                 ELSE rel.subject_entity_id
            END AS entity_id,
            reachable.hop + 1 AS hop
        FROM graph.relations rel
        JOIN reachable
          ON rel.subject_entity_id = reachable.entity_id
          OR rel.object_entity_id = reachable.entity_id
        WHERE reachable.hop < :max_hops
    ),
    best_hop AS (
        SELECT entity_id, MIN(hop) AS hop FROM reachable GROUP BY entity_id
    )
    SELECT c.id AS chunk_id, c.document_id AS document_id, c.text AS text,
           d.title AS doc_title, MIN(bh.hop) AS hop
    FROM graph.entity_source_chunks esc
    JOIN best_hop bh ON bh.entity_id = esc.entity_id
    JOIN library.chunks c ON c.id = esc.chunk_id
    JOIN library.documents d ON d.id = c.document_id
    WHERE d.ingest_status = 'indexed'
    GROUP BY c.id, c.document_id, c.text, d.title
    ORDER BY hop ASC
    LIMIT :limit
    """
)


def _candidate_names(triplets) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for t in triplets:
        for name in (t.subject, t.object):
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                names.append(name)
    return names


def _extraction_text(question: str, history: list) -> str:
    """Текст, из которого извлекаются сущности вопроса.

    К самому вопросу подмешиваются последние реплики диалога: без них уточняющие и
    местоименные вопросы («а что насчёт пересдачи по ней же?») не дают ни одной сущности,
    и graph-ретривал впустую откатывается на векторный поиск.
    """
    if not history:
        return question
    recent = history[-settings.history_messages_for_extraction :]
    lines = [f"{m.role}: {m.content}" for m in recent]
    lines.append(f"user: {question}")
    return "\n".join(lines)


async def retrieve(session: AsyncSession, question: str, history: list | None = None) -> list[RetrievedChunk]:
    triplets = await extract_triplets(_extraction_text(question, history or []), mode="query")
    names = _candidate_names(triplets)
    if not names:
        return []

    seed_ids: list[uuid.UUID] = []
    for name in names:
        entity = await find_entity(session, name)
        if entity is not None:
            seed_ids.append(entity.id)

    if not seed_ids:
        logger.info("graph_retriever: no entities resolved for question (candidates=%s)", names)
        return []

    result = await session.execute(
        _TRAVERSE_AND_COLLECT_SQL,
        {
            "seed_ids": seed_ids,
            "max_hops": settings.graph_traversal_max_hops,
            "limit": settings.max_chunks_to_use,
        },
    )
    rows = result.mappings().all()

    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["document_id"],
            text=row["text"],
            score=1.0 / (1 + row["hop"]),
            source="graph",
            doc_title=row["doc_title"],
        )
        for row in rows
    ]
