"""Ties judge -> graph/vector retriever -> context_assembler together into the single
`retrieve()` call app/generation/job_queue.py's process_job makes before generation.
"""

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.metrics import CONTEXT_CHUNKS, EMPTY_CONTEXT, JUDGE_DECISIONS, STAGE_DURATION
from app.api.schemas import HistoryMessage
from app.retrieval import context_assembler, graph_retriever, judge, vector_retriever

logger = logging.getLogger("graph-rag-svc.retrieval.orchestrator")


async def retrieve(
    session_factory: async_sessionmaker[AsyncSession],
    question: str,
    profile: dict,
    history: list[HistoryMessage] | None = None,
) -> dict:
    started = time.perf_counter()
    chosen_route = judge.route(question, profile)
    query_embedding_model: str | None = None
    route_used = chosen_route

    async with session_factory() as session:
        if chosen_route == "graph":
            chunks = await graph_retriever.retrieve(session, question, history or [])
            if not chunks:
                # Defensive fallback (not the judge's job, see judge.py): an empty
                # graph hit is the clearest signal that vector search is more likely
                # to be useful for this question.
                logger.info("orchestrator: graph retrieval empty for question, falling back to vector")
                chunks, query_embedding_model = await vector_retriever.retrieve(session, question)
                route_used = "graph->vector"
        else:
            chunks, query_embedding_model = await vector_retriever.retrieve(session, question)

    retrieval_seconds = time.perf_counter() - started
    retrieval_ms = retrieval_seconds * 1000
    JUDGE_DECISIONS.labels(route=route_used.replace("->", "_to_")).inc()
    STAGE_DURATION.labels(stage="retrieval").observe(retrieval_seconds)
    CONTEXT_CHUNKS.observe(len(chunks))
    if not chunks:
        EMPTY_CONTEXT.inc()
    return context_assembler.assemble(
        chunks,
        retrieval_ms=retrieval_ms,
        query_embedding_model=query_embedding_model,
        route_used=route_used,
    )
