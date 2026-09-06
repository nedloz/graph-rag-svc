"""Plain vector-RAG fallback: embed the question via embedding-svc, then a pgvector
top-k search over library.chunk_embeddings — this is the "готовая предобработанная
база чанков" entry point finally wired up the way rag-svc was always meant to
(see graph-rag-svc README).
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import LibraryChunk, LibraryChunkEmbedding, LibraryDocument
from app.graph.embeddings import embed_text_with_model
from app.retrieval.types import RetrievedChunk


async def retrieve(session: AsyncSession, question: str) -> tuple[list[RetrievedChunk], str]:
    query_embedding, query_model = await embed_text_with_model(question)

    distance_expr = LibraryChunkEmbedding.embedding.cosine_distance(query_embedding)
    stmt = (
        select(
            LibraryChunk.id,
            LibraryChunk.document_id,
            LibraryChunk.text,
            LibraryDocument.title.label("doc_title"),
            distance_expr.label("distance"),
        )
        .join(LibraryChunkEmbedding, LibraryChunkEmbedding.chunk_id == LibraryChunk.id)
        .join(LibraryDocument, LibraryDocument.id == LibraryChunk.document_id)
        # Сравнивать можно только векторы одной модели: у разных моделей разные векторные
        # пространства, и косинусная близость между ними бессмысленна. Во время смены
        # модели это даёт меньше результатов, но никогда — неверные (README, §13.1).
        .where(LibraryChunkEmbedding.embedding_model == query_model)
        # Чанки документов, снятых с публикации или не доиндексированных, в выдачу
        # попадать не должны.
        .where(LibraryDocument.ingest_status == "indexed")
        .order_by(distance_expr)
        .limit(settings.max_chunks_to_use)
    )
    rows = (await session.execute(stmt)).all()

    chunks = [
        RetrievedChunk(
            chunk_id=row.id,
            doc_id=row.document_id,
            text=row.text,
            score=1 - row.distance,  # cosine similarity
            source="vector",
            doc_title=row.doc_title,
        )
        for row in rows
    ]
    return chunks, query_model
