"""Client for embedding-svc — used to embed entity canonical names for the pgvector
similarity fallback in entity resolution (app/graph/resolution.py) and to embed the
question in app/retrieval/vector_retriever.py.

embedding-svc is the single source of embeddings for the whole system: ingest-worker
computes chunk vectors through it too. That was not always so — ingest-worker used to load
its own SentenceTransformer and pad/truncate vectors to a dimension the database column
never had, which silently failed every insert; see the README, §13.1. Going through one
HTTP service keeps every vector in the same space by construction, and the dimension is
verified against the DB columns at startup (app/core/startup_checks.py).
"""

import httpx

from app.core.config import settings
from app.core.internal_auth import get_internal_auth_headers


async def embed_texts_with_model(texts: list[str]) -> tuple[list[list[float]], str]:
    if not texts:
        return [], ""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.embedding_svc_url.rstrip('/')}/embed",
            json={"texts": texts, "normalize": True},
            headers=get_internal_auth_headers(),
        )
        response.raise_for_status()
        data = response.json()
    return data["embeddings"], data["model"]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings, _ = await embed_texts_with_model(texts)
    return embeddings


async def embed_text(text: str) -> list[float]:
    embeddings = await embed_texts([text])
    return embeddings[0]


async def embed_text_with_model(text: str) -> tuple[list[float], str]:
    embeddings, model = await embed_texts_with_model([text])
    return embeddings[0], model
