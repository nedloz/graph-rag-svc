"""Client for embedding-svc — used to embed entity canonical names for the pgvector
similarity fallback in entity resolution (app/graph/resolution.py) and to embed the
question and quick-answer lookup text in the notebook pipeline (app/pipeline/service.py).

embedding-svc is the single source of embeddings for the whole system: ingest-worker
computes chunk vectors through it too. That was not always so — ingest-worker used to load
its own SentenceTransformer and pad/truncate vectors to a dimension the database column
never had, which silently failed every insert; see the README, §13.1. Going through one
HTTP service keeps every vector in the same space by construction, and the dimension is
verified against the DB columns at startup (app/core/startup_checks.py).
"""

import httpx

from app.core.config import settings
from app.mtls import httpx_kwargs


async def embed_texts_with_model(texts: list[str], kind: str = "passage") -> tuple[list[list[float]], str]:
    """Векторы и имя модели. `kind` — `passage` (документ) или `query` (поисковый запрос):
    префикс модели (`passage: ` / `query: ` у E5) ставит embedding-svc, здесь его добавлять нельзя.
    """
    if not texts:
        return [], ""

    async with httpx.AsyncClient(timeout=60.0, **httpx_kwargs()) as client:
        response = await client.post(
            f"{settings.embedding_svc_url.rstrip('/')}/embed",
            json={"texts": texts, "normalize": True, "kind": kind},
        )
        response.raise_for_status()
        data = response.json()
    return data["embeddings"], data["model"]


async def embed_texts(texts: list[str], kind: str = "passage") -> list[list[float]]:
    embeddings, _ = await embed_texts_with_model(texts, kind=kind)
    return embeddings


async def embed_text(text: str, kind: str = "passage") -> list[float]:
    embeddings = await embed_texts([text], kind=kind)
    return embeddings[0]


async def embed_text_with_model(text: str, kind: str = "passage") -> tuple[list[float], str]:
    embeddings, model = await embed_texts_with_model([text], kind=kind)
    return embeddings[0], model
