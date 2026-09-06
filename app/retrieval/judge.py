"""Routes a question between graph-RAG and plain vector-RAG. Deliberately isolated in
its own tiny function so a real complexity-classifier model can later replace the body
of route() without touching callers (app/retrieval/orchestrator.py) — see graph-rag-svc
README, "Пайплайн".

The defensive fallback for when graph retrieval finds nothing lives in the orchestrator,
not here: that's a runtime safety net independent of which strategy the judge picked.
"""

from typing import Literal

from app.core.config import settings

Route = Literal["graph", "vector"]


def route(question: str, profile: dict) -> Route:
    mode = settings.judge_mode
    if mode == "always_vector":
        return "vector"
    # "always_graph" (default) and "heuristic" (not yet implemented — falls back to the
    # stated default rather than guessing at classification logic) both resolve to "graph".
    return "graph"
