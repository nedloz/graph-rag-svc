"""GraphRAG отключён (спецификация 2026-10-27-notebook-pipeline-design): приложение не
загружает графовую логику, не отдаёт /internal/graph/*, а стартовая проверка размерности
не смотрит на колонки схемы graph.*."""

import json
import subprocess
import sys

import pytest

GRAPH_MODULES = (
    "app.graph.store",
    "app.graph.builder",
    "app.graph.curation",
    "app.graph.resolution",
    "app.ontology.extractor",
    "app.ontology.llm_backend",
    "app.ontology.prompts",
    "app.ontology.schema",
    "app.api.graph_admin",
    "app.retrieval.graph_retriever",
)


def test_importing_app_main_does_not_load_graph_modules():
    # Отдельный процесс: sys.modules текущего уже может быть загрязнён другими тестами.
    code = (
        "import sys, app.main\n"
        f"loaded = [m for m in {GRAPH_MODULES!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


# Приложение целиком (с его метриками в общем реестре Prometheus) поднимается в отдельном
# процессе: иначе повторная регистрация метрик ломает tests/test_metrics.py. TestClient
# создаётся без `with`, поэтому startup-события (очередь, мост, проверка эмбеддингов) не
# запускаются и БД не нужна.
_PROBE = """
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
print(json.dumps({
    "statuses": {
        f"{m} {p}": getattr(client, m)(p).status_code
        for m, p in %(requests)r
    },
    "paths": [getattr(route, "path", "") for route in app.routes],
    "ping": client.get("/ping").json(),
}))
"""

_GRAPH_REQUESTS = [
    ("get", "/internal/graph/build"),
    ("post", "/internal/graph/build"),
    ("get", "/internal/graph/entities"),
    ("get", "/internal/graph/relations"),
]


@pytest.fixture(scope="module")
def probe() -> dict:
    code = _PROBE % {"requests": _GRAPH_REQUESTS}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_graph_routes_are_gone(probe):
    assert set(probe["statuses"].values()) == {404}


def test_no_graph_routes_registered(probe):
    assert not [p for p in probe["paths"] if p.startswith("/internal/graph")]


def test_ping_has_no_judge_mode(probe):
    assert "judge_mode" not in probe["ping"]


def test_startup_checks_cover_only_library_columns():
    from app.core import startup_checks

    assert startup_checks._VECTOR_COLUMNS == (("library.chunk_embeddings", "embedding"),)


def test_graph_build_api_disabled_by_default():
    from app.core.config import Settings

    assert Settings(_env_file=None).graph_build_api_enabled is False
