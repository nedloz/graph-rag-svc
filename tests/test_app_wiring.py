"""Сборка сервиса после перехода на пайплайн ноутбука: удалённые модули и настройки больше нигде
не упоминаются, все рабочие модули импортируются, /ping и старт приложения подключены к llm-proxy
и индексу базы знаний.

Приложение целиком поднимается в отдельном процессе: метрики регистрируются в общем реестре
Prometheus, и повторная регистрация в этом процессе сломала бы tests/test_metrics.py.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DELETED_MODULES = (
    "app.retrieval.judge",
    "app.retrieval.orchestrator",
    "app.retrieval.vector_retriever",
    "app.retrieval.context_assembler",
    "app.generation.llm_backends",
    "app.generation.prompt_builder",
)

# Упоминания удалённых модулей, файлов и настроек. README обновляет отдельная задача.
FORBIDDEN = re.compile(
    r"llm_backends|prompt_builder|context_assembler|vector_retriever|retrieval\.orchestrator|"
    r"retrieval import [^\n]*\b(judge|orchestrator)\b|retrieval\.judge|requirements-local|WITH_LOCAL_LLM|"
    r"llm_provider|LLM_PROVIDER|openai_api_key|OPENAI_|openai_model|ollama_url|ollama_model|OLLAMA_|"
    r"transformers_model|TRANSFORMERS_MODEL|settings\.device|^DEVICE=|JUDGE_MODE|warn_if_development_only_provider|"
    r"settings\.(max_chunk_chars|max_context_chars|max_question_chars|max_history_chars|max_history_message_chars)|"
    r"^(MAX_CHUNK_CHARS|MAX_CONTEXT_CHARS|MAX_QUESTION_CHARS|MAX_HISTORY_CHARS)=|MAX_HISTORY_CHARS",
    re.MULTILINE,
)


def _subprocess(code: str, env: dict | None = None) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, **(env or {})},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_deleted_modules_are_gone():
    assert [m for m in DELETED_MODULES if importlib.util.find_spec(m) is not None] == []
    assert not (ROOT / "requirements-local.txt").exists()


def test_nothing_mentions_deleted_modules_or_settings():
    files = [*ROOT.glob("app/**/*.py"), *ROOT.glob("cli/**/*.py"), *ROOT.glob("tests/**/*.py"),
             ROOT / "Dockerfile", ROOT / ".env.example", ROOT / "requirements.txt"]
    this = Path(__file__).resolve()
    hits = []
    for path in files:
        if path.resolve() == this:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if FORBIDDEN.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert hits == []


def test_every_working_module_imports():
    # Модули GraphRAG помечены DISABLED и при работе сервиса не загружаются — их не импортируем.
    code = r"""
import importlib, pkgutil, json, app
failed = {}
count = 0
for info in pkgutil.walk_packages(app.__path__, "app."):
    spec = importlib.util.find_spec(info.name)
    source = open(spec.origin, encoding="utf-8").read(400) if spec.origin and spec.origin.endswith(".py") else ""
    if '"' * 3 + "DISABLED" in source:
        continue
    try:
        importlib.import_module(info.name)
        count += 1
    except Exception as exc:
        failed[info.name] = repr(exc)
print(json.dumps({"failed": failed, "count": count}))
"""
    result = json.loads(_subprocess(code))
    assert result["failed"] == {}
    assert result["count"] > 20


def test_ping_reports_llm_base_url_instead_of_provider():
    code = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "print(json.dumps(TestClient(app).get('/ping').json()))\n"
    )
    ping = json.loads(_subprocess(code, env={"LLM_BASE_URL": "https://llm-proxy:8443/v1"}))
    assert "llm_provider" not in ping
    assert ping["llm_base_url"] == "https://llm-proxy:8443/v1"


def test_settings_defaults(monkeypatch):
    from app.core.config import Settings

    for name in ("EMBEDDING_DIM", "LLM_BASE_URL", "LLM_MODEL", "INDEX_REFRESH_INTERVAL_SEC"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://llm-proxy:8080/v1"
    assert s.llm_model == "default"
    assert s.index_refresh_interval_sec == 60
    assert s.embedding_dim == 768
    for removed in ("llm_provider", "openai_api_key", "openai_model", "ollama_url", "ollama_model",
                    "transformers_model", "device", "judge_mode"):
        assert not hasattr(s, removed), removed


def test_process_llm_client_points_at_proxy():
    code = (
        "from app.generation import job_queue\n"
        "print(job_queue.llm.base_url + ' ' + job_queue.llm.model)\n"
    )
    out = _subprocess(code, env={"LLM_BASE_URL": "http://llm-proxy:8080/v1", "LLM_MODEL": "default"})
    assert out == "http://llm-proxy:8080/v1 default"


def test_startup_loads_index_after_embedding_check_and_shutdown_cancels_refresh():
    code = r"""
import asyncio, json
import app.main as m

order = []

async def verify():
    order.append("verify")

async def load(session_factory):
    order.append("load")
    assert session_factory is m.SessionLocal

async def refresh(session_factory):
    order.append("refresh")
    return False

async def bridge_start():
    order.append("bridge")

async def queue_start():
    order.append("queue")

async def noop():
    pass

m.verify_embedding_setup = verify
m.knowledge_index.load = load
m.knowledge_index.refresh_if_stale = refresh
m.bridge.start = bridge_start
m.bridge.stop = noop
m.job_queue.start = queue_start
m.job_queue.stop = noop
m.settings.index_refresh_interval_sec = 0.01

async def scenario():
    await m.startup_event()
    task = m.app.state.index_refresh_task
    for _ in range(200):
        if "refresh" in order:
            break
        await asyncio.sleep(0.01)
    await m.shutdown_event()
    return task

task = asyncio.run(scenario())
print(json.dumps({"order": [o for i, o in enumerate(order) if o != "refresh" or o not in order[:i]],
                  "cancelled": task.cancelled()}))
"""
    result = json.loads(_subprocess(code))
    assert result["order"][:2] == ["verify", "load"]
    assert {"bridge", "queue", "refresh"} <= set(result["order"])
    assert result["cancelled"] is True
