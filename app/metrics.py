"""Метрики Prometheus: HTTP-метрики приложения и бизнес-метрики сервиса.

/metrics отдаёт только prometheus (политика прокси, infra/mtls/*.conf); наружу через шлюз его
нет. HTTP-метрики группируются по шаблону маршрута, а не по реальному пути, поэтому число
значений метки handler ограничено числом маршрутов.
"""

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator


def setup_metrics(app: FastAPI) -> None:
    """Подключает сбор HTTP-метрик и маршрут GET /metrics."""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
    )
    instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=False)


GENERATE_REQUESTS = Counter(
    "pochemuchnik_generate_requests_total",
    "Принятые запросы /generate по режиму доставки",
    ["mode"],  # sync | ws_bridge
)
JUDGE_DECISIONS = Counter(
    "pochemuchnik_judge_decisions_total",
    "Маршрут поиска контекста",
    ["route"],  # graph | vector | graph_to_vector (пустой граф, сработал откат на вектор)
)
STAGE_DURATION = Histogram(
    "pochemuchnik_stage_duration_seconds",
    "Длительность этапов обработки запроса",
    ["stage"],  # retrieval | generation
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160),
)
CONTEXT_CHUNKS = Histogram(
    "pochemuchnik_context_chunks",
    "Число чанков в контексте ответа",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
)
EMPTY_CONTEXT = Counter(
    "pochemuchnik_empty_context_total",
    "Запросы, для которых не нашлось ни одного чанка",
)
LLM_ERRORS = Counter(
    "pochemuchnik_llm_errors_total",
    "Ошибки генерации, пришедшие от провайдера LLM",
)
RETRIEVAL_FAILURES = Counter(
    "pochemuchnik_retrieval_failures_total",
    "Сбои поиска контекста (запрос завершается ошибкой)",
)
