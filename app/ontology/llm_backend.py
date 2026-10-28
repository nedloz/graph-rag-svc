"""DISABLED: GraphRAG отключён (спецификация 2026-10-27-notebook-pipeline-design). Код не поддерживается и не вызывается.

Non-streaming LLM completion used for ontology extraction — a single JSON response
is needed, not a token stream. Как и все вызовы модели в сервисе, идёт через llm-proxy
(app/generation/llm_client.py): провайдера и ключа в сервисе больше нет, модель выбирает
прокси. GRAPH_EXTRACTION_MODEL остаётся только меткой прогона извлечения (app/graph/builder.py).
"""

from app.core.config import settings
from app.generation.llm_client import LlmClient

EXTRACTION_TEMPERATURE = 0.1
EXTRACTION_MAX_TOKENS = 1000


async def complete(messages: list[dict[str, str]]) -> str:
    text, _ = await LlmClient(settings.llm_base_url, settings.llm_model).chat(
        messages, temperature=EXTRACTION_TEMPERATURE, max_tokens=EXTRACTION_MAX_TOKENS
    )
    return text
