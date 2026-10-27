"""Интерфейс LLM, который нужен пайплайну. Реализацию (клиент llm-proxy) даёт вызывающий."""

from typing import Protocol


class LlmChat(Protocol):
    async def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, int | None]:
        """Один ответ модели целиком: (текст, total_tokens или None)."""
        ...
