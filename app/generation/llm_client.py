"""Клиент LLM: OpenAI-совместимый `POST /v1/chat/completions` контейнера `llm-proxy`.

Все вызовы модели (реранк, переписывание вопроса, ответ) идут сюда. Провайдера и ключа в сервисе
больше нет: прокси сам выбирает роутер или Ollama, подставляет модель режима вместо присланной
`model: "default"` и свой `Authorization` (main/infra/llm-proxy/README.md). Поэтому `api_key` —
заглушка, а `model` по умолчанию — `default`.

Клиент `openai.AsyncOpenAI` создаётся на каждый вызов вместе со своим `httpx.AsyncClient`: контекст
mTLS (`app/mtls.py`) собирается при создании клиента, а сертификат обновляется раз в сутки —
клиент на весь процесс так и держал бы истёкший.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.mtls import httpx_kwargs

logger = logging.getLogger("graph-rag-svc.generation.llm_client")

# Прокси подставляет свой ключ; без непустого значения клиент openai не создаётся.
_PLACEHOLDER_KEY = "unused"
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


class LlmClient:
    """Реализация `app.pipeline.llm.LlmChat` (+ потоковый ответ) поверх llm-proxy."""

    def __init__(self, base_url: str, model: str, *, max_retries: int = 2) -> None:
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries

    def _client(self) -> AsyncOpenAI:
        # read-таймаут для стриминга — «столько секунд без единого нового фрагмента»: зависший
        # провайдер обрывается, а длинная, но продвигающаяся генерация — нет. Без таймаута
        # зависший вызов держал бы воркер, а при LLM_HTTP_CONCURRENCY=1 — всю очередь.
        timeout = httpx.Timeout(settings.llm_request_timeout_sec, connect=settings.llm_connect_timeout_sec)
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=_PLACEHOLDER_KEY,
            http_client=httpx.AsyncClient(timeout=timeout, **httpx_kwargs()),
            timeout=timeout,
            max_retries=self.max_retries,
        )

    async def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, int | None]:
        """Ответ целиком: (текст, total_tokens или None). Ошибки вызова не глушатся."""
        async with self._client() as client:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        text = response.choices[0].message.content if response.choices else None
        tokens = response.usage.total_tokens if response.usage else None
        return text or "", tokens

    async def stream(
        self,
        messages: list[dict],
        usage: dict[str, Any] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1100,
    ) -> AsyncIterator[dict]:
        """События `{"type": "chunk"|"error", "content": str}`; токены — в переданный `usage`.

        Любая ошибка (недоступный прокси, ответ 5xx, обрыв потока) — одно событие `error`, а не
        исключение: вызывающий отдаёт его в `sink.error`, как раньше делал `stream_model`.
        """
        usage = usage if usage is not None else {}
        try:
            async with self._client() as client:
                stream = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async with stream:
                    async for chunk in stream:
                        if getattr(chunk, "usage", None):
                            usage.update({k: getattr(chunk.usage, k, None) for k in _USAGE_FIELDS})
                        content = chunk.choices[0].delta.content if chunk.choices else None
                        if content:
                            yield {"type": "chunk", "content": content}
        except Exception as exc:
            logger.warning("Ошибка вызова llm-proxy (%s): %s", self.base_url, exc)
            yield {"type": "error", "content": str(exc) or exc.__class__.__name__}
