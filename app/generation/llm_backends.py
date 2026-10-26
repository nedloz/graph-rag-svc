"""Streaming answer generation — ported verbatim (logic unchanged, only import paths
and config access adjusted) from services/llm-svc/app/main.py's stream_model() and its
per-provider backends. This is a separate, streaming-oriented module from
app/ontology/llm_backend.py (non-streaming, used for triplet extraction) — they serve
different callers and are not meant to be merged.
"""

import asyncio
import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.mtls import httpx_kwargs

logger = logging.getLogger("graph-rag-svc.generation.llm_backends")

_local_pipeline = None
_local_tokenizer = None


def _http_timeout() -> httpx.Timeout:
    """Таймаут для потоковых вызовов LLM.

    read-таймаут для стриминга означает «столько секунд без единого нового фрагмента»:
    зависший провайдер обрывается, а длинная, но продвигающаяся генерация — нет. Раньше
    здесь стояло `timeout=None`, из-за чего зависший вызов держал воркер бесконечно и при
    `LLM_HTTP_CONCURRENCY=1` останавливал всю очередь.
    """
    return httpx.Timeout(
        settings.llm_request_timeout_sec,
        connect=settings.llm_connect_timeout_sec,
    )


async def stream_model(messages: list[dict[str, str]], usage: dict[str, Any] | None = None):
    usage = usage if usage is not None else {}
    provider = settings.llm_provider
    if provider == "openai":
        async for item in _stream_openai(messages, usage):
            yield item
    elif provider == "ollama":
        async for item in _stream_ollama(messages, usage):
            yield item
    elif provider == "local":
        # ВНИМАНИЕ: только для локальной разработки, см. _stream_local_transformers().
        async for item in _stream_local_transformers(messages, usage):
            yield item
    else:
        async for item in _stream_stub(messages, usage):
            yield item


async def _stream_openai(messages: list[dict[str, str]], usage: dict[str, Any]):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield {"type": "error", "content": "Библиотека openai не установлена"}
        return

    if not settings.openai_api_key:
        yield {"type": "error", "content": "Не задан OPENAI_API_KEY"}
        return

    client_ai = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.llm_request_timeout_sec)
    try:
        stream = await client_ai.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage.update(chunk.usage.model_dump())
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                yield {"type": "chunk", "content": content}
    except Exception as exc:
        yield {"type": "error", "content": str(exc)}


async def _stream_ollama(messages: list[dict[str, str]], usage: dict[str, Any]):
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    payload = {"model": settings.ollama_model, "messages": messages, "stream": True}
    try:
        async with httpx.AsyncClient(timeout=_http_timeout(), **httpx_kwargs()) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    yield {"type": "error", "content": f"Ollama error {resp.status_code}: {await resp.aread()}"}
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield {"type": "chunk", "content": chunk}
                    if data.get("done"):
                        usage["prompt_tokens"] = data.get("prompt_eval_count")
                        usage["completion_tokens"] = data.get("eval_count")
                        if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    except Exception as exc:
        yield {"type": "error", "content": str(exc)}


def _escape_local_prompt_content(text: str) -> str:
    return (
        text.replace("<system>", "&lt;system&gt;")
        .replace("</system>", "&lt;/system&gt;")
        .replace("<user>", "&lt;user&gt;")
        .replace("</user>", "&lt;/user&gt;")
        .replace("<assistant>", "&lt;assistant&gt;")
        .replace("</assistant>", "&lt;/assistant&gt;")
        .replace("<developer>", "&lt;developer&gt;")
        .replace("</developer>", "&lt;/developer&gt;")
        .replace("<tool>", "&lt;tool&gt;")
        .replace("</tool>", "&lt;/tool&gt;")
    )


async def _stream_local_transformers(messages: list[dict[str, str]], usage: dict[str, Any]):
    """ТОЛЬКО ДЛЯ ЛОКАЛЬНОЙ РАЗРАБОТКИ. Не использовать в прод-среде с остальными контейнерами.

    Это единственный backend, который грузит модель в процесс самого сервиса, а не в отдельный
    контейнер. Последствия в проде: пересборка или рестарт `graph-rag-svc` заново загружает
    модель; память под модель конкурирует с памятью самого сервиса; образ тянет
    `torch`/`transformers`/`accelerate`. Для локальной модели в проде используйте
    `LLM_PROVIDER=ollama` — он обслуживается отдельным контейнером, как и `embedding-svc`.
    """
    global _local_pipeline, _local_tokenizer
    try:
        import torch
        from threading import Thread
        from transformers import AutoTokenizer, TextIteratorStreamer, pipeline
    except ImportError:
        # Не сбой окружения, а поведение по умолчанию: torch/transformers лежат в
        # requirements-local.txt и в образ не попадают, пока не собрать с WITH_LOCAL_LLM=true.
        yield {
            "type": "error",
            "content": (
                "LLM_PROVIDER=local требует torch/transformers. Они не входят в образ по "
                "умолчанию: пересоберите graph-rag-svc с WITH_LOCAL_LLM=true либо "
                "используйте LLM_PROVIDER=ollama."
            ),
        }
        return

    if _local_pipeline is None:
        try:
            logger.info("Loading local model %s", settings.transformers_model)
            _local_tokenizer = AutoTokenizer.from_pretrained(settings.transformers_model)
            _local_pipeline = pipeline(
                "text-generation",
                model=settings.transformers_model,
                tokenizer=_local_tokenizer,
                torch_dtype=torch.float16 if settings.device == "cuda" else torch.float32,
                device=0 if settings.device == "cuda" else -1,
            )
        except Exception as exc:
            yield {"type": "error", "content": f"Ошибка загрузки модели: {exc}"}
            return

    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = _escape_local_prompt_content(m.get("content", ""))
        prompt += f"<{role}>\n{content}\n</{role}>\n"
    prompt += "<assistant>\n"

    prompt_tokens = _local_tokenizer(prompt, return_tensors="pt")
    usage["prompt_tokens"] = int(prompt_tokens.input_ids.shape[-1])

    streamer = TextIteratorStreamer(_local_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=prompt_tokens.input_ids.to(_local_pipeline.device),
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=_local_tokenizer.eos_token_id,
    )
    thread = Thread(target=_local_pipeline.model.generate, kwargs=generation_kwargs)
    thread.start()

    completion_token_estimate = 0
    for new_text in streamer:
        if new_text:
            completion_token_estimate += max(1, len(_local_tokenizer.encode(new_text, add_special_tokens=False)))
            yield {"type": "chunk", "content": new_text}
        await asyncio.sleep(0)

    usage["completion_tokens"] = completion_token_estimate
    usage["total_tokens"] = usage.get("prompt_tokens", 0) + completion_token_estimate


async def _stream_stub(messages: list[dict[str, str]], usage: dict[str, Any]):
    stub_text = "Я работаю в режиме заглушки (stub). Настройте LLM_PROVIDER в .env."
    for word in stub_text.split(" "):
        yield {"type": "chunk", "content": word + " "}
        await asyncio.sleep(0.05)
    usage["prompt_tokens"] = len(" ".join(m["content"] for m in messages).split())
    usage["completion_tokens"] = len(stub_text.split())
