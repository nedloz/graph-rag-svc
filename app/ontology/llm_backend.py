"""Non-streaming LLM completion used for ontology extraction — a single JSON response
is needed, not a token stream, so this is deliberately simpler than
app/generation/llm_backends.py (ported from llm-svc), which streams answers.

Extraction can use a different (typically smaller/cheaper) model than answer generation,
via GRAPH_EXTRACTION_MODEL; when unset, it falls back to the same provider/model settings
used for generation.
"""

import asyncio
import json
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger("graph-rag-svc.ontology.llm_backend")

_local_pipeline = None
_local_tokenizer = None


async def complete(messages: list[dict[str, str]]) -> str:
    provider = settings.llm_provider
    if provider == "openai":
        return await _complete_openai(messages)
    if provider == "ollama":
        return await _complete_ollama(messages)
    if provider == "local":
        return await _complete_local_transformers(messages)
    return _complete_stub(messages)


async def _complete_openai(messages: list[dict[str, str]]) -> str:
    from openai import AsyncOpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = settings.graph_extraction_model or settings.openai_model
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.llm_request_timeout_sec)
    response = await client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


async def _complete_ollama(messages: list[dict[str, str]]) -> str:
    model = settings.graph_extraction_model or settings.ollama_model
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(settings.llm_request_timeout_sec, connect=settings.llm_connect_timeout_sec)
    ) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    return data.get("message", {}).get("content", "")


async def _complete_local_transformers(messages: list[dict[str, str]]) -> str:
    """ТОЛЬКО ДЛЯ ЛОКАЛЬНОЙ РАЗРАБОТКИ. Не использовать в прод-среде с остальными контейнерами.

    См. пояснение в app/generation/llm_backends.py::_stream_local_transformers — модель
    грузится в процесс сервиса, а не в отдельный контейнер. В проде используйте `ollama`.
    """
    global _local_pipeline, _local_tokenizer

    # torch/transformers лежат в requirements-local.txt и в образ по умолчанию не ставятся,
    # поэтому импорт здесь может не пройти. Сообщаем причину явно: иначе наружу уходил бы
    # голый ImportError, по которому не видно, что дело в сборке образа, а не в коде.
    try:
        import torch
        from transformers import AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "LLM_PROVIDER=local требует torch/transformers. Они не входят в образ по "
            "умолчанию: пересоберите graph-rag-svc с WITH_LOCAL_LLM=true либо используйте "
            "LLM_PROVIDER=ollama."
        ) from exc

    model_id = settings.graph_extraction_model or settings.transformers_model

    if _local_pipeline is None:
        logger.info("Loading local extraction model %s", model_id)
        _local_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _local_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=_local_tokenizer,
            torch_dtype=torch.float16 if settings.device == "cuda" else torch.float32,
            device=0 if settings.device == "cuda" else -1,
        )

    prompt = ""
    for m in messages:
        prompt += f"<{m.get('role', 'user')}>\n{m.get('content', '')}\n</{m.get('role', 'user')}>\n"
    prompt += "<assistant>\n"

    def _generate() -> str:
        output = _local_pipeline(
            prompt,
            max_new_tokens=1000,
            do_sample=False,
            temperature=0.1,
            pad_token_id=_local_tokenizer.eos_token_id,
            return_full_text=False,
        )
        return output[0]["generated_text"]

    return await asyncio.get_running_loop().run_in_executor(None, _generate)


def _complete_stub(messages: list[dict[str, str]]) -> str:
    return json.dumps({"triplets": []})
