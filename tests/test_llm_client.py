"""LlmClient: OpenAI-совместимый клиент llm-proxy поверх настоящего `openai.AsyncOpenAI`.

Сеть подменяется на уровне httpx (MockTransport, приходит через `httpx_kwargs()` — тот же путь,
которым в боевом режиме приходит контекст mTLS), поэтому проверяется и разбор SSE клиентом openai.
"""

import asyncio
import inspect
import json

import httpx
import pytest

from app.generation import llm_client
from app.generation.llm_client import LlmClient
from app.pipeline.llm import LlmChat

BASE_URL = "http://llm-proxy:8080/v1"


def _completion(content: str, usage: dict | None) -> dict:
    body = {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _sse(*events: dict) -> bytes:
    lines = [f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _delta(content: str) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "qwen",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


USAGE = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
USAGE_CHUNK = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "qwen", "choices": [], "usage": USAGE}


class Proxy:
    """Подделка llm-proxy: отвечает обработчиком и запоминает запросы и выдачи httpx_kwargs()."""

    def __init__(self, monkeypatch, handler) -> None:
        self.requests: list[httpx.Request] = []
        self.kwargs_calls = 0

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        def fake_httpx_kwargs() -> dict:
            self.kwargs_calls += 1
            return {"transport": httpx.MockTransport(wrapped)}

        monkeypatch.setattr(llm_client, "httpx_kwargs", fake_httpx_kwargs)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def _client() -> LlmClient:
    return LlmClient(BASE_URL, "default", max_retries=0)


async def _collect(stream) -> list[dict]:
    return [event async for event in stream]


def test_chat_returns_text_and_total_tokens(monkeypatch):
    proxy = Proxy(monkeypatch, lambda r: httpx.Response(200, json=_completion("{\"ranked\": []}", USAGE)))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    text, tokens = asyncio.run(_client().chat(messages, temperature=0.0, max_tokens=750))

    assert (text, tokens) == ("{\"ranked\": []}", 13)
    request = proxy.requests[0]
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    # Ключа у сервиса нет: прокси подставляет свой, сюда уходит заглушка.
    assert request.headers["authorization"] == "Bearer unused"
    body = proxy.body()
    assert body["model"] == "default"
    assert body["messages"] == messages
    assert body["temperature"] == 0.0 and body["max_tokens"] == 750
    assert not body.get("stream")


def test_chat_without_usage_returns_none_tokens(monkeypatch):
    Proxy(monkeypatch, lambda r: httpx.Response(200, json=_completion("ответ", None)))
    assert asyncio.run(_client().chat([], temperature=0.0, max_tokens=10)) == ("ответ", None)


def test_chat_error_propagates(monkeypatch):
    # Ошибку вызова в реранке и переписывании не глушим: process_job считает её сбоем ретривала.
    Proxy(monkeypatch, lambda r: httpx.Response(503, json={"error": {"message": "no config"}}))
    with pytest.raises(Exception):
        asyncio.run(_client().chat([], temperature=0.0, max_tokens=10))


def test_stream_yields_chunks_and_fills_usage(monkeypatch):
    proxy = Proxy(
        monkeypatch,
        lambda r: httpx.Response(
            200,
            content=_sse(_delta("При"), _delta(""), _delta("вет"), USAGE_CHUNK),
            headers={"content-type": "text/event-stream"},
        ),
    )
    usage: dict = {}

    events = asyncio.run(_collect(_client().stream([{"role": "user", "content": "q"}], usage=usage)))

    assert events == [{"type": "chunk", "content": "При"}, {"type": "chunk", "content": "вет"}]
    assert usage == USAGE
    body = proxy.body()
    assert body["model"] == "default"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    # Параметры финального ответа ноутбука по умолчанию.
    assert body["temperature"] == 0.3 and body["max_tokens"] == 1100


def test_stream_connection_error_becomes_error_event(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    Proxy(monkeypatch, refuse)

    events = asyncio.run(_collect(_client().stream([{"role": "user", "content": "q"}])))

    assert len(events) == 1 and events[0]["type"] == "error"
    assert events[0]["content"]


def test_stream_http_error_becomes_error_event(monkeypatch):
    Proxy(monkeypatch, lambda r: httpx.Response(502, json={"error": {"message": "upstream down"}}))
    events = asyncio.run(_collect(_client().stream([{"role": "user", "content": "q"}])))
    assert [e["type"] for e in events] == ["error"]


def test_tls_kwargs_are_built_per_call(monkeypatch):
    # Сертификат живёт сутки: контекст собирается на каждый вызов, а не один раз на процесс.
    proxy = Proxy(
        monkeypatch,
        lambda r: httpx.Response(200, json=_completion("ok", USAGE))
        if not json.loads(r.content).get("stream")
        else httpx.Response(200, content=_sse(_delta("a"), USAGE_CHUNK), headers={"content-type": "text/event-stream"}),
    )
    client = _client()

    asyncio.run(client.chat([], temperature=0.0, max_tokens=10))
    asyncio.run(client.chat([], temperature=0.0, max_tokens=10))
    asyncio.run(_collect(client.stream([])))

    assert proxy.kwargs_calls == 3
    assert len(proxy.requests) == 3


def test_timeouts_come_from_settings(monkeypatch):
    proxy = Proxy(monkeypatch, lambda r: httpx.Response(200, json=_completion("ok", None)))
    monkeypatch.setattr(llm_client.settings, "llm_request_timeout_sec", 77)
    monkeypatch.setattr(llm_client.settings, "llm_connect_timeout_sec", 4)

    asyncio.run(_client().chat([], temperature=0.0, max_tokens=10))

    timeout = proxy.requests[0].extensions["timeout"]
    assert timeout["connect"] == 4 and timeout["read"] == 77


def test_satisfies_pipeline_protocol():
    expected = inspect.signature(LlmChat.chat)
    actual = inspect.signature(LlmClient.chat)
    assert list(actual.parameters) == list(expected.parameters)
