import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

from prometheus_client import REGISTRY

from app.api import http as http_api
from app.generation import job_queue
from app.retrieval import orchestrator
from app.retrieval.types import RetrievedChunk


def sample(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def chunk() -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), doc_id=uuid.uuid4(), text="t", score=0.9, source="graph")


@asynccontextmanager
async def session_factory():
    yield object()


def run_retrieve(monkeypatch, *, vector):
    async def vector_retrieve(*_args):
        return vector, "bge-m3"

    monkeypatch.setattr(orchestrator.vector_retriever, "retrieve", vector_retrieve)
    return asyncio.run(orchestrator.retrieve(session_factory, "вопрос", {}))


def test_vector_route_with_hits_is_counted_with_chunks_and_stage_time(monkeypatch):
    decisions = sample("pochemuchnik_judge_decisions_total", route="vector")
    stage = sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval")
    chunks_count = sample("pochemuchnik_context_chunks_count")
    chunks_sum = sample("pochemuchnik_context_chunks_sum")

    run_retrieve(monkeypatch, vector=[chunk(), chunk()])

    assert sample("pochemuchnik_judge_decisions_total", route="vector") == decisions + 1
    assert sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval") == stage + 1
    assert sample("pochemuchnik_context_chunks_count") == chunks_count + 1
    assert sample("pochemuchnik_context_chunks_sum") == chunks_sum + 2


def test_no_chunks_at_all_counts_empty_context(monkeypatch):
    before = sample("pochemuchnik_empty_context_total")
    run_retrieve(monkeypatch, vector=[])
    assert sample("pochemuchnik_empty_context_total") == before + 1


def test_found_context_does_not_count_as_empty(monkeypatch):
    before = sample("pochemuchnik_empty_context_total")
    run_retrieve(monkeypatch, vector=[chunk()])
    assert sample("pochemuchnik_empty_context_total") == before


class Sink:
    async def started(self, _job):
        pass

    async def chunk(self, _job, _text):
        pass

    async def completed(self, _job, _usage):
        pass

    async def cancelled(self, _job, _usage):
        pass

    async def error(self, _job, _message):
        pass


def make_job():
    return SimpleNamespace(request_id="r1", question="q", profile={}, history=[], rag=None)


def patch_pipeline(monkeypatch, events):
    async def retrieve(*_args):
        return {"chunks": []}

    async def stream_model(_messages, usage):
        for event in events:
            yield event

    async def record(*_args, **_kwargs):
        pass

    monkeypatch.setattr(job_queue, "retrieve", retrieve)
    monkeypatch.setattr(job_queue, "build_messages", lambda _job: [])
    monkeypatch.setattr(job_queue, "stream_model", stream_model)
    monkeypatch.setattr(job_queue.retrieval_log, "record", record)


def test_llm_error_event_counts_llm_error_and_generation_stage(monkeypatch):
    patch_pipeline(monkeypatch, [{"type": "error", "content": "boom"}])
    errors = sample("pochemuchnik_llm_errors_total")
    stage = sample("pochemuchnik_stage_duration_seconds_count", stage="generation")

    asyncio.run(job_queue.process_job(make_job(), Sink()))

    assert sample("pochemuchnik_llm_errors_total") == errors + 1
    assert sample("pochemuchnik_stage_duration_seconds_count", stage="generation") == stage + 1


def test_successful_generation_does_not_count_llm_error(monkeypatch):
    patch_pipeline(monkeypatch, [{"type": "chunk", "content": "ok"}])
    errors = sample("pochemuchnik_llm_errors_total")
    asyncio.run(job_queue.process_job(make_job(), Sink()))
    assert sample("pochemuchnik_llm_errors_total") == errors


def test_retrieval_failure_is_counted_and_still_raises(monkeypatch):
    patch_pipeline(monkeypatch, [])

    async def broken(*_args):
        raise ValueError("db down")

    monkeypatch.setattr(job_queue, "retrieve", broken)
    before = sample("pochemuchnik_retrieval_failures_total")
    try:
        asyncio.run(job_queue.process_job(make_job(), Sink()))
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert sample("pochemuchnik_retrieval_failures_total") == before + 1


def test_generate_counts_accepted_requests_by_mode(monkeypatch):
    class Queue:
        def full(self):
            return False

    async def enqueue(*_args):
        return 1

    monkeypatch.setattr(http_api, "job_queue", SimpleNamespace(queue=Queue(), enqueue=enqueue))
    req = SimpleNamespace(response_mode="ws_bridge", request_id="r1", consumer=None,
                          delivery=SimpleNamespace(assistant_message_id="a1"))
    before = sample("pochemuchnik_generate_requests_total", mode="ws_bridge")

    asyncio.run(http_api.generate(req, caller="chat-svc"))

    assert sample("pochemuchnik_generate_requests_total", mode="ws_bridge") == before + 1
