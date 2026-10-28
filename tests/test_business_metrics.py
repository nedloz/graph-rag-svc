import asyncio
import uuid
from types import SimpleNamespace

from prometheus_client import REGISTRY

from app.api import http as http_api
from app.generation import job_queue
from app.pipeline.index import Candidate
from app.pipeline.quick_answers import QuickAnswer
from app.pipeline.service import RetrievalResult


def sample(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def candidate(idx: int) -> Candidate:
    return Candidate(idx=idx, chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), chunk_index=idx,
                     title="Документ", category="", section="", text="t", rerank_score=0.9)


def hybrid(selected) -> RetrievalResult:
    return RetrievalResult(quick_answer=None, messages=[], selected=selected, links=[], rerank_tokens=None,
                           query_model="e5-base", retrieval_ms=1.0)


def quick() -> RetrievalResult:
    return RetrievalResult(quick_answer=QuickAnswer(answer="Готовый ответ", question="q", rrf_score=0.05),
                           messages=[], selected=[], links=[], rerank_tokens=None, query_model="e5-base",
                           retrieval_ms=1.0)


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


class Llm:
    def __init__(self, events) -> None:
        self.events = events

    async def stream(self, _messages, usage=None, temperature=0.3, max_tokens=1100):
        for event in self.events:
            yield event


def patch_pipeline(monkeypatch, events, result=None):
    async def run_retrieval(*_args):
        return result if result is not None else hybrid([])

    async def record(*_args, **_kwargs):
        pass

    monkeypatch.setattr(job_queue, "run_retrieval", run_retrieval)
    monkeypatch.setattr(job_queue, "llm", Llm(events))
    monkeypatch.setattr(job_queue.retrieval_log, "record", record)


def run_job():
    asyncio.run(job_queue.process_job(make_job(), Sink()))


def test_hybrid_route_with_hits_is_counted_with_chunks_and_stage_time(monkeypatch):
    patch_pipeline(monkeypatch, [], result=hybrid([candidate(0), candidate(1)]))
    decisions = sample("pochemuchnik_judge_decisions_total", route="hybrid")
    stage = sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval")
    chunks_count = sample("pochemuchnik_context_chunks_count")
    chunks_sum = sample("pochemuchnik_context_chunks_sum")

    run_job()

    assert sample("pochemuchnik_judge_decisions_total", route="hybrid") == decisions + 1
    assert sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval") == stage + 1
    assert sample("pochemuchnik_context_chunks_count") == chunks_count + 1
    assert sample("pochemuchnik_context_chunks_sum") == chunks_sum + 2


def test_quick_answer_is_counted_by_route_without_context_or_generation(monkeypatch):
    patch_pipeline(monkeypatch, [], result=quick())
    decisions = sample("pochemuchnik_judge_decisions_total", route="quick_answer")
    stage = sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval")
    generation = sample("pochemuchnik_stage_duration_seconds_count", stage="generation")
    chunks_count = sample("pochemuchnik_context_chunks_count")
    empty = sample("pochemuchnik_empty_context_total")

    run_job()

    assert sample("pochemuchnik_judge_decisions_total", route="quick_answer") == decisions + 1
    assert sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval") == stage + 1
    # Быстрому ответу контекст не нужен: он не считается ни «пустым контекстом», ни генерацией.
    assert sample("pochemuchnik_stage_duration_seconds_count", stage="generation") == generation
    assert sample("pochemuchnik_context_chunks_count") == chunks_count
    assert sample("pochemuchnik_empty_context_total") == empty


def test_no_chunks_at_all_counts_empty_context(monkeypatch):
    patch_pipeline(monkeypatch, [], result=hybrid([]))
    before = sample("pochemuchnik_empty_context_total")
    run_job()
    assert sample("pochemuchnik_empty_context_total") == before + 1


def test_found_context_does_not_count_as_empty(monkeypatch):
    patch_pipeline(monkeypatch, [], result=hybrid([candidate(0)]))
    before = sample("pochemuchnik_empty_context_total")
    run_job()
    assert sample("pochemuchnik_empty_context_total") == before


def test_llm_error_event_counts_llm_error_and_generation_stage(monkeypatch):
    patch_pipeline(monkeypatch, [{"type": "error", "content": "boom"}])
    errors = sample("pochemuchnik_llm_errors_total")
    stage = sample("pochemuchnik_stage_duration_seconds_count", stage="generation")

    run_job()

    assert sample("pochemuchnik_llm_errors_total") == errors + 1
    assert sample("pochemuchnik_stage_duration_seconds_count", stage="generation") == stage + 1


def test_successful_generation_does_not_count_llm_error(monkeypatch):
    patch_pipeline(monkeypatch, [{"type": "chunk", "content": "ok"}])
    errors = sample("pochemuchnik_llm_errors_total")
    run_job()
    assert sample("pochemuchnik_llm_errors_total") == errors


def test_retrieval_failure_is_counted_and_still_raises(monkeypatch):
    patch_pipeline(monkeypatch, [])

    async def broken(*_args):
        raise ValueError("db down")

    monkeypatch.setattr(job_queue, "run_retrieval", broken)
    before = sample("pochemuchnik_retrieval_failures_total")
    stage = sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval")
    try:
        run_job()
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert sample("pochemuchnik_retrieval_failures_total") == before + 1
    assert sample("pochemuchnik_stage_duration_seconds_count", stage="retrieval") == stage


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
