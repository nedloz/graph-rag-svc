"""LLM-реранк (ячейки 19, 23): разбор JSON и откат к порядку RRF при любом сбое."""

import asyncio
import json
import uuid

import pytest

from app.pipeline.index import Candidate
from app.pipeline.rerank import RERANK_SYSTEM, parse_json_response, rerank
from pipeline_fakes import FakeLlm

DOC = uuid.uuid4()


def _cand(idx: int, rrf: float = 0.0, text: str = "текст") -> Candidate:
    return Candidate(
        idx=idx, chunk_id=uuid.uuid4(), document_id=DOC, chunk_index=idx, title=f"Док {idx}",
        category="Общие документы", section="Раздел", text=text, rrf_score=rrf,
    )


def test_parse_json_response_plain_and_fenced():
    assert parse_json_response('{"ranked": []}') == {"ranked": []}
    assert parse_json_response('```json\n{"ranked": [{"id": 1, "score": 0.5}]}\n```') == {
        "ranked": [{"id": 1, "score": 0.5}]
    }
    assert parse_json_response('```JSON\n{"a": 1}```') == {"a": 1}


def test_parse_json_response_text_around_object():
    assert parse_json_response('Вот результат: {"ranked": [{"id": 3}]} — готово') == {"ranked": [{"id": 3}]}


@pytest.mark.parametrize("garbage", ["не знаю", "", "{ broken", "[1, 2]", "}{"])
def test_parse_json_response_raises_value_error_on_garbage(garbage):
    with pytest.raises(ValueError):
        parse_json_response(garbage)


def _run(llm, candidates, top_n=18):
    return asyncio.run(rerank(llm, "Как пересдать экзамен?", candidates, top_n=top_n))


def test_rerank_valid_json_reorders_and_ignores_duplicates():
    candidates = [_cand(10, 0.03), _cand(11, 0.02), _cand(12, 0.01)]
    reply = '```json\n{"ranked": [{"id": 12, "score": 0.9}, {"id": 10, "score": 0.5}, {"id": 12, "score": 0.1}]}\n```'
    llm = FakeLlm((reply, 321))

    ranked, tokens = _run(llm, candidates)

    assert [c.idx for c in ranked] == [12, 10, 11]
    assert [c.rerank_score for c in ranked] == [0.9, 0.5, 0.0]
    assert ranked[0].rrf_score == 0.01  # RRF-оценка кандидата сохраняется
    assert tokens == 321
    # исходные кандидаты не меняются
    assert candidates[2].rerank_score == 0.0


def test_rerank_request_matches_notebook():
    candidates = [_cand(i, text="я" * 2000) for i in range(30)]
    llm = FakeLlm(('{"ranked": []}', None))

    _run(llm, candidates)

    call = llm.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 750
    system, user = call["messages"]
    assert system == {"role": "system", "content": RERANK_SYSTEM}
    assert user["role"] == "user"
    head, payload = user["content"].split("Фрагменты:\n", 1)
    assert head == "Вопрос:\nКак пересдать экзамен?\n\n"
    compact = json.loads(payload)
    assert len(compact) == 28
    assert compact[0] == {
        "id": 0, "title": "Док 0", "category": "Общие документы", "section": "Раздел", "text": "я" * 1100,
    }
    assert "\\u" not in payload  # ensure_ascii=False


@pytest.mark.parametrize(
    "reply",
    [
        "Извините, не могу",  # не JSON
        '{"ranked": [{"id": 999, "score": 1.0}, {"id": -5, "score": 0.7}]}',  # чужие id
        '{"ranked": [{"id": "abc", "score": 1.0}]}',  # id не число
        '{"ranked": [{"score": 1.0}]}',  # нет id
        '{"something": "else"}',  # нет ranked
    ],
)
def test_rerank_falls_back_to_rrf_order(reply):
    candidates = [_cand(10, 0.03), _cand(11, 0.02), _cand(12, 0.01)]
    llm = FakeLlm((reply, 50))

    ranked, tokens = _run(llm, candidates)

    assert [c.idx for c in ranked] == [10, 11, 12]
    assert all(c.rerank_score == 0.0 for c in ranked)
    assert tokens == 50


def test_rerank_cuts_to_top_n():
    candidates = [_cand(i) for i in range(30)]
    ranked, _ = _run(FakeLlm(("мусор", None)), candidates, top_n=18)
    assert [c.idx for c in ranked] == list(range(18))

    ranked, _ = _run(FakeLlm(('{"ranked": [{"id": 29, "score": 0.8}]}', None)), candidates, top_n=18)
    assert [c.idx for c in ranked] == [29] + list(range(17))


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", '"nan"', "1e400", "-1e400"])
def test_rerank_non_finite_score_becomes_zero(bad):
    """json.loads принимает NaN/Infinity/1e400; в ответ и JSONB такая оценка попадать не должна."""
    candidates = [_cand(10, 0.03), _cand(11, 0.02)]
    llm = FakeLlm(('{"ranked": [{"id": 11, "score": %s}, {"id": 10, "score": 0.5}]}' % bad, 7))

    ranked, _ = _run(llm, candidates)

    assert [c.idx for c in ranked] == [11, 10]
    assert [c.rerank_score for c in ranked] == [0.0, 0.5]
