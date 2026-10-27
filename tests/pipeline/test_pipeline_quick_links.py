"""Быстрые ответы (ячейки 32–33) и ссылки (ячейка 15) на подделанной сессии."""

import asyncio
from types import SimpleNamespace

import pytest

from app.pipeline.links import find_links
from app.pipeline.quick_answers import QuickAnswer, find_quick_answer
from pipeline_fakes import FakeSession, bound_values, sql_of

VECTOR = [0.1] * 4


def _variant(text: str, cosine: float, answer: str = "Ответ", question: str = "Вопрос") -> SimpleNamespace:
    # SQL отдаёт варианты по убыванию косинуса — так же их отдаёт и подделка
    return SimpleNamespace(text=text, cosine=cosine, answer=answer, question=question)


VARIANTS = [
    _variant("как получить справку об обучении", 0.95, "Закажите справку в ЛК.", "Справка об обучении"),
    _variant("где находится столовая", 0.50, "На первом этаже.", "Столовая"),
    _variant("когда приходит стипендия", 0.40, "25 числа.", "Стипендия"),
]


def _find(session, question, **kw):
    return asyncio.run(find_quick_answer(session, question, VECTOR, "e5-base", **kw))


def test_quick_answer_when_both_thresholds_pass():
    session = FakeSession(VARIANTS)

    found = _find(session, "Как получить справку об обучении?")

    assert found == QuickAnswer(
        answer="Закажите справку в ЛК.", question="Справка об обучении",
        rrf_score=pytest.approx(1.5 / 61 + 1.0 / 61),
    )
    values = bound_values(session.statements[0])
    assert "e5-base" in values  # только векторы той же модели
    assert "is_active" in sql_of(session.statements[0])  # только активные быстрые ответы


def test_quick_answer_none_below_cosine_threshold():
    rows = [_variant(VARIANTS[0].text, 0.82)] + VARIANTS[1:]
    assert _find(FakeSession(rows), "Как получить справку об обучении?") is None


def test_quick_answer_none_below_rrf_threshold():
    # Лучший по косинусу вариант без единого общего слова: RRF = 1.5/61 < 0.04
    assert _find(FakeSession(VARIANTS), "Оформление документа для военкомата") is None


def test_quick_answer_thresholds_are_parameters():
    assert _find(FakeSession(VARIANTS), "Как получить справку об обучении?", cosine_threshold=0.96) is None
    found = _find(FakeSession(VARIANTS), "Оформление документа для военкомата", rrf_threshold=0.02)
    assert found is not None and found.answer == "Закажите справку в ЛК."


def test_quick_answer_normalizes_question_like_import():
    # Пунктуация и регистр снимаются тем же правилом, что и у сохранённых вариантов
    found = _find(FakeSession(VARIANTS), "  КАК получить справку, об обучении?!  ")
    assert found is not None


def test_quick_answer_empty_inputs():
    assert _find(FakeSession([]), "Как получить справку?") is None
    empty_question = FakeSession(VARIANTS)
    assert _find(empty_question, " ?! ") is None
    assert empty_question.statements == []  # пустой запрос в базу не ходит


def _link(description: str, url: str, cosine: float) -> SimpleNamespace:
    return SimpleNamespace(description=description, url=url, cosine=cosine)


def test_find_links_keeps_http_and_cuts_by_threshold():
    session = FakeSession([
        _link("Старый сайт факультета", "http://miem.hse.ru/old", 0.91),
        _link("Расписание", "https://ruz.hse.ru", 0.76),
        _link("Ровно порог", "https://edu.hse.ru", 0.75),  # в ноутбуке строго > 0.75
    ])

    links = asyncio.run(find_links(session, VECTOR, "e5-base"))

    assert links == [("Старый сайт факультета", "http://miem.hse.ru/old"), ("Расписание", "https://ruz.hse.ru")]
    values = bound_values(session.statements[0])
    assert "e5-base" in values and 3 in values
    assert "is_active" in sql_of(session.statements[0])


def test_find_links_parameters():
    session = FakeSession([_link("Расписание", "https://ruz.hse.ru", 0.76)])
    assert asyncio.run(find_links(session, VECTOR, "e5-base", top_k=5, threshold=0.8)) == []
    assert 5 in bound_values(session.statements[0])
