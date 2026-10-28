"""clamp_request: длина вопроса и истории ограничивается обрезкой, а не отказом."""

from app.api.schemas import HistoryMessage
from app.core.config import Settings
from app.pipeline.limits import clamp_request


def _msg(role: str, content: str) -> HistoryMessage:
    return HistoryMessage(role=role, content=content)


def test_long_question_is_cut_to_limit():
    question, _ = clamp_request("в" * 10000, [])
    assert question == "в" * 4000


def test_question_at_limit_passes_unchanged():
    original = "вопрос " * 571 + "ещё"  # 4000 символов
    assert len(original) == 4000
    question, history = clamp_request(original, [])
    assert question == original and history == []


def test_history_messages_cut_then_oldest_dropped_by_total_budget():
    history = [_msg("user", "a" * 10000), _msg("assistant", "b" * 10000), _msg("user", "c" * 10000)]

    _, clamped = clamp_request("q", history, max_history_chars=8000)

    # Каждая реплика ≤ 4000, в бюджет 8000 помещаются две самые новые — в исходном порядке.
    assert [(m.role, m.content) for m in clamped] == [("assistant", "b" * 4000), ("user", "c" * 4000)]


def test_default_budget_keeps_three_cut_messages():
    history = [_msg("user", "a" * 10000), _msg("assistant", "b" * 10000), _msg("user", "c" * 10000)]
    _, clamped = clamp_request("q", history)
    assert [m.content for m in clamped] == ["a" * 4000, "b" * 4000, "c" * 4000]


def test_default_budget_drops_oldest_beyond_24000():
    history = [_msg("user", str(i) * 4000) for i in range(7)]  # 7 × 4000 = 28000
    _, clamped = clamp_request("q", history)
    assert [m.content[0] for m in clamped] == ["1", "2", "3", "4", "5", "6"]


def test_budget_stops_at_first_message_that_does_not_fit():
    # Как в прежнем _prepare_history: старые реплики за первой не вошедшей не подбираются.
    history = [_msg("user", "x" * 10), _msg("assistant", "y" * 3000), _msg("user", "z" * 1000)]
    _, clamped = clamp_request("q", history, max_history_chars=2000)
    assert [m.content for m in clamped] == ["z" * 1000]


def test_empty_and_whitespace_messages_are_dropped():
    history = [_msg("user", "  \n\t "), _msg("assistant", "ответ"), _msg("user", "")]
    _, clamped = clamp_request("q", history)
    assert [(m.role, m.content) for m in clamped] == [("assistant", "ответ")]


def test_short_input_is_untouched():
    history = [_msg("user", "Сколько пересдач?"), _msg("assistant", "Две.")]
    question, clamped = clamp_request("А по физкультуре?", history)
    assert question == "А по физкультуре?"
    assert [(m.role, m.content) for m in clamped] == [(m.role, m.content) for m in history]


def test_limit_settings_defaults(monkeypatch):
    for name in ("MAX_QUESTION_CHARS", "MAX_HISTORY_MESSAGE_CHARS", "MAX_HISTORY_CHARS"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert (s.max_question_chars, s.max_history_message_chars, s.max_history_chars) == (4000, 4000, 24000)
