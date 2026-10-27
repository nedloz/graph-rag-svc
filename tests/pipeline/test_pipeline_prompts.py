"""Финальный промпт (ячейки 27, 36) и переписывание вопроса с историей (ячейка 35)."""

import asyncio

from app.api.schemas import HistoryMessage
from app.pipeline.prompts import FINAL_SYSTEM, build_messages, language_rule
from app.pipeline.rewrite import REWRITE_SYSTEM, contextualize_question
from pipeline_fakes import FakeLlm

RU_CONTEXT = "Источник: Положение\nКатегория: Общие документы\nРаздел: 1\n\nСправку выдаёт учебный офис."
EN_RULE = "Отвечай полностью на английском языке."
RU_RULE = "Отвечай полностью на русском языке."


def _history(n: int) -> list[HistoryMessage]:
    return [
        HistoryMessage(role="user" if i % 2 == 0 else "assistant", content=f"реплика {i}")
        for i in range(n)
    ]


def test_final_system_states_language_rule_once():
    assert FINAL_SYSTEM.startswith("Ты — AI-ассистент студентов НИУ ВШЭ.")
    assert FINAL_SYSTEM.count("ЯЗЫК ОТВЕТА:") == 1
    assert "Отвечай на языке вопроса пользователя" not in FINAL_SYSTEM
    assert "Пиши естественно, без мата и без упоминания устройства системы." in FINAL_SYSTEM
    assert "по-русски" not in FINAL_SYSTEM


def test_language_rule_counts_latin_vs_cyrillic():
    assert language_rule("How do I get a certificate?") == EN_RULE
    assert language_rule("Как получить справку?") == RU_RULE
    assert language_rule("СОП") == RU_RULE
    assert language_rule("SET") == EN_RULE
    assert language_rule("123") == RU_RULE  # при равенстве — русский, как в ноутбуке


def test_build_messages_english_question_with_russian_context():
    system, user = build_messages("How do I get a certificate?", [], RU_CONTEXT, [])

    assert system["role"] == "system" and user["role"] == "user"
    assert "на английском" in system["content"]
    assert system["content"].startswith(FINAL_SYSTEM)
    assert system["content"].endswith(
        "\nОБЯЗАТЕЛЬНО: " + EN_RULE + " Язык определён по текущему исходному сообщению пользователя. "
        "Историю диалога используй для понимания смысла, но не для выбора языка ответа."
    )
    assert RU_RULE not in system["content"]


def test_build_messages_language_from_original_question_not_history():
    history = [HistoryMessage(role="user", content="Как пересдать экзамен?"),
               HistoryMessage(role="assistant", content="Пересдача проводится дважды.")]
    system, _ = build_messages("And what about the third attempt?", history, RU_CONTEXT, [])
    assert EN_RULE in system["content"]

    system, _ = build_messages("А третья попытка?", [HistoryMessage(role="user", content="How to retake?")], "", [])
    assert RU_RULE in system["content"]


def test_build_messages_user_prompt_layout():
    links = [("Старый сайт", "http://miem.hse.ru/old"), ("Расписание", "https://ruz.hse.ru")]
    _, user = build_messages("Где расписание?", _history(2), RU_CONTEXT, links)

    assert user["content"] == (
        "История текущего диалога:\n"
        "Пользователь: реплика 0\nАссистент: реплика 1\n\n"
        "Материалы:\n" + RU_CONTEXT + "\n\n"
        "Ссылки:\n- Старый сайт: http://miem.hse.ru/old\n- Расписание: https://ruz.hse.ru\n\n"
        "Вопрос пользователя:\nГде расписание?\n\n"
        "Ответ:"
    )


def test_build_messages_empty_context_history_and_links():
    _, user = build_messages("Где расписание?", [], "", [])

    assert "История текущего диалога:\nнет\n\n" in user["content"]
    assert "Материалы:\nрелевантные материалы не найдены\n\n" in user["content"]
    assert "Ссылки:\nСсылки не найдены\n\n" in user["content"]


def test_build_messages_keeps_last_six_history_messages():
    _, user = build_messages("Вопрос", _history(8), "", [])
    assert "реплика 1\n" not in user["content"]
    assert "Пользователь: реплика 2\n" in user["content"]
    assert "Ассистент: реплика 7\n" in user["content"]


def test_contextualize_without_history_skips_llm():
    llm = FakeLlm()
    assert asyncio.run(contextualize_question(llm, "А сроки?", [])) == "А сроки?"
    assert llm.calls == []


def test_contextualize_with_history_uses_last_ten_messages():
    llm = FakeLlm(("  Какие сроки пересдачи экзамена?\n", 40))

    result = asyncio.run(contextualize_question(llm, "А сроки?", _history(12)))

    assert result == "Какие сроки пересдачи экзамена?"
    call = llm.calls[0]
    assert call["temperature"] == 0.0 and call["max_tokens"] == 150
    system, user = call["messages"]
    assert system == {"role": "system", "content": REWRITE_SYSTEM}
    assert user["content"].startswith("История диалога:\nПользователь: реплика 2\nАссистент: реплика 3\n")
    assert user["content"].endswith("Ассистент: реплика 11\n\nТекущий вопрос: А сроки?\n\nПолный вопрос:")
    assert "реплика 1\n" not in user["content"]


def test_contextualize_empty_rewrite_returns_original():
    llm = FakeLlm(("   ", None))
    assert asyncio.run(contextualize_question(llm, "А сроки?", _history(2))) == "А сроки?"
