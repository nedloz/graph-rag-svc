"""Переписывание вопроса с учётом истории диалога (ячейка 35, `contextualize_question`).

Переписанный вопрос идёт только в поиск и реранк; в промпт ответа и в выбор языка — исходный.
"""

from app.api.schemas import HistoryMessage
from app.pipeline.llm import LlmChat
from app.pipeline.prompts import history_lines

REWRITE_HISTORY_MESSAGES = 10
REWRITE_TEMPERATURE = 0.0
REWRITE_MAX_TOKENS = 150

REWRITE_SYSTEM = """
Ты — помощник, который переписывает текущий вопрос пользователя с учетом истории диалога.
Сделай так, чтобы получился самостоятельный, полный вопрос, понятный без контекста истории (раскрой местоимения вроде «это», «тут», «о чём речь»).
Если вопрос уже самостоятельный, оставь его без изменений.
Верни только текст переписанного вопроса, без лишних пояснений.
""".strip()


async def contextualize_question(llm: LlmChat, question: str, history: list[HistoryMessage]) -> str:
    """Самостоятельная формулировка вопроса по последним 10 репликам; без истории — исходный вопрос."""
    if not history:
        return question

    history_text = "\n".join(history_lines(history, REWRITE_HISTORY_MESSAGES))
    prompt = f"История диалога:\n{history_text}\n\nТекущий вопрос: {question}\n\nПолный вопрос:"

    rewritten, _ = await llm.chat(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=REWRITE_TEMPERATURE,
        max_tokens=REWRITE_MAX_TOKENS,
    )
    rewritten = (rewritten or "").strip()
    return rewritten if rewritten else question
