"""Ограничение длины входа: вопрос и история обрезаются до того, как попадут в пайплайн.

Вопрос и история уходят в LLM трижды (переписывание, реранк, ответ) и в BM25-токенизацию на
цикле событий; ни chat-svc, ни контракт /generate длину не ограничивают. Поэтому длинный ввод
обрезается здесь, а не отклоняется: ответ 422 сломал бы контракт для бота соседней команды.
Пределы и порядок отбора истории — как в прежней сборке промпта (до пайплайна ноутбука).
"""

from app.api.schemas import HistoryMessage
from app.core.config import settings


def _cut(value: str, limit: int) -> str:
    return value.strip()[:limit].strip()


def clamp_request(
    question: str,
    history: list[HistoryMessage],
    *,
    max_question_chars: int | None = None,
    max_history_message_chars: int | None = None,
    max_history_chars: int | None = None,
) -> tuple[str, list[HistoryMessage]]:
    """(вопрос, история) в пределах лимитов; по умолчанию лимиты из настроек.

    Каждая реплика обрезается до `max_history_message_chars`, пустые отбрасываются. Затем
    реплики берутся от новых к старым, пока суммарная длина укладывается в `max_history_chars`;
    первая не поместившаяся останавливает отбор. Порядок оставшихся — исходный.
    """
    question_limit = settings.max_question_chars if max_question_chars is None else max_question_chars
    message_limit = settings.max_history_message_chars if max_history_message_chars is None else max_history_message_chars
    history_limit = settings.max_history_chars if max_history_chars is None else max_history_chars

    kept: list[HistoryMessage] = []
    total = 0
    for message in reversed(history or []):
        content = _cut(message.content or "", message_limit)
        if not content:
            continue
        if total + len(content) > history_limit:
            break
        kept.append(message if content == message.content else HistoryMessage(role=message.role, content=content))
        total += len(content)
    kept.reverse()

    return _cut(question or "", question_limit), kept
