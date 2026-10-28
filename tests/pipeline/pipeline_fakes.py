"""Подделки для тестов пайплайна: LLM, сессия БД, фабрика сессий, строки чанков.

Модули app/pipeline получают зависимости параметрами, поэтому ни БД, ни сеть, ни LLM тестам
не нужны.
"""

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql


class FakeLlm:
    """Отвечает заготовленными (текст, токены) по очереди и запоминает вызовы."""

    def __init__(self, *replies: tuple[str, int | None]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, int | None]:
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if not self.replies:
            raise AssertionError("LLM вызван больше раз, чем ожидалось")
        return self.replies.pop(0)


class FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = list(rows)

    def all(self) -> list:
        return list(self._rows)

    def one(self):
        return self._rows[0]


class FakeSession:
    """execute() по очереди отдаёт заготовленные наборы строк и запоминает запросы.

    `executed` — все вызовы по порядку (запрос, параметры). Настройки через set_config строк не
    потребляют и в `statements` не попадают: там только запросы с данными.
    """

    def __init__(self, *results: list) -> None:
        self.results = list(results)
        self.statements: list = []
        self.executed: list[tuple] = []

    async def execute(self, stmt, params=None):
        self.executed.append((stmt, params))
        if "set_config" in str(stmt):
            return FakeResult([])
        self.statements.append(stmt)
        return FakeResult(self.results.pop(0) if self.results else [])


def session_factory(session: FakeSession):
    @asynccontextmanager
    async def factory():
        yield session

    return factory


def bound_values(stmt) -> list:
    """Значения параметров запроса — чтобы проверить фильтры без живой БД."""
    return list(stmt.compile(dialect=postgresql.dialect()).params.values())


def chunk_row(doc: uuid.UUID, index: int, text: str, title: str = "Документ", category: str = "Общие документы",
              section: str | None = "Раздел 1", chunk_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id or uuid.uuid4(),
        document_id=doc,
        chunk_index=index,
        section_path=section,
        text=text,
        title=title,
        category=category,
    )


def sql_of(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))
