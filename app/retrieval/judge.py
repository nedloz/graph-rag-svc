"""Выбор стратегии поиска. GraphRAG отключён (спецификация 2026-10-27-notebook-pipeline-design),
поэтому единственный маршрут — векторный. Модуль удаляется вместе со старым поиском при
переходе на пайплайн из ноутбука.
"""

from typing import Literal

Route = Literal["vector"]


def route(question: str, profile: dict) -> Route:
    return "vector"
