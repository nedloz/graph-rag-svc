"""Prompt templates for the shared ontology-extraction module (app/ontology/extractor.py).

Ported from research/GetTripletsFromChunks.ipynb's `llm_request()` system prompt, with
two variants: extraction over a window of document chunks (offline graph-builder), and
extraction over a single user question (online query pipeline). Both share the same
entity-type list and output contract so the two callers stay interchangeable.
"""

from app.ontology.schema import BASE_ENTITY_TYPES

_ENTITY_TYPES_LIST = ", ".join(BASE_ENTITY_TYPES)

_COMMON_RULES = f"""
Ты — эксперт по анализу документов и нормативных актов НИУ ВШЭ.
Твоя задача: извлечь сущности и их взаимосвязи из текста для построения графа знаний.

ПРАВИЛА:
1. Используй в первую очередь следующие типы сущностей: [{_ENTITY_TYPES_LIST}].
2. Дополнительно извлекай ключевые сущности, специфичные для контекста текста (например, для библиотеки: Библиотека, Издание, Читательский билет, Штраф; для общежития: Комната, Комендант).
3. Связь должна быть коротким глаголом в инфинитиве (например: "сдавать", "подавать", "иметь_право", "аннулировать"). Не "подаёт", а "подавать".
4. Верни строго JSON формата: {{"triplets": [["Субъект", "Связь", "Объект"]]}}.
5. Запрещено писать любой текст, кроме JSON. Запрещено рассуждать вслух.

Пример: "Студент подаёт заявление в учебный офис" → {{"triplets": [["Студент", "подавать", "Заявление"], ["Студент", "подавать", "Учебный офис"]]}}
""".strip()

SYSTEM_PROMPT_BUILD = f"""
Return ONLY JSON. NO THINKING. NO REASONING. NO INTRODUCTIONS. Directly start with {{"triplets": ...}}

{_COMMON_RULES}
""".strip()

SYSTEM_PROMPT_QUERY = f"""
Return ONLY JSON. NO THINKING. NO REASONING. NO INTRODUCTIONS. Directly start with {{"triplets": ...}}

{_COMMON_RULES}

Дополнительно (этот текст — вопрос пользователя, не фрагмент документа):
6. Вопрос может быть коротким и не содержать полного триплета — извлекай то, что есть.
   Если явно назван только субъект/сущность без отношения и объекта, верни триплет вида
   ["<сущность>", "упоминать", "<сущность>"] (одна и та же сущность в качестве субъекта и объекта) —
   это используется только для того, чтобы найти сущность в графе, а не как содержательный факт.
7. Если в тексте нет ни одной сущности из списка типов или явно связанной с университетом — верни {{"triplets": []}}.
""".strip()


def build_messages(text: str, mode: str) -> list[dict[str, str]]:
    if mode == "query":
        system_prompt = SYSTEM_PROMPT_QUERY
        user_prompt = f"Вопрос пользователя: {text}"
    else:
        system_prompt = SYSTEM_PROMPT_BUILD
        user_prompt = f"Текст чанка: {text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
