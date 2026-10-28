"""LLM-реранк кандидатов одним вызовом (ячейки 19 и 23)."""

import dataclasses
import json
import math
import re

from app.pipeline.index import Candidate
from app.pipeline.llm import LlmChat

RERANK_CANDIDATES = 28
RERANK_TEXT_CHARS = 1100
RERANK_TOP_N = 18
RERANK_TEMPERATURE = 0.0
RERANK_MAX_TOKENS = 750

RERANK_SYSTEM = """
Ты переранжируешь фрагменты документов НИУ ВШЭ
по релевантности вопросу пользователя.

Не отвечай на вопрос.

Оцени, насколько каждый фрагмент непосредственно
помогает получить правильный ответ.

Учитывай:
- точную тему и процедуру;
- общий или специальный характер правила;
- сроки;
- условия;
- исключения;
- отрицания;
- категории студентов.

Если пользователь спрашивает об общей категории,
не ставь выше специальный частный случай
только из-за совпадения отдельных слов.

Верни только JSON:
{
  "ranked": [
    {"id": 123, "score": 0.95}
  ]
}
""".strip()


def parse_json_response(text: str) -> dict:
    """JSON-объект из ответа модели: снимает обёртку ```json и текст вокруг объекта.

    Всё, что не разбирается в объект, — `ValueError` (json.JSONDecodeError — его подкласс).
    """
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I)
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except Exception:
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if not (start_obj >= 0 and end_obj > start_obj):
            raise ValueError(f"JSON parse error: {text[:500]}")
        data = json.loads(text[start_obj:end_obj + 1])

    if not isinstance(data, dict):
        raise ValueError(f"JSON parse error: ожидался объект, получено {type(data).__name__}")
    return data


async def rerank(
    llm: LlmChat,
    question: str,
    candidates: list[Candidate],
    top_n: int = RERANK_TOP_N,
) -> tuple[list[Candidate], int | None]:
    """Порядок от LLM; чужие и повторные id пропускаются, остальные кандидаты идут следом по RRF.

    Любой сбой разбора ответа (не JSON, нет `ranked`, id не число) — первые `top_n` кандидатов
    в порядке RRF с `rerank_score = 0.0`, как в ноутбуке. Ошибка самого вызова LLM не
    глушится: без модели не будет и ответа, вызывающий должен увидеть ошибку.
    """
    compact = [
        {
            "id": item.idx,
            "title": item.title,
            "category": item.category,
            "section": item.section,
            "text": item.text[:RERANK_TEXT_CHARS],
        }
        for item in candidates[:RERANK_CANDIDATES]
    ]
    messages = [
        {"role": "system", "content": RERANK_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Вопрос:\n{question}\n\n"
                "Фрагменты:\n"
                + json.dumps(compact, ensure_ascii=False)
            ),
        },
    ]
    raw, tokens = await llm.chat(messages, temperature=RERANK_TEMPERATURE, max_tokens=RERANK_MAX_TOKENS)

    try:
        data = parse_json_response(raw)
        candidate_map = {item.idx: item for item in candidates}
        result = []
        seen = set()

        for row in data.get("ranked", []):
            idx = int(row["id"])
            if idx not in candidate_map or idx in seen:
                continue
            seen.add(idx)
            score = float(row.get("score", 0))
            if not math.isfinite(score):
                # NaN/Infinity/1e400 json.loads пропускает, но в JSON-ответ и JSONB они не лягут.
                score = 0.0
            result.append(dataclasses.replace(candidate_map[idx], rerank_score=score))

        for item in candidates:
            if item.idx in seen:
                continue
            result.append(dataclasses.replace(item, rerank_score=0.0))

        return result[:top_n], tokens

    except Exception:
        return [dataclasses.replace(item, rerank_score=0.0) for item in candidates[:top_n]], tokens
