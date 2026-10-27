"""DISABLED: GraphRAG отключён (спецификация 2026-10-27-notebook-pipeline-design). Код не поддерживается и не вызывается.

Shared ontology-extraction module (requirement: ONE implementation used both by the
offline graph-builder, over chunk windows, and the online query pipeline, over the
user's question — never two separate copies of this logic).

JSON-repair parsing ported from research/GetTripletsFromChunks.ipynb's `parse_llm_json`:
local/open models routinely wrap the JSON in stray text or leave brackets unclosed when
truncated by max_new_tokens, so the parser has to tolerate that rather than assume a
clean `json.loads` will work.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from app.ontology.llm_backend import complete
from app.ontology.prompts import build_messages

logger = logging.getLogger("graph-rag-svc.ontology.extractor")

_JSON_START_RE = re.compile(r"\{.*", re.DOTALL)


@dataclass
class Triplet:
    subject: str
    relation: str
    object: str
    source_chunk_ids: list[str] = field(default_factory=list)
    raw_context: str = ""


def _parse_llm_json(raw_text: str) -> list[list[str]]:
    """Best-effort extraction of `{"triplets": [...]}` from a raw LLM response."""
    if not raw_text:
        return []

    match = _JSON_START_RE.search(raw_text)
    if not match:
        return []

    json_str = match.group()
    for stop_token in ("<|im_end|>", "<|endoftext|>"):
        if stop_token in json_str:
            json_str = json_str.split(stop_token)[0].strip()

    # Best-effort repair of truncated output: close any unbalanced brackets.
    open_sq, close_sq = json_str.count("["), json_str.count("]")
    if open_sq > close_sq:
        json_str += "]" * (open_sq - close_sq)
    open_cu, close_cu = json_str.count("{"), json_str.count("}")
    if open_cu > close_cu:
        json_str += "}" * (open_cu - close_cu)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse extraction JSON: %s | text=%r", exc, raw_text[:200])
        return []

    triplets = data.get("triplets", [])
    if not isinstance(triplets, list):
        return []
    return triplets


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


async def extract_triplets(
    text: str,
    *,
    mode: str = "build",
    source_chunk_ids: list[str] | None = None,
) -> list[Triplet]:
    """Extract [subject, relation, object] triplets from `text`.

    `mode="build"` is used offline over a window of document chunks; `mode="query"` is
    used online over the user's question (tolerates partial/entity-only triplets, see
    app/ontology/prompts.py).
    """
    text = _normalize(text)
    if not text:
        return []

    messages = build_messages(text, mode)

    try:
        raw_output = await complete(messages)
    except Exception:
        logger.exception("Ontology extraction LLM call failed (mode=%s)", mode)
        return []

    raw_triplets = _parse_llm_json(raw_output)

    result: list[Triplet] = []
    for item in raw_triplets:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        subject, relation, obj = _normalize(str(item[0])), _normalize(str(item[1])), _normalize(str(item[2]))
        if not subject or not relation or not obj:
            continue
        result.append(
            Triplet(
                subject=subject,
                relation=relation,
                object=obj,
                source_chunk_ids=list(source_chunk_ids or []),
                raw_context=text,
            )
        )
    return result
