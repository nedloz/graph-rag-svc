"""Prompt-injection heuristics over retrieved context — ported verbatim from
services/llm-svc/app/main.py (PROMPT_INJECTION_PATTERNS, _find_prompt_injection_signals).
"""

import re

PROMPT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override_instructions", re.compile(r"(?i)\b(ignore|disregard|forget|override|bypass)\b.{0,80}\b(previous|prior|system|developer|above|all)\b")),
    ("role_spoofing", re.compile(r"(?is)<\s*/?\s*(system|assistant|developer|tool)\s*>|\b(role\s*:\s*(system|assistant|developer|tool))\b")),
    ("secret_request", re.compile(r"(?i)\b(api[ _-]?key|token|secret|password|credential|env(?:ironment)? variable|system prompt|developer prompt)\b")),
    ("jailbreak", re.compile(r"(?i)\b(DAN|do anything now|jailbreak|prompt injection|developer mode)\b")),
    ("tool_abuse", re.compile(r"(?i)\b(call tool|function call|execute code|run command|shell|bash|powershell|sql query)\b")),
    ("data_exfiltration", re.compile(r"(?i)\b(exfiltrate|leak|dump|reveal|print)\b.{0,80}\b(prompt|instruction|secret|token|key|credentials?)\b")),
]


def find_prompt_injection_signals(text: str) -> list[str]:
    signals: list[str] = []
    if not text:
        return signals
    for name, pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            signals.append(name)
    return signals
