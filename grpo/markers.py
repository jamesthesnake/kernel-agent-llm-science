from __future__ import annotations
import re
from dataclasses import dataclass

THINK_START = "<|think_start|>"
THINK_END   = "<|think_end|>"
ANS_START   = "<|answer_start|>"
ANS_END     = "<|answer_end|>"

THINK_RE = re.compile(re.escape(THINK_START) + r"(.*?)" + re.escape(THINK_END), re.S)
ANS_RE   = re.compile(re.escape(ANS_START)   + r"(.*?)" + re.escape(ANS_END),   re.S)

@dataclass
class MarkerParse:
    ok: bool
    think: str | None
    answer: str | None
    reason: str | None = None

def extract_marked(text: str) -> MarkerParse:
    m1 = THINK_RE.search(text)
    m2 = ANS_RE.search(text)
    if not m1 or not m2:
        return MarkerParse(False, None, None, "missing markers")
    think = m1.group(1).strip()
    answer = m2.group(1).strip()
    if not answer:
        return MarkerParse(False, think, None, "empty answer")
    return MarkerParse(True, think, answer, None)

def format_reward(text: str) -> tuple[int, MarkerParse]:
    parsed = extract_marked(text)
    return (1 if parsed.ok else 0, parsed)
