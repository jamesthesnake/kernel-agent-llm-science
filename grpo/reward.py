from __future__ import annotations
import json, math
from dataclasses import dataclass
from typing import Any, Dict
from grpo.markers import format_reward, MarkerParse

@dataclass
class RewardTerms:
    format_ok: int
    accuracy_ok: int
    performance: float
    total: float
    notes: str | None

def compute_reward(
    generated_text: str,
    results_json: Dict[str, Any] | None,
    *,
    speedup_floor: float = 0.0
) -> tuple[RewardTerms, MarkerParse]:
    fmt, parsed = format_reward(generated_text)
    acc = 0
    perf = 0.0
    notes = None
    if results_json is None:
        return RewardTerms(fmt, 0, 0.0, 0.0, "no results"), parsed

    # Use 'best' entry if present
    best = results_json.get("best")
    if best and best.get("passed"):
        acc = 1
        spd = best.get("speedup_vs_baseline")
        if spd is None or not math.isfinite(spd):
            perf = 0.0
            notes = "no speedup recorded"
        else:
            perf = max(speedup_floor, float(spd))
    else:
        acc = 0
        perf = 0.0
        notes = "no passing config"

    total = float(fmt * acc * perf)
    return RewardTerms(fmt, acc, perf, total, notes), parsed

def parse_plan_json_from_answer(answer_block: str) -> dict | None:
    """Answer MUST be a single raw JSON object. Attempt strict load."""
    try:
        obj = json.loads(answer_block)
        assert isinstance(obj, dict)
        return obj
    except Exception:
        return None
