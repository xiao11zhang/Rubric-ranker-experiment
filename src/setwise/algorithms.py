"""Independent numerical reference for auditing the GPU backend."""
from __future__ import annotations

import math
import re


def hint_type(score, sibling_score=None):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Hint routing requires a valid [0,1] rubric score")
    if score >= 0.6:
        return "rubric"
    if sibling_score is None or sibling_score <= score:
        return "rubric"
    return "demonstration" if score <= 0.3 else "diff"


def grpo_advantages(scores, epsilon=1e-6):
    if len(scores) < 2 or any(not math.isfinite(s) for s in scores):
        raise ValueError("Need >=2 finite group scores")
    mean = sum(scores) / len(scores)
    # Match torch.std() sample standard deviation used by upstream GRPO.
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(scores) - 1))
    return [(s - mean) / (std + epsilon) for s in scores]


def atd_advantage(old_logp, teacher_logp, response_mask, valid_format=True):
    if not len(old_logp) == len(teacher_logp) == len(response_mask):
        raise ValueError("Token alignment mismatch")
    return [(t - s) * m * int(valid_format) for s, t, m in zip(old_logp, teacher_logp, response_mask)]


def combine_advantages(grpo, atd, grpo_weight):
    if len(grpo) != len(atd) or not 0 <= grpo_weight <= 1:
        raise ValueError("Invalid advantage combination")
    return [grpo_weight * a + (1 - grpo_weight) * b for a, b in zip(grpo, atd)]


def clipped_loss(old_logp, new_logp, advantages, mask, clip=0.2):
    if not len(old_logp) == len(new_logp) == len(advantages) == len(mask):
        raise ValueError("Token alignment mismatch")
    denominator = sum(mask)
    if denominator <= 0:
        raise ValueError("Empty response mask")
    losses = []
    for old, new, adv, active in zip(old_logp, new_logp, advantages, mask):
        ratio = math.exp(max(-20, min(20, new - old)))
        losses.append(-min(ratio * adv, max(1 - clip, min(1 + clip, ratio)) * adv) * active)
    return sum(losses) / denominator


def parse_ids(text, count, policy="strict"):
    text = text if isinstance(text, str) else ""
    raw = [int(s) for s in re.findall(r"\[\s*(\d+)\s*\]", text)]
    strict = bool(raw) and bool(re.fullmatch(r"\s*(?:\[\s*\d+\s*\]\s*)+", text)) and len(set(raw)) == len(raw) and all(1 <= x <= count for x in raw)
    cleaned = list(dict.fromkeys(x for x in raw if 1 <= x <= count))
    if policy not in {"strict", "rr_compatible"}:
        raise ValueError("Unknown parser policy")
    return {"ids": cleaned, "strict_valid": strict, "parseable": bool(raw),
            "valid": strict if policy == "strict" else bool(raw),
            "duplicate_count": len(raw) - len(set(raw)),
            "out_of_range_count": sum(not 1 <= x <= count for x in raw)}
