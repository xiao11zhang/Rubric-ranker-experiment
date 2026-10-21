"""Thin verl adapter around the pinned, unchanged RubricRanker reward."""
from __future__ import annotations

import os
import json
import logging

from setwise.algorithms import parse_ids
from setwise.config import load
from setwise.judge import JudgeError, OpenAIJudge
from setwise.upstream import rr_module


def plain(value):
    """Normalize Arrow/NumPy containers without changing score values."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def score_selection(text, extra, config, judge=None):
    extra = plain(extra or {})
    initial = extra.get("initial_list") or []
    # Diagnostics and ATD mask only; reward parsing belongs to original RR.
    parsed = parse_ids(text, len(initial), "rr_compatible")
    result = {"score": -1.0, "format_valid": int(parsed["strict_valid"]),
              "incorrect_format": int(not parsed["strict_valid"]), "num_selected": len(parsed["ids"]),
              "duplicate_count": parsed["duplicate_count"], "out_of_range_count": parsed["out_of_range_count"],
              "judge_calls": 0, "cache_hit": 0, "judge_parse_failures": 0,
              "judge_incomplete": 0, "reward_backend": config["reward"]["backend"]}
    if config["reward"]["backend"] == "mock":
        if not parsed["parseable"]:
            return result
        gold = set(extra.get("debug_gold_ids", []))
        if not gold:
            raise ValueError("Mock reward requires debug_gold_ids")
        chosen = set(parsed["ids"])
        result["score"] = 2 * len(gold & chosen) / (len(gold) + len(chosen))
        return result
    if config["reward"]["backend"] != "openai_compatible":
        raise ValueError("Unknown reward backend")
    if config["reward"]["parser_policy"] != "rr_compatible":
        raise ValueError("Baseline adapter requires upstream RR parsing")
    judge = judge or OpenAIJudge(config["judge"])
    observed = {"failures": 0, "success": False, "scores": []}
    rr = rr_module(lambda _: None, config["judge"]["max_attempts"])
    active = rr._flatten_rubrics((extra.get("rubrics") or {}).get("set_level_rubrics"), rr.SET_LEVEL_RUBRIC_NAMES)

    def call_gpt(prompt):
        value = judge.complete(prompt)
        scores = rr._parse_rubric_scores(value, len(active))
        if scores is not None:
            observed["success"] = True
            observed["scores"] = scores
            if hasattr(judge, "remember"):
                try:
                    judge.remember(prompt, value)
                except Exception:
                    logging.getLogger(__name__).warning("Judge cache write failed; preserving valid RR response")
        else:
            observed["failures"] += 1
        return value

    rr._call_gpt = call_gpt
    result["score"] = rr.compute_score(solution_str=text, extra_info=extra)
    incomplete = bool(parsed["ids"] and active and not observed["success"])
    result["judge_incomplete"] = int(incomplete)
    result["judge_parse_failures"] = observed["failures"]
    result["rubric_scores"] = json.dumps({f"{name}/{i}": value / 10
        for i, ((name, _, _), value) in enumerate(zip(active, observed["scores"]))}, sort_keys=True)
    result.update(getattr(judge, "stats", {}))
    if incomplete and config["reward"]["judge_failure_policy"] == "abort":
        # Explicit controlled change: official RR drops failed set-level rubrics.
        raise JudgeError("RR judge exhausted retries; batch aborted by configured policy")
    return result


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    if data_source != "set_selection":
        raise ValueError("Unexpected data source")
    return score_selection(solution_str, extra_info, load(os.environ["SETWISE_CONFIG"]))
