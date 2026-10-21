from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tomllib


def load(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        local = tomllib.load(stream)
    parent = local.pop("extends", None)
    result = load(path.parent / parent) if parent else {}
    return merge(result, local)


def merge(base, patch):
    result = copy.deepcopy(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate(config, production=False):
    if config["experiment"]["method"] not in {"b0", "b1", "b2"}:
        raise ValueError("Unknown method")
    if config["training"]["group_size"] < 2:
        raise ValueError("GRPO group_size must be >=2")
    if config["training"]["mini_batch_size"] > config["training"]["query_batch_size"]:
        raise ValueError("mini_batch_size exceeds query_batch_size")
    if config["training"]["query_batch_size"] % config["training"]["mini_batch_size"]:
        raise ValueError("query_batch_size must be divisible by mini_batch_size")
    if config["reward"]["scale"] != 1.0:
        raise ValueError("This protocol fixes valid rubric rewards to [0,1]")
    if config["reward"]["parser_policy"] != "rr_compatible":
        raise ValueError("Controlled baselines use upstream RR parsing")
    if config["reward"]["judge_failure_policy"] not in {"abort", "upstream_drop"}:
        raise ValueError("Unknown Judge failure policy")
    if config["ato"]["thresholds"] != [0.3, 0.6]:
        raise ValueError("This protocol fixes hint thresholds to [0.3,0.6]")
    if not 0 <= config["ato"]["grpo_weight"] <= 1:
        raise ValueError("grpo_weight outside [0,1]")
    if config["judge"]["max_attempts"] < 1 or config["judge"]["workers"] < 1:
        raise ValueError("Judge retry/worker counts must be positive")
    if production:
        for field in ("revision", "sft_sha256"):
            if config["model"][field].startswith("UNSET"):
                raise ValueError(f"Lock model.{field} before a GPU run")
        if config["data"]["manifest_sha256"].startswith("UNSET"):
            raise ValueError("Lock data.manifest_sha256 before a GPU run")
        if config["reward"]["backend"] != "openai_compatible":
            raise ValueError("GPU runs require the real judge backend")


def differences(left, right, prefix=""):
    output = []
    for key in sorted(set(left) | set(right)):
        name = f"{prefix}.{key}" if prefix else key
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            output.extend(differences(a, b, name))
        elif a != b:
            output.append((name, a, b))
    return output


def assert_paired(left, right):
    allowed = {"experiment.method", "experiment.name", "runtime.output_dir"}
    unexpected = [d for d in differences(left, right) if d[0] not in allowed]
    if unexpected:
        raise ValueError(f"Uncontrolled paired configuration differences: {unexpected}")
