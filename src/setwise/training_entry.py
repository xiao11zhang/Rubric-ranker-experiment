"""Small documented compatibility patch; no private backend copy is maintained."""
import os
import random
import runpy

from setwise.config import load


def assign_normalized_hint_types(scores, config):
    thresholds = list(config.get("hint_thresholds", []))
    if thresholds != [0.3, 0.6]:
        raise ValueError("Expected normalized RR hint thresholds")
    buckets = list(config.get("hint_bucket_types", []))
    if buckets != ["demonstration", "diff", "rubric"]:
        raise ValueError("Unexpected hint buckets")
    # Invalid reward -1 is excluded from ATD by incorrect_format_mask downstream.
    return [buckets[0] if s <= 0.3 else buckets[1] if s < 0.6 else buckets[2] for s in scores]


def main():
    config = load(os.environ["SETWISE_CONFIG"])
    import numpy as np
    import torch
    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


if __name__ == "__main__":
    main()
