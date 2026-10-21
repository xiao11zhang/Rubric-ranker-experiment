import copy
import json
import math
from pathlib import Path
import tempfile
import tomllib
import unittest

from setwise.algorithms import atd_advantage, clipped_loss, combine_advantages, grpo_advantages, hint_type, parse_ids
from setwise.backend import overrides
from setwise.cli import dump_toml
from setwise.config import assert_paired, load, validate
from setwise.data import student_prompt, validate_row
from setwise.judge import JudgeError
from setwise.rewards import score_selection
from setwise.training_entry import assign_normalized_hint_types

ROOT = Path(__file__).resolve().parents[1]


def example():
    return {"id": "q1", "query": "Which city?", "split": "train", "docs": [
        {"position": 1, "document_id": "wiki:1", "content": "Evidence", "Source Authority": 0.8},
        {"position": 2, "document_id": "wiki:2", "content": "Other", "Source Authority": 0.4}],
        "rubrics": {"set_level_rubrics": {"Relevance": [{"description": "Support city identification", "weight": 2}]},
                    "doc_level_rubrics": {"Source Authority": [{"description": "Reliable source", "weight": 1}]}},
        "silver_ids": [1], "answer": "SECRET_ANSWER"}


def extra(row):
    return {"query_id": row["id"], "query": row["query"], "split": row["split"],
            "initial_list": row["docs"], "rubrics": row["rubrics"]}


class ProtocolTests(unittest.TestCase):
    def test_paired_configs(self):
        assert_paired(load(ROOT / "configs/b1.toml"), load(ROOT / "configs/b2.toml"))

    def test_uncontrolled_difference_rejected(self):
        a = load(ROOT / "configs/b1.toml")
        b = copy.deepcopy(a)
        b["training"]["learning_rate"] *= 2
        with self.assertRaises(ValueError):
            assert_paired(a, b)

    def test_unlocked_run_rejected(self):
        with self.assertRaises(ValueError):
            validate(load(ROOT / "configs/b1.toml"), production=True)

    def test_snapshot_roundtrip(self):
        config = load(ROOT / "configs/b2.toml")
        self.assertEqual(config, tomllib.loads(dump_toml(config)))

    def test_no_privileged_information(self):
        row = example()
        row["split"] = "debug"
        prompt = json.dumps(student_prompt(row))
        self.assertNotIn("SECRET_ANSWER", prompt)
        self.assertNotIn("Support city identification", prompt)

    def test_candidate_mapping(self):
        row = example()
        row["docs"][0]["position"] = 2
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_prescores_required(self):
        row = example()
        del row["docs"][0]["Source Authority"]
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_strict_and_compatible_parser(self):
        text = "Explanation [1] [1] [999]"
        self.assertFalse(parse_ids(text, 2)["valid"])
        parsed = parse_ids(text, 2, "rr_compatible")
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["ids"], [1])
        self.assertFalse(parsed["strict_valid"])

    def test_hint_boundaries(self):
        self.assertEqual(hint_type(0.3, 0.8), "demonstration")
        self.assertEqual(hint_type(0.30001, 0.8), "diff")
        self.assertEqual(hint_type(0.6), "rubric")
        self.assertEqual(hint_type(0.1, 0.1), "rubric")
        self.assertEqual(assign_normalized_hint_types([0.3, 0.30001, 0.6],
            {"hint_thresholds": [0.3, 0.6], "hint_bucket_types": ["demonstration", "diff", "rubric"]}),
            ["demonstration", "diff", "rubric"])

    def test_grpo_sample_std_and_zero_variance(self):
        a = grpo_advantages([0, 1])
        self.assertAlmostEqual(a[1], 0.5 / (math.sqrt(0.5) + 1e-6))
        self.assertEqual(grpo_advantages([1, 1]), [0, 0])

    def test_atd_alignment_and_masks(self):
        self.assertEqual(atd_advantage([-2, -3], [-1, -1], [1, 0]), [1, 0])
        self.assertEqual(atd_advantage([-2], [-1], [1], False), [0])
        with self.assertRaises(ValueError):
            atd_advantage([-1], [], [1])

    def test_b2_reduces_to_b1_at_grpo_weight_one(self):
        advantages = [1, -1]
        old, new, mask = [-2, -3], [-1.9, -3.2], [1, 1]
        combined = combine_advantages(advantages, [20, -30], 1.0)
        self.assertEqual(clipped_loss(old, new, advantages, mask), clipped_loss(old, new, combined, mask))
        def derivative(adv):
            h = 1e-5
            return (clipped_loss(old, [new[0] + h, new[1]], adv, mask) -
                    clipped_loss(old, [new[0] - h, new[1]], adv, mask)) / (2 * h)
        self.assertAlmostEqual(derivative(advantages), derivative(combined))

    def test_reward_aggregation(self):
        class Judge:
            def complete(self, prompt):
                return "9"
        config = load(ROOT / "configs/b1.toml")
        result = score_selection("[1]", extra(example()), config, Judge())
        self.assertAlmostEqual(result["score"], (2 * 0.9 + 0.8) / 3)
        self.assertEqual(result["format_valid"], 1)

    def test_judge_failure_propagates(self):
        class Judge:
            def complete(self, prompt):
                raise JudgeError("offline")
        with self.assertRaises(JudgeError):
            score_selection("[1]", extra(example()), load(ROOT / "configs/b1.toml"), Judge())

    def test_backend_difference_is_only_ato_path(self):
        def mapping(config):
            return dict(item.split("=", 1) for item in overrides(config, "/opt/ATO"))
        a, b = mapping(load(ROOT / "configs/b1.toml")), mapping(load(ROOT / "configs/b2.toml"))
        for key in set(a) & set(b):
            if key.startswith("trainer.") or key.startswith("+trainer.") or key.endswith("policy_loss.loss_mode"):
                continue
            self.assertEqual(a[key], b[key], key)

    def test_no_api_for_invalid_format(self):
        config = load(ROOT / "configs/b1.toml")
        self.assertEqual(score_selection("No IDs", extra(example()), config)["score"], -1)

    def test_distributed_hint_thresholds(self):
        config = load(ROOT / "configs/b2.toml")
        items = dict(item.split("=", 1) for item in overrides(config, "/opt/ATO"))
        cuts = json.loads(items["actor_rollout_ref.actor.self_distillation.hint_thresholds"])
        def upstream_generic(score):
            return sum(score >= cut for cut in cuts)
        self.assertEqual(upstream_generic(0.3), 0)
        self.assertEqual(upstream_generic(math.nextafter(0.3, math.inf)), 1)
        self.assertEqual(upstream_generic(0.6), 2)


if __name__ == "__main__":
    unittest.main()
