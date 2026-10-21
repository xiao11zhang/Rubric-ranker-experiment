"""Differential checks against the COMPLETE pinned upstream module, offline."""
import ast
import copy
import json
from pathlib import Path
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from setwise.config import load
from setwise.data import student_prompt
from setwise.judge import OpenAIJudge
from setwise.rewards import score_selection
from setwise.upstream import ROOT, RR_FUNCTIONS, load_block_templates, rr_module, verify_sources

PROJECT = Path(__file__).resolve().parents[1]


def full_original(reply):
    """Execute entire original file, stubbing only unavailable SDK/env imports."""
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda stream: load_block_templates(stream.name)
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda **kwargs: None
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    module = types.ModuleType("full_original_rr")
    path = ROOT / "rr/set_selection.py"
    module.__file__ = str(path)
    with patch.dict("sys.modules", {"yaml": yaml, "dotenv": dotenv, "openai": openai}):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    module._PROMPT_YAML_PATH = str(ROOT / "rr/set_level_rubric_scoring.yaml")
    module._call_gpt = lambda prompt: reply
    return module


def sample():
    return {"query": "Q", "scenario": "rag", "initial_list": [
        {"content": "A", "Source Authority": None},
        {"content": "B", "Source Authority": 0.8}],
        "rubrics": {"set_level_rubrics": {
            "Consistency": [{"description": "No conflicting evidence", "weight": 1}],
            "Relevance": [{"description": "Relevant evidence", "weight": 3}]},
            "doc_level_rubrics": {"Source Authority": [{"description": "Authoritative", "weight": 2}]}}}


class TextJudge:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class UpstreamTests(unittest.TestCase):
    def test_sources_match_git_blob_hashes(self):
        self.assertEqual(len(verify_sources()["files"]), 7)

    def test_function_bodies_are_original(self):
        original = ast.parse((ROOT / "rr/set_selection.py").read_text())
        module = rr_module(lambda _: "9 8")
        for node in original.body:
            if isinstance(node, ast.FunctionDef) and node.name in RR_FUNCTIONS:
                self.assertEqual(getattr(module, node.name).__code__.co_firstlineno, node.lineno)

    def test_reward_and_prompt_match_complete_original(self):
        config = load(PROJECT / "configs/b1.toml")
        config["reward"]["judge_failure_policy"] = "upstream_drop"
        for scenario in ("rag", "deepresearch"):
            for reply in ("9 8", "0 10", "invalid", "9 8 7"):
                for selection in ("[2] [1]", "notes [1] [1] [999]", "[999]", "no ids", None):
                    with self.subTest(scenario=scenario, reply=reply, selection=selection):
                        extra = sample()
                        extra.update(scenario=scenario, thinking="Intent")
                        original = full_original(reply)
                        expected_prompts = []
                        def complete(prompt):
                            expected_prompts.append(prompt)
                            return reply
                        original._call_gpt = complete
                        judge = TextJudge(reply)
                        with self._quiet(), patch.object(original.logger, "disabled", True):
                            expected = original.compute_score(selection, extra_info=extra)
                            result = score_selection(selection, extra, config, judge)
                        self.assertEqual(result["score"], expected)
                        self.assertEqual(judge.prompts, expected_prompts)

    @staticmethod
    def _quiet():
        import logging
        return patch.object(logging.getLogger("setwise.upstream.rr"), "disabled", True)

    def test_official_rubric_order_and_local_doc_numbers(self):
        judge = TextJudge("9 8")
        score_selection("[2]", sample(), load(PROJECT / "configs/b1.toml"), judge)
        prompt = judge.prompts[0]
        self.assertLess(prompt.index("1. Relevance"), prompt.index("2. Consistency"))
        self.assertIn("[1] B", prompt)
        self.assertNotIn("[2] B", prompt)
        self.assertNotIn("Return ONLY JSON", prompt)

    def test_missing_doc_prescores_match_original(self):
        config = load(PROJECT / "configs/b1.toml")
        extra = sample()
        expected = full_original("9 8").compute_score("[1]", extra_info=extra)
        self.assertEqual(score_selection("[1]", extra, config, TextJudge("9 8"))["score"], expected)

    def test_official_drop_failure_is_explicit(self):
        config = load(PROJECT / "configs/b1.toml")
        config["reward"]["judge_failure_policy"] = "upstream_drop"
        judge = TextJudge("not a score")
        with self._quiet():
            result = score_selection("[2]", sample(), config, judge)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["judge_incomplete"], 1)
        self.assertEqual(len(judge.prompts), 4)

    def test_active_ato_templates_only_have_declared_changes(self):
        upstream = ROOT / "ato"
        active = PROJECT / "src/setwise/prompts"
        for name in ("self_selector.yaml", "self_reflector.yaml"):
            self.assertEqual((active / name).read_bytes(), (upstream / name).read_bytes())
        original = (upstream / "adaptive_hints.yaml").read_text()
        expected = original.replace("three tiers / nine dimensions, scored 0-10",
            "RR set-level and document-level rubrics, aggregate reward 0-1").replace("out of 10 by the judge", "out of 1 by the judge")
        self.assertEqual((active / "adaptive_hints.yaml").read_text(), expected)

    def test_real_data_requires_frozen_prompt(self):
        with self.assertRaises(ValueError):
            student_prompt({"split": "train", "query": "Q", "docs": []})
        frozen = [{"role": "user", "content": "Frozen original prompt"}]
        self.assertEqual(student_prompt({"prompt": frozen}), frozen)

    def test_request_preserves_original_generation_parameters(self):
        judge = OpenAIJudge(load(PROJECT / "configs/b1.toml")["judge"])
        payload, _ = judge._request("Exact original prompt")
        self.assertEqual(set(json.loads(payload)), {"model", "messages"})

    def test_parallel_rollouts_do_not_share_judge_state(self):
        config = load(PROJECT / "configs/b1.toml")
        # Independent full-original imports touch sys.modules, so compute expected
        # results before the parallel section; workers only exercise the adapter.
        expected = [full_original(f"{i} 8").compute_score("[2]", extra_info=sample()) for i in range(8)]
        def worker(index):
            extra = sample()
            extra["query"] = f"Unique query {index}"
            judge = TextJudge(f"{index} 8")
            result = score_selection("[2]", extra, config, judge)
            return result["score"], judge.prompts[0], extra["query"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(8)))
        for index, (actual, prompt, query) in enumerate(results):
            self.assertEqual(actual, expected[index])
            self.assertIn(query, prompt)


if __name__ == "__main__":
    unittest.main()
