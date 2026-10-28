"""Tests for diagnostic accounting, independent of training."""
import unittest

from opsd.validation.run_checks import summarize
from opsd.validation.truncation_audit import diagnose, pattern


class DiagnosticAccountingTests(unittest.TestCase):
    def test_repetition_before_cap_is_not_only_truncation(self):
        record = {"ids": [1, 2, 1, 2], "output": "[1] [2] [1] [2] [", "ended_eos": False, "response_tokens": [0] * 128}
        value = diagnose(record, 4)
        self.assertEqual(value["category"], "duplicate_before_cap")
        self.assertEqual(value["first_duplicate"], {"position": 3, "id": 1})
        self.assertEqual(value["repeated_block"]["block"], [1, 2])

    def test_cap_with_only_unfinished_tail(self):
        record = {"ids": [1, 2], "output": "[1] [2] [3", "ended_eos": False, "response_tokens": [0] * 128}
        self.assertEqual(diagnose(record, 4)["category"], "tail_syntax_only_at_cap")
        self.assertEqual(diagnose(record, 4)["tail_after_last_complete_id"], " [3")

    def test_syntax_valid_is_not_eos_complete(self):
        rows = [{"id": "a", "docs": [1, 2]}, {"id": "b", "docs": [1, 2]}]
        records = [
            {"id": "a", "output": "[1]", "ids": [1], "valid": True,
             "ended_eos": False, "response_tokens": [1, 2]},
            {"id": "b", "output": "[2] [2]", "ids": [2, 2], "valid": False,
             "ended_eos": True, "response_tokens": [3, 4, 5]},
        ]
        result = summarize(records, rows)
        self.assertEqual(result["valid"], 0.5)
        self.assertEqual(result["complete_valid"], 0)
        self.assertEqual(result["ended_eos"], 0.5)
        self.assertEqual(result["duplicate"], 0.5)
        self.assertEqual(result["tokens"], 2.5)


if __name__ == "__main__":
    unittest.main()
