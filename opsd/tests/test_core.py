import unittest

from opsd.common import check_disjoint, digest, messages, parse_output


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.row = {"query": "Which year?", "docs": [{"content": "First"}, {"content": "Second"}],
                    "rubrics": {"set_level_rubrics": {"Relevance": [{"description": "SECRET_REFERENCE_FACT", "weight": 5}]}}}

    def test_pi_is_teacher_only(self):
        self.assertNotIn("SECRET_REFERENCE_FACT", str(messages(self.row)))
        self.assertIn("SECRET_REFERENCE_FACT", str(messages(self.row, teacher=True)))
        self.assertEqual(messages(self.row), messages(self.row, teacher=True, mode="none"))

    def test_output_contract(self):
        self.assertTrue(parse_output("[2] [1]", 2, 2)["valid"])
        for value in ("[1] [1]", "[0]", "[3]", "answer [1]", "", "[1]"):
            self.assertFalse(parse_output(value, 2, 2)["valid"])

    def test_order_changes_hash(self):
        self.assertNotEqual(digest(self.row["docs"]), digest(self.row["docs"][::-1]))

    def test_overlap_normalization(self):
        with self.assertRaises(ValueError):
            check_disjoint([{"query": " Same   Query "}], [{"query": "same query"}])


class ObjectiveTests(unittest.TestCase):
    def test_gradient_and_response_alignment(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed on this local machine")
        from opsd.objective import loss, sampled_log_probs
        logits = torch.tensor([[[3., 1.], [1., 3.], [2., 0.], [0., 0.]]], requires_grad=True)
        tokens = torch.tensor([[0, 0, 1, 0]])
        lp = sampled_log_probs(logits, tokens, 2)
        self.assertEqual(lp.shape, (2,))
        expected = torch.log_softmax(logits[0, 1:3], -1).gather(1, torch.tensor([[1], [0]])).squeeze(1)
        torch.testing.assert_close(lp, expected)
        slp = torch.tensor([-2., -1.], requires_grad=True)
        tlp = torch.tensor([-1., -2.], requires_grad=True)
        objective, _ = loss(slp, tlp)
        objective.backward()
        torch.testing.assert_close(slp.grad, torch.tensor([-.5, .5]))
        self.assertIsNone(tlp.grad)


if __name__ == "__main__":
    unittest.main()
