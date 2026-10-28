import unittest
from .full_validation import comparison_selection


def history(best, f1_tie=None):
    result = []
    for step in (0, 100, 200, 300, 400, 500):
        result.append({'step': step, 'metrics': {'balanced_macro': {
            'all_annotated_supports_at_5': .9 if step == best or step == f1_tie else .7,
            'support_f1_selected': .9 if step == best else .8}}})
    return result


class EvaluationSelectionTest(unittest.TestCase):
    def setUp(self):
        self.plan = {'execution_stop_after_step': 500, 'evaluation_policy': {
            'small_dev_steps': [0, 100, 200, 300, 400, 500], 'full_dev_fixed_steps': [0, 300, 500]}}

    def test_same_budget_and_matched_checkpoints(self):
        correct = history(300)
        correct.append({'step': 1000, 'metrics': {'balanced_macro': {
            'all_annotated_supports_at_5': 1, 'support_f1_selected': 1}}})
        selected = comparison_selection(self.plan, history(200), correct)
        self.assertEqual(selected['matched_steps'], [0, 200, 300, 500])
        self.assertEqual(selected['correct_best_small_step'], 300)
        self.assertEqual(selected['new_best_small_step'], 200)
        self.assertFalse(selected['final_test_accessed'])

    def test_tie_breaking_and_checkpoint_deduplication(self):
        selected = comparison_selection(self.plan, history(500, f1_tie=100), history(300))
        self.assertEqual(selected['new_best_small_step'], 500)
        self.assertEqual(selected['matched_steps'], [0, 300, 500])

    def test_incomplete_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            comparison_selection(self.plan, history(300)[:-1], history(300))
