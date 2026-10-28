import unittest
from .train import safe_stop_limit


class StopBoundaryTest(unittest.TestCase):
    def test_only_complete_saved_evaluated_boundaries(self):
        plan = {'config': {'steps': 1000}, 'checkpoint_steps': [10, 20] + list(range(100, 1001, 100)),
                'small_eval_steps': [0] + list(range(100, 1001, 100))}
        self.assertEqual(safe_stop_limit(plan, 500), 500)
        self.assertEqual(safe_stop_limit(plan, None), 1000)
        for step in (0, -1, 10, 20, 501, 1001):
            with self.assertRaises(ValueError):
                safe_stop_limit(plan, step)

    def test_frozen_500_boundary_is_default_and_cannot_be_overridden(self):
        plan = {'config': {'steps': 1000, 'stop_after_step': 500}, 'execution_stop_after_step': 500,
                'checkpoint_steps': [100, 200, 300, 400, 500, 1000],
                'small_eval_steps': [0, 100, 200, 300, 400, 500, 1000]}
        self.assertEqual(safe_stop_limit(plan, None), 500)
        self.assertEqual(safe_stop_limit(plan, 500), 500)
        for step in (100, 1000):
            with self.assertRaisesRegex(ValueError, 'frozen execution boundary'):
                safe_stop_limit(plan, step)


if __name__ == '__main__':
    unittest.main()
