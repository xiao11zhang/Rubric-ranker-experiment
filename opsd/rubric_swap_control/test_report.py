import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opsd.rubric_swap.reporting import actual_assignment_report
from . import training_only


class ReportTest(unittest.TestCase):
    def fixture(self, root):
        run = root / 'new-run'
        run.mkdir()
        baseline = root / 'opsd/runs/rag-full-epoch-001'
        baseline.mkdir(parents=True)
        mapping = []
        ids = [str(i) for i in range(4000)]
        for i, uid in enumerate(ids):
            fallback = i < 300
            mapping.append({'id': uid, 'donor_id': uid if fallback else str(300 + (i - 300 + 1) % 3700),
                'assignment_type': 'original_fallback' if fallback else 'same_domain_mismatch',
                'primary_domain': 'original_fallback' if fallback else 'sports'})
        samples = [{'id': r['id'], 'rubric_donor_id': r['donor_id'],
                    'rubric_assignment_type': r['assignment_type'], 'rubric_assignment_domain': r['primary_domain']}
                   for r in mapping[:2000]]
        entries = [{'step': start // 4 + 1, 'samples': samples[start:start + 4]} for start in range(0, 2000, 4)]
        (run / 'train.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
        history = [{'step': step, 'metrics': {'balanced_macro': {'all_annotated_supports_at_5': .65 + step / 10000,
            'support_f1_selected': .8, 'support_recall_at_5': .85, 'valid': 1.0}}} for step in (0, 100, 200, 300, 400, 500)]
        for path in (run, baseline):
            (path / 'small-validation-history.json').write_text(json.dumps(history))
        plan = {'config': {'gradient_accumulation': 4, 'pi_mode': 'same_domain_mismatch'},
                'order_ids': ids, 'rubric_mapping': mapping, 'evaluation_policy': {'selection_rule': 'original rule'}}
        (run / 'rubric-assignment-summary.json').write_text(json.dumps(actual_assignment_report(plan, entries, 500)))
        return run, plan

    def test_final_report_explicitly_excludes_full_development(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, plan = self.fixture(root)
            with patch.multiple(training_only, ROOT=root, RUN=run):
                result = training_only.write_report(plan, completed=True)
            self.assertTrue(result['completed'])
            self.assertEqual(result['actual_training_assignments']['records'], 2000)
            self.assertEqual(result['actual_training_assignments']['counts']['original_fallback'], 300)
            self.assertFalse(result['full_development_evaluated'])
            self.assertIn('未进行 9,814 条全量开发评估', (run / 'REPORT.md').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((run / 'full-validation-status.json').read_text())['queries_evaluated'], 0)

    def test_existing_full_stage_is_not_falsely_reported_as_unrun(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, plan = self.fixture(root)
            (root / 'server-logs').mkdir()
            (root / 'server-logs/rubric-mixed-001-full_development.log').write_text('already started')
            with patch.multiple(training_only, ROOT=root, RUN=run):
                with self.assertRaisesRegex(ValueError, 'cannot claim'):
                    training_only.write_report(plan, completed=True)


if __name__ == '__main__':
    unittest.main()
