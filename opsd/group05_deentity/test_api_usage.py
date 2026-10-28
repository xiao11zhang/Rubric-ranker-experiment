import json
from pathlib import Path
import tempfile
import unittest

from .api_usage import summarize


class UsageTest(unittest.TestCase):
    def test_counts_archived_usage_once_and_keeps_missing_unknown(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            run = Path(folder) / 'run-001'
            (run / 'records').mkdir(parents=True)
            (run / 'attempts').mkdir()
            (run / 'run.json').write_text('{}')
            extract = {'response_id': 'x', 'usage': {'prompt_tokens': 12,
                       'completion_tokens': 3, 'total_tokens': 15}}
            old = {'id': 'a', 'status': 'error', 'extraction_response': extract}
            new = {**old, 'status': 'approved', 'review_response': {'response_id': 'y'}}
            (run / 'attempts/a.json').write_text(json.dumps(old))
            (run / 'records/a.json').write_text(json.dumps(new))
            report = summarize(Path(folder))
            self.assertEqual(report['provider_reported_totals']['total_tokens'], 15)
            self.assertEqual(report['phases']['extraction']['unique_responses'], 1)
            self.assertIsNone(report['phases']['review']['total_tokens'])
            self.assertEqual(report['phases']['review']['responses_without_usage'], 1)
            self.assertEqual(report['failed_or_rejected_record_attempts'], 1)
            self.assertIsNone(summarize(Path(folder), ['run-other'])['provider_reported_totals']['total_tokens'])

    def test_no_provider_usage_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            report = summarize(Path(folder))
            self.assertIsNone(report['provider_reported_totals']['total_tokens'])


if __name__ == '__main__':
    unittest.main()
