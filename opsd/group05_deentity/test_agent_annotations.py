from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opsd.common import digest
from opsd.dev_eval.prepare import ROOT, sha
from . import agent_annotations as agent
from .prepare import check_agent_provenance


class AgentAnnotationsTest(unittest.TestCase):
    def make_fixture(self, folder):
        root = Path(folder)
        config, train, baseline, log = [root / name for name in ('config.toml', 'train.jsonl', 'baseline.json', 'log.jsonl')]
        for path in (config, train, baseline, log):
            path.write_text('{}', encoding='utf-8')
        rows = [{'id': uid, 'query': 'Question ' + uid, 'docs': [{'content': 'passage'}],
            'rubrics': {'description': text, 'weight': 3}}
            for uid, text in (('a', 'Alice Smith studied in Paris before 1990.'),
                              ('b', 'At least 3 sources must agree before 1984.'))]
        c = {'train_data': agent.rel(train), 'baseline_plan': agent.rel(baseline)}
        return root, config, rows, (c, {}, rows, log, 'prefix-hash')

    def test_explicit_annotation_preview_and_agent_review(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root, config, rows, source = self.make_fixture(folder)
            base = root / 'agent'
            with patch.object(agent, 'BASE', base), patch.object(agent, 'sources', return_value=source):
                agent.initialize(config, 2)
                raw = root / 'annotations.json'
                raw.write_text(json.dumps({'batch_id': 'batch-0001', 'records': [
                    {'id': 'a', 'entities': [{'type': 'person', 'mentions': ['Alice Smith']},
                                             {'type': 'location', 'mentions': ['Paris']}]},
                    {'id': 'b', 'entities': []}]}), encoding='utf-8')
                agent.commit('batch-0001', raw)
                progress = json.loads((base / 'progress.json').read_text())
                self.assertEqual(progress['annotated_records'], 2)
                self.assertEqual(progress['agent_reviewed_records'], 0)
                preview = json.loads((base / 'batches/batch-0001.preview.json').read_text())
                self.assertIn('before 1990', preview[0]['fields'][0]['converted'])
                self.assertEqual(preview[1]['fields'][0]['original'], preview[1]['fields'][0]['converted'])
                decisions = root / 'decisions.json'
                values = {'batch_id': 'batch-0001', 'records': [{'id': r['id'], 'decision': 'approved',
                    'review_method': agent.IDENTITY['review_method'], 'notes': 'Agent checked source and converted field; numbers retained.'} for r in rows]}
                decisions.write_text(json.dumps(values), encoding='utf-8')
                agent.review('batch-0001', decisions)
                reviewed = json.loads((base / 'batches/batch-0001.review.json').read_text())
                self.assertTrue(all(r['reviewer'] == 'Codex agent' and not r['human_review_claimed'] for r in reviewed))
                self.assertEqual(json.loads((base / 'progress.json').read_text())['agent_reviewed_records'], 2)
                with self.assertRaisesRegex(ValueError, '2000'):
                    agent.finalize()

    def test_missing_decisions_or_wrong_order_never_auto_approve(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root, config, _, source = self.make_fixture(folder)
            with patch.object(agent, 'BASE', root / 'agent'), patch.object(agent, 'sources', return_value=source):
                agent.initialize(config, 2)
                raw = root / 'bad.json'
                raw.write_text(json.dumps({'batch_id': 'batch-0001', 'records': [{'id': 'b', 'entities': []}]}))
                with self.assertRaisesRegex(ValueError, 'coverage/order'):
                    agent.commit('batch-0001', raw)
                self.assertFalse((root / 'agent/batches/batch-0001.review.json').exists())

    def test_agent_provenance_checks_artifact_hash_and_rejects_fake_weights(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder).resolve()
            batch = root / 'batch.json'
            batch.write_text('{}')
            source = {'provider': 'agent_annotations', 'order_ids': ['a'], 'rules': {'keep': 'years'},
                'rules_sha256': digest({'keep': 'years'}), 'actual_baseline_prefix_sha256': 'prefix',
                'batches': [{'batch_id': 'batch-0001', 'path': 'batch.json', 'sha256': sha(batch), 'ids': ['a']}]}
            src = root / 'source.json'
            src.write_text(json.dumps(source))
            code = root / 'tool.py'
            code.write_text('# fixture')
            manifest = {**agent.IDENTITY, 'semantic_review_complete': True, 'rule_version': agent.RULE_VERSION,
                'records': 1, 'batches': 1, 'rules_sha256': source['rules_sha256'],
                'source_manifest_path': 'source.json', 'source_manifest_sha256': sha(src),
                'baseline_actual_prefix_sha256': 'prefix',
                'annotation_and_review_artifacts_sha256': {'source.json': sha(src), 'batch.json': sha(batch)},
                'code_sha256': {'tool.py': sha(code)}}
            check_agent_provenance(manifest, [{'id': 'a'}], root)
            forged = deepcopy(manifest)
            forged['model_files_sha256'] = {'fake.safetensors': 'fake'}
            with self.assertRaisesRegex(ValueError, 'impersonate'):
                check_agent_provenance(forged, [{'id': 'a'}], root)
            batch.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                check_agent_provenance(manifest, [{'id': 'a'}], root)


if __name__ == '__main__':
    unittest.main()
