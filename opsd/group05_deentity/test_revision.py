import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import agent_annotations as agent
from . import revise_unreviewed as revision
from . import test_agent_annotations as fixture_module


class RevisionTest(unittest.TestCase):
    def run_case(self, mode):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root, config, _, source = fixture_module.AgentAnnotationsTest().make_fixture(folder)
            base = root / 'agent'
            with patch.object(agent, 'BASE', base), patch.object(revision, 'BASE', base), \
                    patch.object(agent, 'sources', return_value=source), contextlib.redirect_stdout(io.StringIO()):
                agent.initialize(config, 2)
                original = root / 'original.json'
                value = {'batch_id': 'batch-0001', 'records': [
                    {'id': 'a', 'entities': [{'type': 'person', 'mentions': ['Alice Smith']}]},
                    {'id': 'b', 'entities': []}]}
                original.write_text(json.dumps(value), encoding='utf-8')
                agent.commit('batch-0001', original)
                commit_path = base / 'batches/batch-0001.commit.json'
                old_commit = commit_path.read_bytes()
                new = root / 'revised.json'
                value['records'][0]['entities'] = [{'type': 'location', 'mentions': ['Paris']}]
                if mode == 'invalid':
                    value['records'][0]['entities'][0]['mentions'] = ['Missing Name']
                if mode == 'reviewed':
                    (base / 'batches/batch-0001.review.json').write_text('[]')
                new.write_text(json.dumps(value), encoding='utf-8')
                if mode != 'success':
                    with self.assertRaises(ValueError):
                        revision.revise('batch-0001', new, 'fixture revision')
                    self.assertEqual(commit_path.read_bytes(), old_commit)
                    self.assertFalse((base / 'amendments/batch-0001-revision-001').exists())
                    return
                revision.revise('batch-0001', new, 'fixture revision')
                archive = base / 'amendments/batch-0001-revision-001'
                self.assertEqual((archive / commit_path.name).read_bytes(), old_commit)
                self.assertEqual(json.loads(commit_path.read_text())['annotations_path'], agent.rel(new))
                self.assertFalse((base / 'batches/batch-0001.review.json').exists())
                self.assertEqual(json.loads((base / 'progress.json').read_text())['agent_reviewed_records'], 0)
                agent.context('batch-0001')
                (archive / 'revision.json').write_text('{"tampered":true}')
                with self.assertRaisesRegex(ValueError, 'amendment evidence changed'):
                    agent.context('batch-0001')

    def test_revision_archives_old_evidence_and_does_not_approve(self):
        self.run_case('success')

    def test_invalid_annotation_preserves_old_batch(self):
        self.run_case('invalid')

    def test_reviewed_batch_cannot_be_revised_silently(self):
        self.run_case('reviewed')


if __name__ == '__main__':
    unittest.main()
