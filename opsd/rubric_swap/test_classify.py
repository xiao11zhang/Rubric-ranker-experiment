import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opsd.common import digest
from . import classify
from .rubrics import build_mapping
from .test_rubrics import fixture


class ClassificationTests(unittest.TestCase):
    def test_prompt_uses_only_query(self):
        rows, _ = fixture()
        prompt = classify.classifier_messages(rows[0]['query'])
        self.assertEqual(json.loads(prompt[1]['content']), {'query': rows[0]['query']})
        self.assertNotIn('rubrics', json.dumps(prompt))
        self.assertEqual(tuple(classify.DOMAIN_DESCRIPTIONS), classify.DOMAINS)

    def test_json_and_uncertainty(self):
        raw = '{"primary_domain":"sports","uncertain":false,"reason":"体育主题"}'
        self.assertEqual(classify.parse_prediction('```json\n' + raw + '\n```')['primary_domain'], 'sports')
        unknown = '{"primary_domain":"unknown","uncertain":false,"reason":"歧义"}'
        self.assertTrue(classify.parse_prediction(unknown)['uncertain'])
        for bad in ('answer: ' + raw, '[]', raw.replace('sports', 'bad'), raw.replace('false', '"false"')):
            with self.assertRaises((ValueError, TypeError)):
                classify.parse_prediction(bad)

    def test_failed_predictions_become_review_items(self):
        rows, _ = fixture()
        for raw, failure in (('not json', None), ('{}', None), ('', 'input too long')):
            label = classify.make_label(rows[0], raw, 'fingerprint', failure=failure)
            self.assertEqual(label['primary_domain'], 'unknown')
            self.assertTrue(label['uncertain'])
            self.assertEqual(label['query_hash'], digest(rows[0]['query']))

    def test_report_and_mapping_compatibility(self):
        rows, _ = fixture((4,))
        raw = '{"primary_domain":"sports","uncertain":false,"reason":"体育"}'
        labels = [classify.make_label(row, raw, 'f') for row in rows]
        self.assertEqual(len(build_mapping(rows, labels)), 4)
        self.assertTrue(classify.report(rows, labels)['structurally_ready_for_mapping'])
        labels[0]['uncertain'] = True
        self.assertEqual(classify.report(rows, labels)['needs_review_ids'], [rows[0]['id']])
        singleton = classify.report(rows[1:2], labels[1:2])
        self.assertFalse(singleton['structurally_ready_for_mapping'])

    def test_resume_prefix_rejects_changes_and_partial_line(self):
        rows, _ = fixture((2,))
        label = classify.make_label(rows[0], '{}', 'f')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'labels.jsonl'
            path.write_text(json.dumps(label) + '\n', encoding='utf-8')
            self.assertEqual(len(classify.load_prefix(path, rows, 'f')), 1)
            with self.assertRaises(ValueError): classify.load_prefix(path, rows, 'other')
            with self.assertRaises(ValueError): classify.load_prefix(path, rows[::-1], 'f')
            path.write_text(json.dumps(label), encoding='utf-8')
            with self.assertRaises(ValueError): classify.load_prefix(path, rows, 'f')

    def test_cli_fake_inference_resume_and_model_binding(self):
        rows, _ = fixture((2,))
        for row in rows:
            row['candidate_hash'] = digest(row['docs'])
            row['rubric_hash'] = digest(row['rubrics'])
        raw = '{"primary_domain":"sports","uncertain":false,"reason":"体育"}'
        class Response:
            def tolist(self): return [3, 9]
        class Generated:
            def __getitem__(self, key): return Response()
        tokenizer = SimpleNamespace(eos_token_id=9,
            apply_chat_template=lambda *a, **kw: [1, 2], decode=lambda *a, **kw: raw)
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096),
            eval=lambda: None, generate=lambda **kw: Generated())
        model.to = lambda device: model
        factory = SimpleNamespace(from_pretrained=lambda *a, **kw: model)
        fake_transformers = SimpleNamespace(AutoModelForCausalLM=factory,
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
        fake_torch = SimpleNamespace(bfloat16='bf16', tensor=lambda *a, **kw: None,
            ones_like=lambda x: None, inference_mode=contextlib.nullcontext)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            train = root / 'train.jsonl'
            train.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            model_dir = root / 'model'
            model_dir.mkdir()
            (model_dir / 'config.json').write_text('{}', encoding='utf-8')
            output = root / 'out/labels.jsonl'
            argv = ['classify', '--train', str(train), '--model', str(model_dir), '--output', str(output)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                classify.main()
            self.assertFalse(output.parent.exists())
            modules = {'torch': fake_torch, 'transformers': fake_transformers}
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv + ['--execute']), contextlib.redirect_stdout(io.StringIO()):
                classify.main()
            before = output.read_bytes()
            saved = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(build_mapping(rows, saved)), 2)
            with patch.object(sys, 'argv', argv + ['--execute', '--resume']), contextlib.redirect_stdout(io.StringIO()):
                classify.main()
            self.assertEqual(output.read_bytes(), before)
            (model_dir / 'config.json').write_text('{"changed":true}', encoding='utf-8')
            with patch.object(sys, 'argv', argv + ['--execute', '--resume']):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    classify.main()


if __name__ == '__main__':
    unittest.main()
