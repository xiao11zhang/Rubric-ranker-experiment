import ast
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opsd.common import digest, messages
from .audit import reviewed
from .entities import candidate, teacher_row, transform, validate_candidate
from .extract import convert, prefix
from .full_validation import select_steps
from .reporting import actual_report
from . import extract
from .protocol import actual_baseline_prefix
from .prepare import check_provenance


def fixture():
    rubrics = {'set_level_rubrics': {'Relevance': [
        {'description': 'Alice Smith studied at Oxford in 1990. Alice did not publish 3 books before 2001.', 'weight': 5},
        {'description': 'Compare Alice Smith and Bob; Oxford is not the same as London. Discuss the novel 1984.', 'weight': 3}]},
        'doc_level_rubrics': {'Source Authority': []}}
    row = {'id': 'a', 'query': 'Where did Alice Smith study?', 'docs': [{'content': 'original document'}],
           'rubrics': rubrics, 'rubric_hash': digest(rubrics)}
    entities = [{'type': 'person', 'mentions': ['Alice Smith', 'Alice']},
                {'type': 'person', 'mentions': ['Bob']}, {'type': 'organization', 'mentions': ['Oxford']},
                {'type': 'location', 'mentions': ['London']}, {'type': 'work', 'mentions': ['1984'],
                 'contexts': [{'path': ['set_level_rubrics', 'Relevance', 1, 'description'], 'quote': 'the novel 1984'}]}]
    return row, entities


class Group05Test(unittest.TestCase):
    def test_numeric_work_does_not_remove_same_surface_year(self):
        rubric = {'description': 'Discuss the novel 1984 and events in 1984; preserve at least 3 sources.'}
        entity = {'type': 'work', 'mentions': ['1984'], 'contexts': [{'path': ['description'], 'quote': 'the novel 1984'}]}
        output, _, _ = transform(rubric, [entity])
        self.assertEqual(output['description'], 'Discuss the novel <WORK_A> and events in 1984; preserve at least 3 sources.')

    def test_unscoped_numeric_work_is_not_globally_replaced(self):
        with self.assertRaisesRegex(ValueError, 'exact occurrence contexts'):
            transform({'description': 'Read 1984 in 1984.'}, [{'type': 'work', 'mentions': ['1984']}])

    def test_independent_year_cannot_be_declared_a_numeric_work(self):
        entity = {'type': 'work', 'mentions': ['2010'], 'contexts': [{'path': ['description'], 'quote': 'the year 2010'}]}
        with self.assertRaisesRegex(ValueError, 'year/condition'):
            transform({'description': 'Preserve the year 2010.'}, [entity])

    def test_ambiguous_context_is_rejected(self):
        entity = {'type': 'work', 'mentions': ['1984'], 'contexts': [{'path': ['description'], 'quote': '1984'}]}
        with self.assertRaisesRegex(ValueError, 'exactly one occurrence'):
            transform({'description': 'Read 1984 in 1984.'}, [entity])

    def test_context_roundtrip_is_bound_to_candidate_hash(self):
        row, entities = fixture()
        item = candidate(row, entities, 'fingerprint')
        validate_candidate(row, item)
        changed = deepcopy(item)
        changed['entities'][-1]['contexts'][0]['quote'] = '1990'
        with self.assertRaises(ValueError):
            validate_candidate(row, changed)

    def test_exact_replacement_preserves_conditions_and_metadata(self):
        row, entities = fixture()
        source = deepcopy(row)
        output, normalized, changes = transform(row['rubrics'], entities)
        descriptions = output['set_level_rubrics']['Relevance']
        self.assertEqual(descriptions[0]['description'], '<PERSON_A> studied at <ORG_A> in 1990. <PERSON_A> did not publish 3 books before 2001.')
        self.assertEqual(descriptions[1]['description'], 'Compare <PERSON_A> and <PERSON_B>; <ORG_A> is not the same as <PLACE_A>. Discuss the novel <WORK_A>.')
        self.assertEqual([r['weight'] for r in descriptions], [5, 3])
        self.assertEqual(output['doc_level_rubrics'], row['rubrics']['doc_level_rubrics'])
        self.assertEqual(row, source)
        self.assertEqual(normalized[0]['placeholder'], '<PERSON_A>')
        self.assertEqual(len(changes), 8)

    def test_invalid_entities_rejected_no_substring_corruption(self):
        row, _ = fixture()
        cases = [[{'type': 'person', 'mentions': ['Ali']}],
                 [{'type': 'date', 'mentions': ['1990']}],
                 [{'type': 'location', 'mentions': ['1990']}],
                 [{'type': 'person', 'mentions': ['Alice Smith']}, {'type': 'organization', 'mentions': ['Smith']}],
                 [{'type': 'person', 'mentions': ['Bob', 'Bob']}]]
        for entities in cases:
            with self.assertRaises(ValueError):
                transform(row['rubrics'], entities)

    def test_teacher_only_and_no_entity_case(self):
        row, entities = fixture()
        item = candidate(row, entities, 'f')
        changed = teacher_row(row, {'a': item})
        self.assertEqual(messages(row), messages(changed))
        self.assertNotEqual(messages(row, teacher=True), messages(changed, teacher=True))
        empty = candidate(row, [], 'f')
        self.assertEqual(empty['rubrics'], row['rubrics'])
        self.assertEqual(empty['status'], 'no_entities_detected')
        self.assertEqual(empty['changes'], [])

    def test_reviews_are_bound_to_transformation(self):
        row, entities = fixture()
        item = candidate(row, entities, 'f')
        review = {'id': row['id'], 'original_rubric_hash': item['original_rubric_hash'],
                  'rubric_hash': item['rubric_hash'], 'decision': 'approved', 'reviewer': 'test'}
        self.assertEqual(reviewed([row], [item], [review])['a'], item)
        for key, value in [('decision', 'pending'), ('reviewer', ''), ('rubric_hash', 'old')]:
            invalid = {**review, key: value}
            with self.assertRaises(ValueError): reviewed([row], [item], [invalid])
        altered = deepcopy(item)
        altered['rubrics']['set_level_rubrics']['Relevance'][0]['weight'] = 1
        altered['rubric_hash'] = digest(altered['rubrics'])
        with self.assertRaises(ValueError): validate_candidate(row, altered)

    def test_conversion_errors_never_fall_back_to_original(self):
        row, _ = fixture()
        for text in ('garbage', '{"entities":[{"type":"person","mentions":["Nobody"]}]}'):
            item = convert(row, text, 'f')
            self.assertEqual(item['status'], 'conversion_error')
            self.assertIsNone(item['rubrics'])
        self.assertEqual(convert(row, '{"entities":[]}', 'f')['status'], 'no_entities_detected')

    def test_extraction_reasoning_is_explicit_and_strict(self):
        row, _ = fixture()
        raw = '<think>Identify actual names only.</think>\n{"entities":[]}'
        self.assertEqual(convert(row, raw, 'f')['status'], 'conversion_error')
        self.assertEqual(convert(row, raw, 'f', allow_thinking=True)['status'], 'no_entities_detected')
        for bad in ('<think>unfinished', '<think>x</think>{"entities":[]}}'):
            self.assertEqual(convert(row, bad, 'f', allow_thinking=True)['status'], 'conversion_error')

    def test_resume_binds_source_order_fingerprint_and_newline(self):
        row, entities = fixture()
        item = candidate(row, entities, 'f')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'candidates.jsonl'
            path.write_text(json.dumps(item) + '\n', encoding='utf-8')
            self.assertEqual(prefix(path, [row], 'f'), [item])
            with self.assertRaises(ValueError): prefix(path, [row], 'other')
            with self.assertRaises(ValueError): prefix(path, [{**row, 'query': 'changed'}], 'f')
            path.write_text(json.dumps(item), encoding='utf-8')
            with self.assertRaises(ValueError): prefix(path, [row], 'f')

    def test_actual_report_requires_exact_ids_and_teacher_hashes(self):
        row, entities = fixture()
        item = candidate(row, entities, 'f')
        plan = {'config': {'gradient_accumulation': 1}, 'execution_stop_after_step': 1, 'order_ids': ['a']}
        sample = {'id': 'a', 'teacher_rubric_hash': item['rubric_hash'], 'original_rubric_hash': item['original_rubric_hash']}
        entries = [{'step': 1, 'samples': [sample]}]
        self.assertTrue(actual_report(plan, entries, {'a': item})['matches_baseline_prefix'])
        with self.assertRaises(ValueError): actual_report(plan, [{'step': 2, 'samples': [sample]}], {'a': item})
        with self.assertRaises(ValueError): actual_report(plan, [{'step': 1, 'samples': [{**sample, 'teacher_rubric_hash': 'wrong'}]}], {'a': item})

    def test_full_development_selection_fixed_before_scores(self):
        history = [{'step': step, 'metrics': {'balanced_macro': {'all_annotated_supports_at_5': score, 'support_f1_selected': 0.5}}}
                   for step, score in [(0, 0.3), (100, 0.4), (200, 0.6), (300, 0.5), (400, 0.4), (500, 0.4)]]
        plan = {'small_eval_steps': [0, 100, 200, 300, 400, 500]}
        self.assertEqual(select_steps(plan, history), [0, 200, 300, 500])
        with self.assertRaises(ValueError): select_steps(plan, history[:-1])

    def test_baseline_actual_prefix_is_verified(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            entry = {'step': 1, 'samples': [{'id': 'a'}, {'id': 'b'}]}
            (root / 'train.jsonl').write_text(json.dumps(entry) + '\n', encoding='utf-8')
            c = {'stop_after_step': 1, 'gradient_accumulation': 2}
            baseline = {'order_ids': ['a', 'b']}
            self.assertEqual(actual_baseline_prefix(root, baseline, c), digest([entry]))
            with self.assertRaises(ValueError):
                actual_baseline_prefix(root, {'order_ids': ['b', 'a']}, c)

    def test_extraction_provenance_must_match(self):
        row, entities = fixture()
        manifest = {'order_ids': ['a'], 'train_sha256': 'source'}
        item = candidate(row, entities, digest(manifest))
        check_provenance([item], manifest, [row], 'source')
        with self.assertRaises(ValueError): check_provenance([item], manifest, [row], 'different')
        with self.assertRaises(ValueError): check_provenance([{**item, 'pipeline_fingerprint': 'wrong'}], manifest, [row], 'source')

    def test_fake_extraction_cli_persists_and_resumes_without_reloading(self):
        row, entities = fixture()
        raw = json.dumps({'entities': entities})
        class Response:
            def tolist(self): return [3, 9]
        class Generated:
            def __getitem__(self, key): return Response()
        tokenizer = SimpleNamespace(eos_token_id=9, apply_chat_template=lambda *a, **kw: [1, 2],
                                    decode=lambda *a, **kw: raw)
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16000),
                                eval=lambda: None, generate=lambda **kw: Generated())
        model.to = lambda device: model
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model))
        torch = SimpleNamespace(bfloat16='bf16', tensor=lambda *a, **kw: None,
                                ones_like=lambda a: None, inference_mode=contextlib.nullcontext)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder).resolve()
            model_dir = root / 'model'
            model_dir.mkdir()
            (model_dir / 'config.json').write_text('{}', encoding='utf-8')
            config = root / 'config.toml'
            config.write_text('test=1', encoding='utf-8')
            source, baseline_plan = root / 'source.jsonl', root / 'baseline.json'
            source.write_text(json.dumps(row) + '\n', encoding='utf-8')
            baseline_plan.write_text('{}', encoding='utf-8')
            output = root / 'data/candidates.jsonl'
            c = {'derived_data': str(output), 'baseline_plan': str(baseline_plan),
                 'train_data': str(source), 'model': str(root / 'sft-model')}
            argv = ['extract', '--config', str(config), '--model', str(model_dir), '--execute']
            with patch.object(extract, 'load', return_value=(c, {}, [row], [row], root / 'baseline-run')):
                with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    extract.main()
                item = json.loads(output.read_text(encoding='utf-8'))
                validate_candidate(row, item)
                self.assertEqual(item['status'], 'needs_review')
                before = output.read_bytes()
                with patch.object(sys, 'argv', argv + ['--resume']), contextlib.redirect_stdout(io.StringIO()):
                    extract.main()
                self.assertEqual(output.read_bytes(), before)
                (model_dir / 'config.json').write_text('{"changed":true}', encoding='utf-8')
                with patch.object(sys, 'argv', argv + ['--resume']):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        extract.main()

    def test_original_opsd_micro_update_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        def micro(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'index')
            for n in ast.walk(loop):
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tp' for t in n.targets):
                    n.value = ast.Constant('teacher-prompt')
                if isinstance(n, ast.Dict):
                    pairs = [(k, v) for k, v in zip(n.keys, n.values) if not isinstance(k, ast.Constant) or k.value not in ('teacher_rubric_hash', 'original_rubric_hash')]
                    n.keys, n.values = [x[0] for x in pairs], [x[1] for x in pairs]
            return ast.dump(loop, include_attributes=False)
        self.assertEqual(micro(root / 'full_run/train.py'), micro(root / 'group05_deentity/train.py'))


if __name__ == '__main__':
    unittest.main()
