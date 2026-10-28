from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

from . import api_sampling as audit
from .api_extract import process, read_records
from .test_api_extract import FakeAPI, rows


class APISamplingTest(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent)

    def test_sampling_is_fixed_uses_subset_and_does_not_consume_global_rng(self):
        selected = [{'id': str(i)} for i in range(1000)]
        state = random.getstate()
        required = audit.required_ids(selected, {}, {})
        self.assertEqual(state, random.getstate())
        self.assertEqual(required, audit.required_ids(selected, {}, {}))
        self.assertLess(len(required), 100)
        self.assertTrue(set(str(i) for i in range(5)).issubset(required))
        imported = {str(i): {} for i in range(100)}
        self.assertFalse(set(audit.required_ids(selected, {}, imported)) & set(imported))

    def test_numeric_work_and_shared_alias_are_always_reviewed(self):
        selected = [{'id': str(i)} for i in range(1000)]
        records = {'998': {'status': 'validated', 'entities': [{'type': 'work', 'mentions': ['1984']}]},
                   '999': {'status': 'validated', 'entities': [{'type': 'work', 'mentions': ['Hamlet']}, {'type': 'person', 'mentions': ['Hamlet']}]}}
        self.assertTrue({'998', '999'}.issubset(audit.required_ids(selected, records, {})))

    def test_program_validation_does_not_auto_approve_semantics(self):
        selected = rows()[1:]
        with self.temporary() as folder:
            base = Path(folder)
            api = FakeAPI()
            record = process(selected[0], api, 'f', base / 'records/0001.json')
            records = {'b': record}
            self.assertEqual(len(api.calls), 1)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                audit.check(base, selected, records, {})
            result = audit.create(base, selected, records, {})
            self.assertEqual(result['api_calls'], 0)
            template = json.loads((base / 'audit/decisions.template.json').read_text(encoding='utf-8'))
            self.assertEqual(template[0]['decision'], 'pending')
            with self.assertRaises(ValueError): audit.approve(base, selected, records, template)
            template[0].update(decision='approved', notes='Agent checked no entities and year/quantity preserved.')
            audit.approve(base, selected, records, template)
            required, checked = audit.check(base, selected, records, {})
            self.assertEqual(required, ['b'])
            self.assertTrue(checked['b'][0]['semantic_reviewed'])
            changed = deepcopy(record)
            changed['candidate']['rubric_hash'] = 'changed'
            with self.assertRaises(ValueError): audit.check(base, selected, {'b': changed}, {})

    def test_agent_can_fix_failed_annotation_without_second_api_call(self):
        selected = rows()[:1]
        with self.temporary() as folder:
            base = Path(folder)
            api = FakeAPI([{'type': 'person', 'mentions': ['missing name']}])
            record = process(selected[0], api, 'f', base / 'records/0001.json')
            self.assertEqual(record['status'], 'error')
            records = {'a': record}
            corrections = [{'id': 'a', 'entities': [{'type': 'person', 'mentions': ['Alice']}, {'type': 'location', 'mentions': ['Paris']}],
                            'notes': 'Agent corrected hallucinated mention; checked both names, 1990 and 3.'}]
            audit.repair(base, selected, records, 'f', corrections)
            self.assertEqual(len(api.calls), 1)
            self.assertTrue(records['a']['agent_corrected'])
            self.assertEqual(records['a']['status'], 'validated')
            self.assertEqual(len(list((base / 'attempts').glob('*.json'))), 1)
            loaded = read_records(base, selected, 'f')
            audit.check(base, selected, loaded, {})


if __name__ == '__main__':
    unittest.main()
