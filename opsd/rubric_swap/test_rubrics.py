import ast
from collections import Counter
import copy
from pathlib import Path
import random
import unittest

from opsd.common import digest, messages
from .rubrics import DOMAINS, build_mapping, teacher_row
from .prepare import disjoint_output


def fixture(counts=(4, 3)):
    rows, labels = [], []
    for group, count in enumerate(counts):
        for i in range(count):
            uid = f'{group}-{i}'
            row = {'id': uid, 'query': 'question ' + uid, 'docs': [{'content': uid}],
                   'rubrics': [{'criterion': uid, 'weight': i + 1}]}
            rows.append(row)
            labels.append({'id': uid, 'query_hash': digest(row['query']),
                           'primary_domain': DOMAINS[group]})
    return rows, labels


class RubricTests(unittest.TestCase):
    def test_bijection_domain_hash_and_rng(self):
        rows, labels = fixture()
        before = random.getstate()
        mapping = build_mapping(rows, labels)
        self.assertEqual(before, random.getstate())
        self.assertEqual(mapping, build_mapping(rows, labels))
        self.assertEqual(Counter(x['id'] for x in mapping), Counter(x['donor_id'] for x in mapping))
        domains = {x['id']: x['primary_domain'] for x in labels}
        for pair in mapping:
            self.assertNotEqual(pair['id'], pair['donor_id'])
            self.assertEqual(domains[pair['id']], domains[pair['donor_id']])
            self.assertNotEqual(pair['recipient_rubric_hash'], pair['donor_rubric_hash'])

    def test_duplicate_rubrics_and_impossible_groups(self):
        rows, labels = fixture((6,))
        for i in range(3):
            rows[i]['rubrics'] = rows[0]['rubrics']
        self.assertEqual(len(build_mapping(rows, labels)), 6)
        rows[3]['rubrics'] = rows[0]['rubrics']
        with self.assertRaisesRegex(ValueError, 'No valid bijection'):
            build_mapping(rows, labels)
        with self.assertRaises(ValueError):
            build_mapping(*fixture((1,)))

    def test_invalid_labels_rejected(self):
        rows, labels = fixture()
        for change in ('unknown', 'query', 'missing', 'duplicate'):
            altered = copy.deepcopy(labels)
            if change == 'unknown': altered[0]['primary_domain'] = 'unknown'
            if change == 'query': altered[0]['query_hash'] = 'wrong'
            if change == 'missing': altered.pop()
            if change == 'duplicate': altered.append(altered[0])
            with self.assertRaises(ValueError):
                build_mapping(rows, altered)

    def test_only_teacher_rubric_changes(self):
        rows, labels = fixture()
        original = copy.deepcopy(rows)
        by_id = {r['id']: r for r in rows}
        donors = {r['id']: r['donor_id'] for r in build_mapping(rows, labels)}
        row = rows[0]
        swapped = teacher_row(row, by_id, donors, 'same_domain_mismatch')
        self.assertEqual(messages(row), messages(swapped))
        self.assertEqual(swapped['query'], row['query'])
        self.assertEqual(swapped['docs'], row['docs'])
        self.assertEqual(swapped['rubrics'], by_id[donors[row['id']]]['rubrics'])
        self.assertEqual(rows, original)
        self.assertIs(teacher_row(row, by_id, donors, 'correct'), row)

    def test_protected_paths(self):
        protected = Path('/protected/baseline').resolve()
        for output in (protected, protected / 'step-300', protected.parent):
            with self.assertRaises(ValueError):
                disjoint_output(output, [protected])
        disjoint_output(protected.parent / 'new-run', [protected])

    def test_update_matches_frozen_baseline_except_teacher_and_audit(self):
        root = Path(__file__).resolve().parents[1]
        old = ast.parse((root / 'full_run/train.py').read_text(encoding='utf-8'))
        new = ast.parse((root / 'rubric_swap/train.py').read_text(encoding='utf-8'))
        def micro(tree):
            loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name) and n.target.id == 'index')
            loop = copy.deepcopy(loop)
            for node in ast.walk(loop):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tp' for t in node.targets):
                    node.value = ast.Constant('teacher-prompt')
                if isinstance(node, ast.Dict):
                    pairs = [(k, v) for k, v in zip(node.keys, node.values)
                             if not isinstance(k, ast.Constant) or k.value not in (
                                 'rubric_donor_id', 'rubric_assignment_type', 'rubric_assignment_domain')]
                    node.keys, node.values = [x[0] for x in pairs], [x[1] for x in pairs]
            return ast.dump(loop, include_attributes=False)
        self.assertEqual(micro(old), micro(new))


if __name__ == '__main__':
    unittest.main()
