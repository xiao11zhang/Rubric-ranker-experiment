from collections import Counter
import copy
import unittest

from opsd.common import digest, messages
from .rubrics import build_mapping, validate_mapping, teacher_row, FALLBACK_LABEL
from .reporting import summarize_assignments, actual_assignment_report
from .test_rubrics import fixture


def mixed_fixture():
    rows, labels = fixture((4, 3))
    labels[0].update(primary_domain='unknown', uncertain=True, classification_status='needs_review')
    labels[-1].update(uncertain=True, classification_status='classification_error')
    return rows, labels


class MixedAssignmentsTest(unittest.TestCase):
    def test_fallback_labels_are_self_and_excluded_from_other_donors(self):
        rows, labels = mixed_fixture()
        original = copy.deepcopy(labels)
        result = build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)
        fallback = {labels[0]['id'], labels[-1]['id']}
        self.assertEqual(Counter(p['id'] for p in result), Counter(p['donor_id'] for p in result))
        for pair in result:
            if pair['id'] in fallback:
                self.assertEqual(pair['id'], pair['donor_id'])
                self.assertEqual(pair['primary_domain'], FALLBACK_LABEL)
                self.assertEqual(pair['assignment_type'], FALLBACK_LABEL)
            else:
                self.assertNotIn(pair['donor_id'], fallback)
                self.assertNotEqual(pair['id'], pair['donor_id'])
        self.assertEqual(labels, original)

    def test_student_and_fallback_teacher_inputs_unchanged(self):
        rows, labels = mixed_fixture()
        result = build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)
        donors = {p['id']: p['donor_id'] for p in result}
        by_id = {r['id']: r for r in rows}
        for row in rows:
            target = teacher_row(row, by_id, donors, 'same_domain_mismatch')
            self.assertEqual(messages(row), messages(target))
            if row['id'] == donors[row['id']]:
                self.assertIs(target, row)
                self.assertEqual(messages(row, teacher=True), messages(target, teacher=True))
            else:
                self.assertNotEqual(messages(row, teacher=True), messages(target, teacher=True))

    def test_fallback_not_a_solution_for_infeasible_clear_domains(self):
        rows, labels = fixture((2,))
        labels[0].update(primary_domain='unknown', uncertain=True)
        with self.assertRaisesRegex(ValueError, 'No valid bijection'):
            build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)

    def test_strict_policy_still_rejects_unknown_and_bad_labels(self):
        rows, labels = mixed_fixture()
        with self.assertRaises(ValueError):
            build_mapping(rows, labels)
        labels[0]['query_hash'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'query hash'):
            build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)

    def test_summary_uses_actual_prefix_not_full_dataset_fraction(self):
        rows, labels = mixed_fixture()
        result = build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)
        ids = [r['id'] for r in rows]
        full = summarize_assignments(result, ids)
        prefix = summarize_assignments(result, ids[:2])
        self.assertEqual(full['counts'][FALLBACK_LABEL], 2)
        self.assertEqual(prefix['ratios'][FALLBACK_LABEL], .5)
        self.assertNotEqual(full['ratios'][FALLBACK_LABEL], prefix['ratios'][FALLBACK_LABEL])

    def test_actual_report_checks_assignment_and_order(self):
        rows, labels = mixed_fixture()
        mapping = build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)
        assignments = {p['id']: p for p in mapping}
        ids = [r['id'] for r in rows]
        def sample(uid):
            p = assignments[uid]
            return {'id': uid, 'rubric_donor_id': p['donor_id'],
                    'rubric_assignment_type': p['assignment_type'], 'rubric_assignment_domain': p['primary_domain']}
        plan = {'config': {'gradient_accumulation': 2, 'pi_mode': 'same_domain_mismatch'},
                'order_ids': ids, 'rubric_mapping': mapping}
        entries = [{'step': 1, 'samples': [sample(uid) for uid in ids[:2]]}]
        report = actual_assignment_report(plan, entries, 1)
        self.assertEqual(report['records'], 2)
        self.assertEqual(report['counts'][FALLBACK_LABEL], 1)
        self.assertFalse(report['full_epoch_completed'])
        changed = copy.deepcopy(entries)
        changed[0]['samples'][0]['rubric_assignment_type'] = 'same_domain_mismatch'
        with self.assertRaisesRegex(ValueError, 'assignment log'):
            actual_assignment_report(plan, changed, 1)
        entries[0]['samples'].reverse()
        with self.assertRaisesRegex(ValueError, 'training prefix'):
            actual_assignment_report(plan, entries, 1)

    def test_validator_rejects_tampered_fallback_donor(self):
        rows, labels = mixed_fixture()
        result = build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)
        a = next(p for p in result if p['assignment_type'] == FALLBACK_LABEL)
        b = next(p for p in result if p['assignment_type'] != FALLBACK_LABEL)
        a['donor_id'], b['donor_id'] = b['donor_id'], a['donor_id']
        for p in (a, b):
            p['donor_rubric_hash'] = digest(next(r['rubrics'] for r in rows if r['id'] == p['donor_id']))
        with self.assertRaises(ValueError):
            validate_mapping(rows, labels, result, uncertain_policy=FALLBACK_LABEL)

    def test_all_fallback_is_legal_without_fake_cross_domain_pairs(self):
        rows, labels = fixture((3,))
        for label in labels:
            label.update(primary_domain='unknown', uncertain=True)
        result = build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)
        self.assertTrue(all(p['id'] == p['donor_id'] for p in result))
        self.assertEqual(summarize_assignments(result, [r['id'] for r in rows])['ratios'][FALLBACK_LABEL], 1)

    def test_4000_mapping_and_2000_actual_denominator(self):
        rows, labels = fixture((4000,))
        for label in labels[:617]:
            label.update(primary_domain='unknown', uncertain=True)
        mapping = build_mapping(rows, labels, uncertain_policy=FALLBACK_LABEL)
        ids = [r['id'] for r in rows]
        assignments = {m['id']: m for m in mapping}
        self.assertEqual(summarize_assignments(mapping, ids)['counts'],
                         {'same_domain_mismatch': 3383, 'original_fallback': 617, 'correct_original': 0})
        samples = [{'id': uid, 'rubric_donor_id': assignments[uid]['donor_id'],
                    'rubric_assignment_type': assignments[uid]['assignment_type'],
                    'rubric_assignment_domain': assignments[uid]['primary_domain']} for uid in ids[:2000]]
        entries = [{'step': i // 4 + 1, 'samples': samples[i:i + 4]} for i in range(0, 2000, 4)]
        plan = {'config': {'pi_mode': 'same_domain_mismatch', 'gradient_accumulation': 4},
                'order_ids': ids, 'rubric_mapping': mapping}
        actual = actual_assignment_report(plan, entries, 500)
        self.assertEqual(actual['records'], 2000)
        self.assertEqual(actual['ratios'][FALLBACK_LABEL], 617 / 2000)
        self.assertFalse(actual['full_epoch_completed'])
