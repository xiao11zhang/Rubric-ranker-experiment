"""Regression coverage for distinct entities sharing a context-scoped spelling."""
import unittest

from .entities import transform


class ScopedAliasesTest(unittest.TestCase):
    def test_title_and_character_keep_distinct_placeholders(self):
        source = {'description': 'In "Hamlet", Prince Hamlet appears in 1601; do not change 3 sources.'}
        entities = [
            {'type': 'work', 'mentions': ['Hamlet'], 'contexts': [
                {'path': ['description'], 'quote': '"Hamlet"'}]},
            {'type': 'person', 'mentions': ['Hamlet'], 'contexts': [
                {'path': ['description'], 'quote': 'Prince Hamlet'}]},
        ]
        actual, _, _ = transform(source, entities)
        self.assertEqual(actual['description'],
                         'In "<WORK_A>", Prince <PERSON_A> appears in 1601; do not change 3 sources.')

    def test_same_occurrence_cannot_have_two_entities(self):
        entities = [{'type': kind, 'mentions': ['Hamlet'], 'contexts': [
            {'path': ['description'], 'quote': 'Prince Hamlet'}]} for kind in ('work', 'person')]
        with self.assertRaisesRegex(ValueError, 'Overlapping entities'):
            transform({'description': 'Prince Hamlet appears.'}, entities)

    def test_unscoped_shared_alias_rejected_in_either_order(self):
        scoped = {'type': 'work', 'mentions': ['Hamlet'], 'contexts': [
            {'path': ['description'], 'quote': '"Hamlet"'}]}
        plain = {'type': 'person', 'mentions': ['Hamlet']}
        for entities in ([scoped, plain], [plain, scoped]):
            with self.assertRaisesRegex(ValueError, 'Repeated/ambiguous alias'):
                transform({'description': '"Hamlet" and Prince Hamlet.'}, entities)


if __name__ == '__main__':
    unittest.main()
