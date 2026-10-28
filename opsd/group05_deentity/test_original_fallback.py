import unittest
from .original_fallback import identity
from .entities import validate_candidate


class OriginalFallbackTest(unittest.TestCase):
    def test_entire_rubric_preserved_and_not_no_entity_claim(self):
        row = {'id':'synthetic', 'query':'q', 'rubrics':{'items':[{'description':'Alice visited Paris in 1990, not 2000.', 'weight':3}]}}
        item = identity(row, 'test', 'Ambiguous individual versus work title')
        self.assertEqual(item['rubrics'], row['rubrics'])
        self.assertEqual(item['changes'], [])
        self.assertFalse(item['deentity_success'])
        self.assertEqual(item['processing_mode'], 'original_fallback')
        validate_candidate(row, item)

    def test_reason_required(self):
        with self.assertRaises(ValueError):
            identity({'id':'x'}, 'test', '')


if __name__ == '__main__':
    unittest.main()
