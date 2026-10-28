import unittest
from .metrics import answer_hit, score
from .prepare import convert


class EvidenceMetricsTest(unittest.TestCase):
    def setUp(self):
        self.row = {'docs': [{'content': 'Alpha bridge'}, {'content': 'The city is Paris.'}, {'content': 'Distractor'}],
                    'gold_support_ids': [1, 2], 'answer_aliases': ['Paris']}

    def test_missing_bridge_can_still_hit_answer(self):
        value = score(self.row, '[2]')
        self.assertEqual(value['answer_hit_at_5'], 1)
        self.assertEqual(value['support_recall_at_5'], .5)
        self.assertEqual(value['all_annotated_supports_at_5'], 0)

    def test_invalid_is_not_repaired(self):
        for output in ('[1] [1] [2]', 'Explanation: [1] [2]', '[0] [1] [2]', '[]'):
            value = score(self.row, output)
            self.assertEqual(value['support_recall_at_5'], 0)
            self.assertEqual(value['valid'], 0)

    def test_complete_evidence_and_ndcg(self):
        value = score(self.row, '[1] [2]')
        self.assertEqual(value['all_annotated_supports_at_2'], 1)
        self.assertEqual(value['support_ndcg_at_5'], 1)
        self.assertEqual(value['support_f1_selected'], 1)

    def test_answer_boundary_and_yes_no(self):
        self.row['answer_aliases'] = ['Par']
        self.assertEqual(answer_hit(self.row, [2]), 0)
        self.row['answer_aliases'] = ['yes']
        self.assertIsNone(answer_hit(self.row, [2]))

    def test_hotpot_support_mapping_and_missing_sentence(self):
        row = {'id': 'example', 'question': 'Question?', 'answer': 'answer', 'type': 'bridge',
               'context': {'title': ['Distractor', 'Evidence'], 'sentences': [['Not useful.'], ['Support.']]},
               'supporting_facts': {'title': ['Evidence'], 'sent_id': [0]}}
        self.assertEqual(convert('hotpot', row)['gold_support_ids'], [2])
        row['supporting_facts']['sent_id'] = [1]
        with self.assertRaisesRegex(ValueError, 'missing_support_sentence'):
            convert('hotpot', row)

    def test_musique_uses_source_idx_not_array_position(self):
        row = {'id': '2hop__example', 'question': 'Question?', 'answer': 'answer', 'answer_aliases': [],
               'answerable': True, 'paragraphs': [
                   {'idx': 77, 'title': 'Evidence', 'paragraph_text': 'Support.', 'is_supporting': True},
                   {'idx': 55, 'title': 'Other', 'paragraph_text': 'Distractor.', 'is_supporting': False}],
               'question_decomposition': [{'paragraph_support_idx': 77}]}
        self.assertEqual(convert('musique', row)['gold_support_ids'], [1])
        row['question_decomposition'] = [{'paragraph_support_idx': 55}]
        with self.assertRaisesRegex(ValueError, 'support_decomposition_mismatch'):
            convert('musique', row)


if __name__ == '__main__':
    unittest.main()
