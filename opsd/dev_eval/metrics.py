"""Paragraph-support metrics; no answer generation or LLM judge."""
from collections import Counter
import math
import re
import unicodedata

from opsd.common import parse_output


def tokens(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def answer_hit(row, ids):
    aliases = [tokens(a) for a in row['answer_aliases']]
    aliases = [a for a in aliases if a and a not in (['yes'], ['no']) and len(''.join(a)) > 1]
    if not aliases:
        return None
    for i in ids:
        # Never concatenate documents, which could create a false match at a boundary.
        text = ' ' + ' '.join(tokens(row['docs'][i - 1]['content'])) + ' '
        if any(' ' + ' '.join(a) + ' ' in text for a in aliases):
            return 1.0
    return 0.0


def score(row, output):
    parsed = parse_output(output, len(row['docs']))
    ids = parsed['ids'] if parsed['valid'] else []
    gold = set(row['gold_support_ids'])
    hits = len(set(ids) & gold)
    precision = hits / len(ids) if ids else 0.0
    recall = hits / len(gold)
    result = {'valid': float(parsed['valid']), 'selected_count': len(ids), 'support_precision_selected': precision,
              'support_recall_selected': recall, 'support_f1_selected': 2 * precision * recall / (precision + recall) if hits else 0.0,
              'all_annotated_supports_selected': float(gold.issubset(ids))}
    for k in (2, 5, 10):
        prefix = ids[:k]
        result[f'support_recall_at_{k}'] = len(set(prefix) & gold) / len(gold)
        result[f'all_annotated_supports_at_{k}'] = float(gold.issubset(prefix))
        dcg = sum(1 / math.log2(j + 2) for j, i in enumerate(prefix) if i in gold)
        ideal = sum(1 / math.log2(j + 2) for j in range(min(k, len(gold))))
        result[f'support_ndcg_at_{k}'] = dcg / ideal
        result[f'answer_hit_at_{k}'] = answer_hit(row, prefix)
    return result


def bm25(row):
    """Fixed within-candidate-pool lexical baseline, k1=1.2, b=.75."""
    docs = [Counter(tokens(d['content'])) for d in row['docs']]
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / len(docs)
    query = set(tokens(row['query']))
    df = {term: sum(term in d for d in docs) for term in query}
    scores = []
    for d, length in zip(docs, lengths):
        total = 0.0
        for term in query:
            f = d[term]
            if f:
                idf = math.log(1 + (len(docs) - df[term] + .5) / (df[term] + .5))
                total += idf * f * 2.2 / (f + 1.2 * (.25 + .75 * length / avg))
        scores.append(total)
    return sorted(range(1, len(docs) + 1), key=lambda i: (-scores[i - 1], i))


def aggregate(records):
    result = {'queries': len(records)}
    for key in records[0]:
        values = [r[key] for r in records if r[key] is not None]
        result[key] = sum(values) / len(values) if values else None
        if key.startswith('answer_hit'):
            result[key + '_queries'] = len(values)
    return result
