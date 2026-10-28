"""Audit full official development splits and freeze a label-independent sample."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'opsd/dev_eval/data/dev-v1'
RAW = Path('/ai/yaodong/selector/data/closed_pool/raw')
SEED = 'rag-dev-v1-20261026'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def convert(name, r):
    if name == 'hotpot':
        titles, sentences = r['context']['title'], r['context']['sentences']
        if len(titles) != len(set(titles)) or len(titles) != len(sentences):
            raise ValueError('ambiguous_context_titles')
        if len(r['supporting_facts']['title']) != len(r['supporting_facts']['sent_id']):
            raise ValueError('misaligned_supporting_facts')
        for title, sent in zip(r['supporting_facts']['title'], r['supporting_facts']['sent_id']):
            if title not in titles or not 0 <= sent < len(sentences[titles.index(title)]):
                raise ValueError('missing_support_sentence')
        docs = [{'content': title + '\n' + ''.join(sents)} for title, sents in zip(titles, sentences)]
        gold = [i + 1 for i, title in enumerate(titles) if title in set(r['supporting_facts']['title'])]
        stratum = r['type']
        aliases = [r['answer']]
        source_ids = titles
    else:
        if not r['answerable']:
            raise ValueError('unanswerable')
        paragraphs = r['paragraphs']
        source_ids = [p['idx'] for p in paragraphs]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError('ambiguous_paragraph_ids')
        docs = [{'content': p['title'] + '\n' + p['paragraph_text']} for p in paragraphs]
        gold = [i + 1 for i, p in enumerate(paragraphs) if p['is_supporting']]
        decomposition = {d['paragraph_support_idx'] for d in r['question_decomposition']}
        if decomposition != {p['idx'] for p in paragraphs if p['is_supporting']}:
            raise ValueError('support_decomposition_mismatch')
        stratum = r['id'].split('__')[0]
        aliases = [r['answer']] + r['answer_aliases']
    if not gold or not docs or any(not d['content'].strip() for d in docs):
        raise ValueError('empty_evidence')
    from opsd.common import digest
    return {'id': name + ':validation:' + r['id'], 'source_id': r['id'], 'dataset': name,
            'split': 'validation', 'stratum': stratum, 'query': r['question'], 'docs': docs,
            'candidate_hash': digest(docs), 'source_candidate_ids': source_ids,
            'gold_support_ids': gold, 'answer_aliases': aliases}


def main():
    import pyarrow.parquet as pq
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [ROOT / 'opsd' / name for name in ('train.py', 'objective.py', 'common.py')]
    protected_paths += [ROOT / 'opsd/sft/configs/opsd_after_sft.toml']
    blocked = defaultdict(list)
    input_paths = [ROOT / 'opsd/data/rr_rag/train.jsonl', ROOT / 'opsd/data/rr_rag/val.jsonl',
                   ROOT / 'opsd/sft/data/rr_rag_v1/records.json']
    for path in input_paths:
        records = json.loads(path.read_text()) if path.suffix == '.json' else jsonl(path)
        for row in records:
            blocked[norm(row['query'])].append(str(path.relative_to(ROOT)) + ':' + row['id'])
    # Also conservatively exclude all official RAG SFT questions, including discarded labels.
    from opsd.sft.prepare import parse_rag, RAG_SYSTEM
    path = ROOT / 'opsd/sft/data/raw/rubricranker_sft-data.json'
    input_paths.append(path)
    for i, row in enumerate(json.loads(path.read_text())):
        if row['system'].startswith(RAG_SYSTEM):
            blocked[norm(parse_rag(row)['query'])].append('official_rag_sft:' + str(i))
    blocked_tokens = {q: set(q.split()) for q in blocked}
    inverted = defaultdict(set)
    for q, tokens in blocked_tokens.items():
        for token in tokens:
            inverted[token].add(q)
    exclusions, audits, selected, seen = [], {}, [], set()
    for name in ('hotpot', 'musique'):
        path = RAW / (name + '_validation.parquet')
        eligible, reasons = [], Counter()
        raw = pq.read_table(path).to_pylist()
        for r in raw:
            q = norm(r['question'])
            reason, matches = None, []
            if q in blocked:
                reason, matches = 'normalized_query_overlap', blocked[q]
            elif q in seen:
                reason = 'duplicate_development_query'
            else:
                tokens = set(q.split())
                possible = set().union(*(inverted[t] for t in tokens))
                for other in possible:
                    other_tokens = blocked_tokens[other]
                    if min(len(tokens), len(other_tokens)) >= 5 and len(tokens & other_tokens) / len(tokens | other_tokens) >= .8:
                        matches.extend(blocked[other])
                if matches:
                    reason = 'token_jaccard_at_least_0.8'
            if reason is None:
                try:
                    row = convert(name, r)
                except ValueError as exc:
                    reason = str(exc)
            if reason:
                reasons[reason] += 1
                exclusions.append({'dataset': name, 'source_id': r['id'], 'query': r['question'],
                                   'reason': reason, 'matches': matches})
                continue
            seen.add(q)
            eligible.append(row)
        # Freeze 100 per dataset by SHA256(question ID + fixed salt), before any predictions.
        chosen = sorted(eligible, key=lambda r: hashlib.sha256((SEED + r['id']).encode()).hexdigest())[:100]
        selected.extend(chosen)
        audits[name] = {'source_path': str(path), 'source_sha256': sha(path), 'raw_rows': len(raw),
                        'eligible': len(eligible), 'exclusions': dict(reasons), 'selected': len(chosen),
                        'eligible_strata': dict(Counter(r['stratum'] for r in eligible)),
                        'selected_strata': dict(Counter(r['stratum'] for r in chosen)),
                        'candidate_counts': dict(Counter(len(r['docs']) for r in chosen)),
                        'all_supports_present': True}
        (OUT / (name + '-eligible.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in eligible))
    if len(selected) != 200:
        raise ValueError('Expected 200 frozen development queries')
    target = OUT / 'queries.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
    (OUT / 'exclusions.json').write_text(json.dumps(exclusions, indent=2, ensure_ascii=False))
    protocol = {'version': 'dev-v1', 'purpose': 'development_only', 'query_count': len(selected),
        'selection': 'first 100 per dataset by SHA256(fixed_salt + source_id), no model/label-based selection',
        'selection_salt': SEED, 'queries_sha256': sha(target), 'datasets': audits,
        'decoding': 'greedy', 'max_new_tokens': 128, 'batch_size': 8,
        'prompt': 'unchanged opsd.common.encode_prompt; natural variable-length selection; no rubric/answers/evidence labels',
        'metric_k': [2, 5, 10], 'primary_metric': 'all_annotated_supports_at_5',
        'invalid_output_policy': 'strict parser; entire ranking empty; no deduplication, backfill or repair',
        'candidate_pool': 'official closed distractor/context pools; original order; not RR BGE open retrieval reproduction',
        'evidence': 'binary annotated paragraph supports; unannotated passages are not asserted universally irrelevant',
        'answer_hit': 'auxiliary normalized token-sequence exact alias match; yes/no and single-character aliases excluded',
        'independence_limit': 'normalized exact + lexical Jaccard >=0.8 exclusion; no semantic/pretraining contamination guarantee',
        'test_data_accessed': False, 'generator_used': False, 'teacher_used': False,
        'input_sha256': {str(p): sha(p) for p in input_paths},
        'protected_sha256': {str(p): sha(p) for p in protected_paths}}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    print(json.dumps(protocol, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
