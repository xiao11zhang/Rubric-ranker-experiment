"""Local agent spot checks and explicit corrections; never calls an API."""
from collections import Counter
import json
import math
import random
import re

from opsd.common import digest
from opsd.dev_eval.prepare import sha
from opsd.full_run.utils import atomic_json
from .entities import candidate, descriptions, validate_candidate

PROGRAM_METHOD = 'single_api_extraction_and_deterministic_program_validation'
AGENT_METHOD = 'agent_semantic_review_and_deterministic_program_validation'
POLICY = {'version': 'agent-risk-and-sample-v1', 'seed': 20261027,
          'fraction': 0.05, 'minimum': 20, 'pilot': 5, 'empty_stratum': 5,
          'per_entity_type': 2,
          'mandatory': ['numeric_work_titles', 'same_spelling_for_distinct_entities'],
          'no_per_record_api_review': True}


def required_ids(rows, records, imported):
    eligible = [r['id'] for r in rows if r['id'] not in imported]
    if not eligible:
        return []
    shuffled = list(eligible)
    random.Random(POLICY['seed']).shuffle(shuffled)
    n = min(len(eligible), max(POLICY['minimum'], math.ceil(len(eligible) * POLICY['fraction'])))
    required = set(shuffled[:n]) | set(eligible[:POLICY['pilot']])
    empty, by_type = [], {}
    for uid in eligible:
        record = records.get(uid, {})
        if record.get('status') != 'validated':
            continue
        entities = record['entities']
        if not entities:
            empty.append(uid)
        aliases = Counter(m for e in entities for m in e['mentions'])
        if any(count > 1 for count in aliases.values()) or any(
                e['type'] == 'work' and any(re.fullmatch(r'[\d\s.,:/+%\-–]+', m) for m in e['mentions']) for e in entities):
            required.add(uid)
        for kind in {e['type'] for e in entities}:
            by_type.setdefault(kind, []).append(uid)
    rank = lambda uid: digest([POLICY['seed'], uid])
    required.update(sorted(empty, key=rank)[:POLICY['empty_stratum']])
    for values in by_type.values():
        required.update(sorted(values, key=rank)[:POLICY['per_entity_type']])
    return [uid for uid in eligible if uid in required]


def valid_review(record, value):
    return (value.get('id') == record['id'] and value.get('rubric_hash') == record['candidate']['rubric_hash']
            and value.get('original_rubric_hash') == record['candidate']['original_rubric_hash']
            and value.get('decision') == 'approved' and value.get('review_method') == AGENT_METHOD
            and value.get('semantic_reviewed') is True and value.get('human_review_claimed') is False
            and isinstance(value.get('notes'), str) and bool(value['notes'].strip()))


def reviews(base, rows, records):
    result = {}
    for index, row in enumerate(rows):
        record = records.get(row['id'])
        path = base / 'audit/reviews' / f'{index + 1:04d}.json'
        if not path.is_file() or not record or record['status'] != 'validated':
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if valid_review(record, value):
            result[row['id']] = (value, path)
    return result


def create(base, rows, records, imported):
    required = required_ids(rows, records, imported)
    checked = reviews(base, rows, records)
    preview, decisions = [], []
    for row in rows:
        uid = row['id']
        record = records.get(uid)
        if not record or uid in imported:
            continue
        if record['status'] == 'validated' and (uid not in required or uid in checked):
            continue
        before = dict(descriptions(row['rubrics']))
        after = dict(descriptions(record['candidate']['rubrics'])) if record.get('candidate') else {}
        preview.append({'id': uid, 'status': record['status'], 'error': record.get('error'),
                        'entities': record.get('entities'), 'rubric_hash': record.get('candidate', {}).get('rubric_hash'),
                        'fields': [{'path': list(path), 'original': text, 'converted': after.get(path)} for path, text in before.items()]})
        if record['status'] == 'validated':
            decisions.append({'id': uid, 'rubric_hash': record['candidate']['rubric_hash'], 'decision': 'pending', 'notes': ''})
    plan = {'policy': POLICY, 'required_ids': required, 'already_agent_reviewed': list(checked),
            'missing_or_unreviewed_ids': [uid for uid in required if uid not in checked],
            'api_calls': 0, 'program_checks_are_not_semantic_review': True}
    atomic_json(base / 'audit/plan.json', plan)
    atomic_json(base / 'audit/preview.json', preview)
    # Do not overwrite a decision file being authored by the agent.
    template = base / 'audit/decisions.template.json'
    atomic_json(template, decisions)
    return {'required_reviews': len(required), 'completed_reviews': len(checked),
            'preview_records': len(preview), 'preview': str(base / 'audit/preview.json'), 'api_calls': 0}


def approve(base, rows, records, decisions):
    index = {r['id']: i for i, r in enumerate(rows)}
    if not isinstance(decisions, list) or len({d['id'] for d in decisions}) != len(decisions):
        raise ValueError('Audit decisions must be a list with distinct IDs')
    for decision in decisions:
        uid = decision['id']
        record = records.get(uid)
        if uid not in index or not record or record['status'] != 'validated':
            raise ValueError('Audit requires a program-validated API candidate')
        if (decision.get('decision') != 'approved' or decision.get('rubric_hash') != record['candidate']['rubric_hash']
                or not isinstance(decision.get('notes'), str) or not decision['notes'].strip()):
            raise ValueError('Explicit notes and matching hash required; no blanket automatic approval')
        value = {'id': uid, 'rubric_hash': record['candidate']['rubric_hash'],
                 'original_rubric_hash': record['candidate']['original_rubric_hash'],
                 'decision': 'approved', 'reviewer': 'Codex agent', 'review_method': AGENT_METHOD,
                 'semantic_reviewed': True, 'human_review_claimed': False, 'notes': decision['notes'][:600]}
        path = base / 'audit/reviews' / f'{index[uid] + 1:04d}.json'
        if path.exists():
            prior = json.loads(path.read_text(encoding='utf-8'))
            if prior != value:
                atomic_json(base / 'audit/history' / f'{index[uid] + 1:04d}-{sha(path)[:16]}.json', prior)
        atomic_json(path, value)
    return {'agent_reviews_recorded': len(decisions), 'api_calls': 0}


def repair(base, rows, records, fingerprint, corrections):
    by_id = {row['id']: (index, row) for index, row in enumerate(rows)}
    if not isinstance(corrections, list) or len({x['id'] for x in corrections}) != len(corrections):
        raise ValueError('Corrections require unique IDs')
    approvals = []
    for correction in corrections:
        uid = correction['id']
        if uid not in records or uid not in by_id or not isinstance(correction.get('notes'), str) or not correction['notes'].strip():
            raise ValueError('Correction requires existing API record and explicit agent semantic-review notes')
        index, row = by_id[uid]
        item = candidate(row, correction['entities'], fingerprint)
        validate_candidate(row, item)
        record = records[uid]
        path = base / 'records' / f'{index + 1:04d}.json'
        atomic_json(base / 'attempts' / f'{index + 1:04d}-{sha(path)[:16]}.json', record)
        record = {**record, 'status': 'validated', 'entities': correction['entities'], 'candidate': item,
                  'agent_corrected': True, 'agent_correction_notes': correction['notes'][:600]}
        atomic_json(path, record)
        records[uid] = record
        approvals.append({'id': uid, 'rubric_hash': item['rubric_hash'], 'decision': 'approved', 'notes': correction['notes']})
    approve(base, rows, records, approvals)
    return {'corrected_and_agent_reviewed': len(approvals), 'api_calls': 0}


def check(base, rows, records, imported):
    required = required_ids(rows, records, imported)
    checked = reviews(base, rows, records)
    # Agent corrections always require their own explicit semantic review.
    required = list(dict.fromkeys(required + [uid for uid, r in records.items() if r.get('agent_corrected')]))
    missing = [uid for uid in required if uid not in checked]
    if missing:
        raise ValueError(f'Agent risk/sample audit incomplete: {len(missing)} records; run --audit')
    return required, checked
