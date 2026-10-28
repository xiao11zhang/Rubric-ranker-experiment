"""Offline, fixed, one-to-one rubric assignment; no global RNG consumption."""
from collections import defaultdict
import random

from opsd.common import digest

DOMAINS = ('entertainment', 'music_performance', 'sports', 'history_military',
           'politics_law_governance', 'geography_buildings_transport',
           'natural_science_ecology', 'medicine_health', 'computing_engineering',
           'business_economics_finance', 'literature_language_art',
           'religion_mythology_philosophy', 'education_academia')
FALLBACK_LABEL = 'original_fallback'


def assignment_labels(rows, labels, uncertain_policy='error'):
    """Derive execution labels without rewriting the classifier's raw output."""
    if uncertain_policy not in ('error', FALLBACK_LABEL):
        raise ValueError('Unsupported uncertain-label policy')
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows) or len({x['id'] for x in labels}) != len(labels):
        raise ValueError('Duplicate IDs')
    if set(by_id) != {x['id'] for x in labels}:
        raise ValueError('Labels must cover exactly the training IDs')
    result = {}
    for label in labels:
        row = by_id[label['id']]
        if label['query_hash'] != digest(row['query']):
            raise ValueError('Label query hash mismatch: ' + str(row['id']))
        domain = label['primary_domain']
        if domain not in (*DOMAINS, 'unknown') or type(label.get('uncertain', False)) is not bool:
            raise ValueError('Invalid domain/uncertainty: ' + str(row['id']))
        uncertain = domain == 'unknown' or label.get('uncertain', False) or label.get('classification_status') == 'classification_error'
        if uncertain and uncertain_policy == 'error':
            raise ValueError('Unreviewed/unknown domain: ' + str(row['id']))
        result[row['id']] = {'primary_domain': FALLBACK_LABEL if uncertain else domain,
            'assignment_type': FALLBACK_LABEL if uncertain else 'same_domain_mismatch',
            'source_primary_domain': domain, 'source_uncertain': label.get('uncertain', False),
            'label_acceptance': 'original_rubric_fallback' if uncertain else 'automatic_without_manual_review',
            'fallback_reason': ('classification_error' if label.get('classification_status') == 'classification_error'
                                else 'unknown_or_uncertain') if uncertain else None}
    return result


def build_mapping(rows, labels, seed=20261027, *, uncertain_policy='error'):
    by_id = {r['id']: r for r in rows}
    assignments = assignment_labels(rows, labels, uncertain_policy)
    groups = defaultdict(list)
    result = []
    for row in rows:
        assignment = assignments[row['id']]
        if assignment['assignment_type'] == FALLBACK_LABEL:
            h = digest(row['rubrics'])
            result.append({'id': row['id'], 'donor_id': row['id'], **assignment,
                           'recipient_rubric_hash': h, 'donor_rubric_hash': h})
        else:
            groups[assignment['primary_domain']].append(row)
    rng = random.Random(seed)
    for domain in sorted(groups):
        buckets = defaultdict(list)
        for row in groups[domain]:
            buckets[digest(row['rubrics'])].append(row['id'])
        blocks = list(buckets.values())
        rng.shuffle(blocks)
        for block in blocks:
            block.sort(key=str)
            rng.shuffle(block)
        shift = max(map(len, blocks))
        ordered = [i for block in blocks for i in block]
        if 2 * shift > len(ordered):
            raise ValueError(f'No valid bijection in {domain}: n={len(ordered)}, largest identical-rubric group={shift}')
        donors = ordered[shift:] + ordered[:shift]
        for recipient, donor in zip(ordered, donors):
            assert recipient != donor
            assert digest(by_id[recipient]['rubrics']) != digest(by_id[donor]['rubrics'])
            result.append({'id': recipient, 'donor_id': donor, **assignments[recipient],
                           'recipient_rubric_hash': digest(by_id[recipient]['rubrics']),
                           'donor_rubric_hash': digest(by_id[donor]['rubrics'])})
    result = sorted(result, key=lambda x: str(x['id']))
    validate_mapping(rows, labels, result, uncertain_policy=uncertain_policy)
    return result


def validate_mapping(rows, labels, mapping, *, uncertain_policy='error'):
    by_id = {r['id']: r for r in rows}
    assignments = assignment_labels(rows, labels, uncertain_policy)
    if len(mapping) != len(rows) or {m['id'] for m in mapping} != set(by_id):
        raise ValueError('Mapping must cover every training row once')
    if len({m['donor_id'] for m in mapping}) != len(rows) or {m['donor_id'] for m in mapping} != set(by_id):
        raise ValueError('Donors must be a bijection over the training IDs')
    for pair in mapping:
        row, donor = by_id[pair['id']], by_id[pair['donor_id']]
        assignment = assignments[row['id']]
        if any(pair.get(k) != v for k, v in assignment.items()):
            raise ValueError('Assignment metadata differs from source labels')
        if pair['recipient_rubric_hash'] != digest(row['rubrics']) or pair['donor_rubric_hash'] != digest(donor['rubrics']):
            raise ValueError('Mapping rubric hash mismatch')
        if assignment['assignment_type'] == FALLBACK_LABEL:
            if row['id'] != donor['id']:
                raise ValueError('Fallback record must retain its own rubric')
        elif (row['id'] == donor['id'] or digest(row['query']) == digest(donor['query']) or
              digest(row['rubrics']) == digest(donor['rubrics']) or
              assignments[donor['id']]['assignment_type'] == FALLBACK_LABEL or
              assignment['primary_domain'] != assignments[donor['id']]['primary_domain']):
            raise ValueError('Invalid same-domain mismatch')


def teacher_row(row, by_id, mapping, mode):
    if mode == 'correct':
        return row
    if mode != 'same_domain_mismatch':
        raise ValueError('Unsupported rubric mode')
    donor = by_id[mapping[row['id']]]
    if donor['id'] == row['id']:
        return row
    return {**row, 'rubrics': donor['rubrics'], 'rubric_hash': digest(donor['rubrics'])}
