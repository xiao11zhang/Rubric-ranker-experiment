"""Assignment proportions from planned IDs or verified ACTUAL training logs."""
from collections import Counter, defaultdict

from .rubrics import FALLBACK_LABEL

ASSIGNMENT_TYPES = ('same_domain_mismatch', FALLBACK_LABEL, 'correct_original')


def summarize_assignments(mapping, ids, mode='same_domain_mismatch'):
    if len(ids) != len(set(ids)):
        raise ValueError('Assignment summary must not double-count training IDs')
    by_id = {r['id']: r for r in mapping}
    counts, domains = Counter(), defaultdict(Counter)
    for uid in ids:
        if mode == 'correct':
            kind, domain = 'correct_original', 'not_applicable'
        else:
            assignment = by_id[uid]
            kind, domain = assignment['assignment_type'], assignment['primary_domain']
        if kind not in ASSIGNMENT_TYPES:
            raise ValueError('Unknown assignment type')
        counts[kind] += 1
        domains[domain][kind] += 1
    return {'records': len(ids), 'counts': {k: counts[k] for k in ASSIGNMENT_TYPES},
        'ratios': {k: counts[k] / len(ids) if ids else 0.0 for k in ASSIGNMENT_TYPES},
        'domain_counts': {d: dict(c) for d, c in sorted(domains.items())}}


def actual_assignment_report(plan, entries, step):
    samples = [s for row in entries for s in row['samples']]
    ids = [s['id'] for s in samples]
    n = step * plan['config']['gradient_accumulation']
    if ids != plan['order_ids'][:n] or len(ids) != n or [r['step'] for r in entries] != list(range(1, step + 1)):
        raise ValueError('Actual log does not match the exact training prefix')
    by_id = {r['id']: r for r in plan['rubric_mapping']}
    for sample in samples:
        assignment = by_id.get(sample['id'])
        donor = assignment['donor_id'] if assignment else sample['id']
        kind = assignment['assignment_type'] if assignment else 'correct_original'
        domain = assignment['primary_domain'] if assignment else 'not_applicable'
        if (sample['rubric_donor_id'] != donor or sample['rubric_assignment_type'] != kind or
                sample['rubric_assignment_domain'] != domain):
            raise ValueError('Actual assignment log differs from the frozen mapping')
    return {'source': 'verified_actual_training_log', 'completed_steps': step,
        'configured_training_records': len(plan['order_ids']),
        'full_epoch_completed': n == len(plan['order_ids']),
        **summarize_assignments(plan['rubric_mapping'], ids, plan['config']['pi_mode'])}
