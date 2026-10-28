"""Read-only usage accounting across API runs and archived record attempts.

Totals are provider-reported, never inferred from text length or missing usage.
Transport retries without a saved response cannot be billed/accounted locally.
"""
import argparse
import hashlib
import json
from pathlib import Path

from opsd.dev_eval.prepare import ROOT
from opsd.full_run.utils import utc


def summarize(base, run_names=None):
    phases = {name: {'unique_responses': 0, 'responses_with_usage': 0,
                    'responses_without_usage': 0, 'prompt_tokens': None,
                    'completion_tokens': None, 'total_tokens': None,
                    'reasoning_tokens': None, 'cached_prompt_tokens': None}
              for name in ('extraction', 'review')}
    seen, returned_models, failed_attempts, diagnostics = set(), set(), 0, 0
    runs = []
    for run in sorted(Path(base).glob('run-*')):
        if run_names and run.name not in run_names:
            continue
        if not (run / 'run.json').is_file():
            continue
        runs.append(run.name)
        files = sorted((run / 'records').glob('*.json')) + sorted((run / 'attempts').glob('*.json'))
        for path in files:
            row = json.loads(path.read_text(encoding='utf-8'))
            failed_attempts += row.get('status') in ('error', 'needs_revision')
            for phase, key in (('extraction', 'extraction_response'), ('review', 'review_response')):
                response = row.get(key)
                if not response:
                    continue
                # A saved extraction may be reused after review transport failure.
                identity = response.get('response_id') or (
                    row['id'], phase, hashlib.sha256(json.dumps(response, sort_keys=True,
                        ensure_ascii=False).encode()).hexdigest())
                identity = str(identity)
                if identity in seen:
                    continue
                seen.add(identity)
                stats = phases[phase]
                stats['unique_responses'] += 1
                if response.get('returned_model'):
                    returned_models.add(response['returned_model'])
                usage = response.get('usage')
                if not isinstance(usage, dict) or not usage:
                    stats['responses_without_usage'] += 1
                    continue
                stats['responses_with_usage'] += 1
                for field, alternate in (('prompt_tokens', 'input_tokens'),
                                         ('completion_tokens', 'output_tokens'),
                                         ('total_tokens', 'total_tokens')):
                    value = usage.get(field, usage.get(alternate))
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        stats[field] = (stats[field] or 0) + value
                for field, details, key in (
                        ('reasoning_tokens', 'completion_tokens_details', 'reasoning_tokens'),
                        ('cached_prompt_tokens', 'prompt_tokens_details', 'cached_tokens')):
                    value = (usage.get(details) or {}).get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        stats[field] = (stats[field] or 0) + value
        diagnostics += (run / 'service-diagnostic.json').is_file()
    total = {}
    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens',
                  'reasoning_tokens', 'cached_prompt_tokens'):
        values = [phase[field] for phase in phases.values() if phase[field] is not None]
        total[field] = sum(values) if values else None
    return {'utc': utc(), 'runs': runs, 'phases': phases,
            'provider_reported_totals': total, 'returned_models': sorted(returned_models),
            'failed_or_rejected_record_attempts': failed_attempts,
            'separate_diagnostic_attempts': diagnostics,
            'billing_total_verified': False,
            'limitations': ['Missing usage is unknown, not zero.',
                'Provider totals do not establish billing for transport/server failures or hidden HTTP retries.',
                'Repeated saved responses are deduplicated by response ID, or record/phase/response hash.',
                'Successful responses from archived rejected attempts are included, not only approved rows.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'opsd/group05_deentity/data/agent/api')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='append', help='Select run name(s); default includes all history')
    args = parser.parse_args()
    result = summarize(args.base, args.run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
