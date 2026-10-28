"""Freeze a new run against the existing full-epoch baseline, without training."""
import argparse
import json
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from opsd.common import digest, encode_prompt, read_rows
from opsd.dev_eval.prepare import ROOT, sha, jsonl
from .rubrics import build_mapping, teacher_row
from .reporting import summarize_assignments


def verify(plan):
    for name in ('original_protected_sha256', 'input_and_execution_sha256', 'sft_model_sha256'):
        for path, expected in plan[name].items():
            if sha(path) != expected:
                raise ValueError('Frozen file changed: ' + path)


def disjoint_output(output, protected):
    output = Path(output).resolve()
    for item in protected:
        item = Path(item).resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ValueError('Output overlaps protected baseline/model: ' + str(item))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.plan.exists():
        raise FileExistsError(args.plan)
    c = tomllib.loads(args.config.read_text(encoding='utf-8'))
    uncertain_policy = c.get('uncertain_policy', 'error')
    if uncertain_policy not in ('error', 'original_fallback'):
        raise ValueError('Unsupported uncertain-label policy')
    baseline_path = ROOT / c['baseline_plan']
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    verify(baseline)
    allowed = {'output', 'pi_mode'}
    for key, value in baseline['config'].items():
        if key not in allowed and c.get(key) != value:
            raise ValueError('Baseline training setting differs: ' + key)
    if c['pi_mode'] not in ('correct', 'same_domain_mismatch'):
        raise ValueError('Unsupported pi_mode')
    baseline_root = ROOT / baseline['config']['output']
    guard = baseline_root / 'step-300'
    if not (guard / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError('Baseline step-300 adapter must exist: ' + str(guard))
    root = ROOT / c['output']
    disjoint_output(root, [baseline_root, c['model']])
    if root.exists():
        raise FileExistsError(root)
    for destination in (args.plan.resolve(), args.plan.resolve().with_suffix('.mapping.json')):
        disjoint_output(destination, [baseline_root, c['model']])
        if destination.exists():
            raise FileExistsError(destination)
    rows = read_rows(ROOT / c['train_data'])
    if [rows[i]['id'] for i in baseline['order_indices']] != baseline['order_ids']:
        raise ValueError('Baseline training order/data changed')
    if c['steps'] * c['gradient_accumulation'] != len(rows):
        raise ValueError('Expected exactly one complete training epoch')
    stop_step = c.get('stop_after_step', c['steps'])
    if (not 0 < stop_step <= c['steps'] or stop_step not in baseline['checkpoint_steps'] or
            stop_step not in baseline['small_eval_steps']):
        raise ValueError('Stop step must be a saved and development-evaluated boundary')
    mapping = []
    labels_path = ROOT / c['domain_labels']
    if c['pi_mode'] == 'same_domain_mismatch':
        mapping = build_mapping(rows, jsonl(labels_path), c['rubric_seed'], uncertain_policy=uncertain_policy)
    all_assignments = summarize_assignments(mapping, baseline['order_ids'], c['pi_mode'])
    prefix_assignments = summarize_assignments(mapping,
        baseline['order_ids'][:stop_step * c['gradient_accumulation']], c['pi_mode'])
    for field, kind in (('expected_fallback_records', 'original_fallback'),
                        ('expected_mismatch_records', 'same_domain_mismatch')):
        if field in c and all_assignments['counts'][kind] != c[field]:
            raise ValueError('Assignment count differs from approved configuration: ' + field)
    donors = {r['id']: r['donor_id'] for r in mapping}
    by_id = {r['id']: r for r in rows}
    # Preflight changed teacher prompt lengths before any run is created.
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True, revision=c['revision'])
    model_config = AutoConfig.from_pretrained(c['model'], local_files_only=True, revision=c['revision'])
    maxima = {'student': 0, 'teacher': 0}
    for row in rows:
        target = teacher_row(row, by_id, donors, c['pi_mode'])
        lengths = {'student': len(encode_prompt(tokenizer, row, top_k=c['top_k'])),
                   'teacher': len(encode_prompt(tokenizer, target, teacher=True, mode='correct', top_k=c['top_k']))}
        if max(lengths.values()) > c['max_prompt_tokens'] or max(lengths.values()) + c['max_response_tokens'] > model_config.max_position_embeddings:
            raise ValueError('Prompt budget exceeded; no silent filtering/truncation: ' + str(row['id']))
        maxima = {k: max(maxima[k], lengths[k]) for k in maxima}
    mapping_path = args.plan.resolve().with_suffix('.mapping.json')
    files = [args.config.resolve(), baseline_path.resolve(), *Path(__file__).parent.glob('*.py'),
             *guard.rglob('*')]
    if mapping:
        files.append(labels_path.resolve())
    frozen = {str(p.resolve()): sha(p) for p in files if p.is_file()}
    plan = {**baseline, 'config': c, 'config_file_sha256': sha(args.config),
            'rubric_mapping': mapping, 'rubric_mapping_path': str(mapping_path),
            'rubric_mapping_sha256': digest(mapping), 'maximum_prompt_tokens': maxima,
            'baseline_step300': str(guard), 'baseline_plan': str(baseline_path),
            'uncertain_policy': uncertain_policy, 'execution_stop_after_step': stop_step,
            'assignment_summary_all': all_assignments,
            'assignment_summary_planned_prefix': prefix_assignments,
            'evaluation_policy': {'small_dev_steps': [s for s in baseline['small_eval_steps'] if s <= stop_step],
                'full_dev_fixed_steps': [0, 300, 500], 'additional_full_dev': 'best_small_development_checkpoint',
                'matched_baseline_steps': True, 'selection_rule': baseline['best_rule'],
                'full_dev_records': baseline['full_dev_records'], 'final_test_accessed': False},
            'input_and_execution_sha256': {**baseline['input_and_execution_sha256'], **frozen}}
    # Existing checkpoint module binds resumes to this whole experiment identity,
    # including rubric assignments, labels, code, SFT files and baseline config.
    plan['config_sha256'] = digest({k: v for k, v in plan.items() if k != 'config_sha256'})
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open('x', encoding='utf-8') as stream:
        json.dump(mapping, stream, ensure_ascii=False, indent=2)
    with args.plan.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'plan': str(args.plan), 'mode': c['pi_mode'], 'mapped_rows': len(mapping),
                      'training_started': False, 'maximum_prompt_tokens': maxima,
                      'stop_after_step': stop_step, 'all_assignments': all_assignments,
                      'planned_prefix_assignments': prefix_assignments}))


if __name__ == '__main__':
    main()
