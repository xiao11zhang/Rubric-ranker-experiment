"""Evaluate preselected Group 05 checkpoints on the same full DEVELOPMENT data."""
import argparse
import json
from pathlib import Path

from opsd.dev_eval.prepare import ROOT, jsonl, sha
from opsd.full_run.evaluation import evaluate
from opsd.full_run.utils import atomic_json, select_best, utc
from .prepare import runtime
from .reporting import actual_report


def select_steps(plan, history):
    if [r['step'] for r in history] != plan['small_eval_steps']:
        raise ValueError('Expected complete dev200 history through step 500')
    return sorted({0, 300, 500, select_best(history)['step']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('opsd/group05_deentity/plans/deentity-001.json'))
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    plan = json.loads(a.plan.read_text(encoding='utf-8'))
    _, derived = runtime(plan)
    c = plan['config']
    root = ROOT / c['output']
    actual_report(plan, jsonl(root / 'train.jsonl'), derived)
    coverage = json.loads((root / 'coverage.json').read_text(encoding='utf-8'))
    if not coverage['matches_baseline_prefix'] or coverage['steps'] != 500:
        raise ValueError('Training did not complete the planned 500 steps')
    history = json.loads((root / 'small-validation-history.json').read_text(encoding='utf-8'))
    steps = select_steps(plan, history)
    selection = {'selected_before_full_validation': True, 'steps': steps,
                 'best_small_step': select_best(history)['step'], 'best_rule': plan['best_rule'],
                 'final_test_accessed': False}
    selection_path = root / 'full-validation-selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text(encoding='utf-8')) != selection:
        raise ValueError('Full development checkpoint selection changed')
    if not a.execute:
        print(json.dumps({**selection, 'evaluation_started': False}, indent=2))
        return
    atomic_json(selection_path, selection)
    rows = jsonl(plan['full_dev_path'])
    if len(rows) != plan['full_dev_records'] or len(rows) != 9814:
        raise ValueError('Full development data differs from baseline')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True, revision=c['revision'])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = None
    results = {}
    baseline_root = Path(plan['baseline_root'])
    for step in steps:
        key = sha(root / f'step-{step}/adapter_model.safetensors') if step else sha(Path(c['model']) / 'merge-manifest.json')
        cached_sft = baseline_root / 'full-validation/step-0/summary.json'
        if step == 0 and cached_sft.is_file():
            saved = json.loads(cached_sft.read_text(encoding='utf-8'))
            if saved['model_key'] != key or saved['query_ids'] != [r['id'] for r in rows]:
                raise ValueError('Shared baseline SFT evaluation differs')
            results['0'] = saved
            atomic_json(root / 'reports/shared-sft-reference.json', {'path': str(cached_sft), 'sha256': sha(cached_sft), 'model_key': key})
            continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(c['model'], revision=c['revision'], local_files_only=True,
                torch_dtype=getattr(torch, c['dtype']), attn_implementation='sdpa').to(c['student_device'])
        if step:
            name = f'step{step}'
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(model, str(root / f'step-{step}'), adapter_name=name, is_trainable=False)
            else:
                model.load_adapter(str(root / f'step-{step}'), adapter_name=name, is_trainable=False)
                model.set_adapter(name)
        def progress(done, total):
            atomic_json(root / 'status.json', {'utc': utc(), 'stage': 'full_development', 'step': step, 'done': done, 'total': total})
        results[str(step)] = evaluate(model, tokenizer, rows, root / 'full-validation' / f'step-{step}', model_key=key, progress=progress)
    atomic_json(root / 'full-validation-summary.json', {**selection, 'evaluated_steps': steps, 'results': results,
                'note': '9814 development questions include dev200; not an independent final test.'})
    lines = ['# 第五组全量开发集评测', '',
             '9814题包含用于选点的dev200；最终测试未使用。SFT结果在模型及题目ID一致时复用基线。', '',
             '| 模型 | Step | All-support@5（balanced macro） | 集合F1 |', '|---|---:|---:|---:|']
    baseline_summary = baseline_root / 'full-validation-summary.json'
    reference = json.loads(baseline_summary.read_text(encoding='utf-8'))['results'] if baseline_summary.exists() else {}
    for step in steps:
        for name, result in (('正确Rubric基线', reference.get(str(step))), ('第五组去实体', results[str(step)])):
            if result is None:
                continue
            expected_key = sha(baseline_root / f'step-{step}/adapter_model.safetensors') if step else sha(Path(c['model']) / 'merge-manifest.json')
            if name == '正确Rubric基线' and (result['model_key'] != expected_key or result['query_ids'] != [r['id'] for r in rows]):
                raise ValueError('Baseline comparison model/data mismatch')
            m = result['metrics']['balanced_macro']
            lines.append(f"| {name} | {step} | {m['all_annotated_supports_at_5']:.2%} | {m['support_f1_selected']:.2%} |")
    missing = [s for s in steps if str(s) not in reference]
    if missing:
        lines += ['', f'正确组尚缺这些相同步数的全量开发结果：{missing}；未补齐前不能宣称该点的同预算差异。']
    reports = root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    (reports / 'FULL_DEVELOPMENT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    from .protocol import verify
    verify(plan)
    atomic_json(root / 'status.json', {'utc': utc(), 'stage': 'complete', 'report': str(reports / 'FULL_DEVELOPMENT.md')})


if __name__ == '__main__':
    main()
