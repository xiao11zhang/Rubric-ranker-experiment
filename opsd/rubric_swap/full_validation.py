"""Explicit post-training full DEVELOPMENT evaluation; original baseline stays read-only."""
import argparse
import json
from pathlib import Path

from opsd.common import digest
from opsd.dev_eval.prepare import ROOT, sha, jsonl
from opsd.full_run.evaluation import evaluate
from opsd.full_run.utils import atomic_json, select_best, utc
from .prepare import verify
from .reporting import actual_assignment_report


def comparison_selection(plan, new_history, correct_history):
    limit = plan['execution_stop_after_step']
    expected = plan['evaluation_policy']['small_dev_steps']
    if [r['step'] for r in new_history] != expected:
        raise ValueError('New-group development history is incomplete')
    correct = [r for r in correct_history if r['step'] <= limit]
    if [r['step'] for r in correct] != expected:
        raise ValueError('Baseline history does not cover the same optimization budget')
    new_best, correct_best = select_best(new_history)['step'], select_best(correct)['step']
    selected = sorted(set(plan['evaluation_policy']['full_dev_fixed_steps']) | {new_best, correct_best})
    if any(step > limit for step in selected):
        raise ValueError('Full development selection exceeds the approved training budget')
    # Both groups at the same selected steps; step-0 is the shared SFT model.
    return {'new_best_small_step': new_best, 'correct_best_small_step': correct_best,
            'matched_steps': selected, 'selected_before_full_development': True,
            'final_test_accessed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if digest({k: v for k, v in plan.items() if k != 'config_sha256'}) != plan['config_sha256']:
        raise ValueError('Experiment plan identity changed')
    verify(plan)
    root = ROOT / plan['config']['output']
    baseline_plan = json.loads(Path(plan['baseline_plan']).read_text())
    baseline_root = ROOT / baseline_plan['config']['output']
    selection = comparison_selection(plan, json.loads((root / 'small-validation-history.json').read_text()),
        json.loads((baseline_root / 'small-validation-history.json').read_text()))
    actual = actual_assignment_report(plan, jsonl(root / 'train.jsonl'), plan['execution_stop_after_step'])
    if actual['records'] != 2000 or not (root / 'step-500/checkpoint.json').is_file():
        raise ValueError('Expected completed 2000-record training and saved step-500')
    if not args.execute:
        print(json.dumps({'status': 'evaluation_plan_verified', 'selection': selection,
                          'actual_assignments': actual, 'inference_started': False}))
        return
    selection_path = root / 'full-validation-selection.json'
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError('Full development selection is already frozen differently')
    else:
        atomic_json(selection_path, selection)
    rows = jsonl(plan['full_dev_path'])
    if len(rows) != 9814 or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Expected original 9814 development records only')
    model, tokenizer = None, None
    results = {'correct': {}, 'mixed': {}}
    for group in ('correct', 'mixed'):
        for step in selection['matched_steps']:
            if step == 0:
                checkpoint = None
                key = sha(Path(plan['config']['model']) / 'merge-manifest.json')
                source = baseline_root / 'full-validation/step-0'
            else:
                checkpoint = (baseline_root if group == 'correct' else root) / f'step-{step}'
                key = sha(checkpoint / 'adapter_model.safetensors')
                source = (baseline_root / 'full-validation' / f'step-{step}') if group == 'correct' else None
            if source is not None and (source / 'summary.json').is_file():
                cached = json.loads((source / 'summary.json').read_text())
                if (cached['model_key'] != key or cached['query_ids'] != [r['id'] for r in rows] or
                        sha(source / 'predictions.jsonl') != cached['prediction_sha256']):
                    raise ValueError('Baseline cached validation differs from frozen model/data')
                results[group][str(step)] = cached
                continue
            # Any additional baseline evaluation writes ONLY under the new-group output.
            destination = root / 'full-validation' / f'{group}-step-{step}'
            if model is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(plan['config']['model'], local_files_only=True)
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                model = AutoModelForCausalLM.from_pretrained(plan['config']['model'], local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
            from peft import PeftModel
            if checkpoint is not None:
                name = f'{group}_step{step}'
                if not isinstance(model, PeftModel):
                    model = PeftModel.from_pretrained(model, str(checkpoint), adapter_name=name, is_trainable=False)
                else:
                    model.load_adapter(str(checkpoint), adapter_name=name, is_trainable=False)
                    model.set_adapter(name)
            def progress(done, total):
                atomic_json(root / 'full-validation-status.json', {'utc': utc(), 'stage': 'full_development',
                    'group': group, 'step': step, 'done': done, 'total': total})
            # SFT is always reused above; never accidentally evaluate it with an active adapter.
            if checkpoint is None:
                raise ValueError('Shared SFT cache is missing; baseline must remain read-only')
            results[group][str(step)] = evaluate(model, tokenizer, rows, destination, model_key=key, progress=progress)
    verify(plan)
    summary = {'selection': selection, 'actual_training_assignments': actual, 'results': results,
               'scope': '9814 development queries including dev200; not an independent final test',
               'final_test_accessed': False, 'baseline_modified': False}
    atomic_json(root / 'full-validation-summary.json', summary)
    lines = ['# 自动领域部分同领域 Rubric 错配：全量开发集对照', '',
        '共用同一SFT起点、训练顺序、原OPSD损失。比较前500次更新；不加载正确组adapter作为错配训练起点。',
        f"实际处理{actual['records']}条；错配{actual['counts']['same_domain_mismatch']}条"
        f"（{actual['ratios']['same_domain_mismatch']:.2%}），原Rubric回退{actual['counts']['original_fallback']}条"
        f"（{actual['ratios']['original_fallback']:.2%}）。", 
        f"按同一dev200规则，正确组最佳step={selection['correct_best_small_step']}，混合组最佳step={selection['new_best_small_step']}。",
        '全量9814题包含用于选点的dev200；仍为开发评估，最终测试集未使用。', '',
        '| 范围 | 组 | Step | All-support@5 | Support Recall@5 | Support nDCG@5 | 集合精度 | 集合F1 | 平均选择数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('balanced_macro', 'pooled', 'hotpot', 'musique'):
        for group in ('correct', 'mixed'):
            for step in selection['matched_steps']:
                m = results[group][str(step)]['metrics'][scope]
                lines.append(f"| {scope} | {group} | {step} | {m['all_annotated_supports_at_5']:.2%} | {m['support_recall_at_5']:.2%} | {m['support_ndcg_at_5']:.2%} | {m['support_precision_selected']:.2%} | {m['support_f1_selected']:.2%} | {m['selected_count']:.2f} |")
    lines += ['', '本组使用自动领域标签，不能称为人工审核通过。617条保留原Rubric；其余同领域错配仍可能受分类错误影响。',
              '整份Rubric替换同时改变条目、权重和长度，不能宣称严格控制结构和长度。',
              '固定一对一映射构造于4000条训练记录；仅使用前2000条时，两组实际Rubric边际分布不保证相同。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(root / 'full-validation-status.json', {'utc': utc(), 'stage': 'complete', 'report': str(root / 'REPORT.md')})


if __name__ == '__main__':
    main()
