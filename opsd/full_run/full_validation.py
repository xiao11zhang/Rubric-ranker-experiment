"""Post-training full DEVELOPMENT comparison; never opens a test split."""
import json
from pathlib import Path

from opsd.dev_eval.prepare import ROOT, sha, jsonl
from .prepare import PLAN
from .evaluation import evaluate
from .utils import atomic_json, select_best, utc


def main():
    plan = json.loads(PLAN.read_text())
    c = plan['config']
    root = ROOT / c['output']
    if not json.loads((root / 'coverage.json').read_text())['all_records_exactly_once']:
        raise ValueError('Training coverage not complete')
    for p, h in {**plan['input_and_execution_sha256'], **plan['original_protected_sha256'], **plan['sft_model_sha256']}.items():
        if sha(p) != h:
            raise ValueError('Frozen input/code/model changed: ' + p)
    history = json.loads((root / 'small-validation-history.json').read_text())
    if [r['step'] for r in history] != plan['small_eval_steps']:
        raise ValueError('Small development evaluations are incomplete')
    best = select_best(history)
    steps = sorted({0, 500, 1000, best['step']})
    atomic_json(root / 'full-validation-selection.json', {'selected_before_full_validation': True,
        'steps': steps, 'best_small_step': best['step'], 'best_rule': plan['best_rule'], 'final_test_accessed': False})
    rows = jsonl(plan['full_dev_path'])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
    results = {}
    for step in steps:
        if step:
            adapter = root / f'step-{step}'
            model_key = sha(adapter / 'adapter_model.safetensors')
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(model, str(adapter), adapter_name=f'step{step}', is_trainable=False)
            else:
                model.load_adapter(str(adapter), adapter_name=f'step{step}', is_trainable=False)
                model.set_adapter(f'step{step}')
        else:
            model_key = sha(Path(c['model']) / 'merge-manifest.json')
        def progress(done, total):
            value = {'utc': utc(), 'stage': 'full_validation', 'step': step, 'done': done, 'total': total}
            atomic_json(root / 'status.json', value)
            print(json.dumps(value), flush=True)
        results[str(step)] = evaluate(model, tokenizer, rows, root / 'full-validation' / f'step-{step}',
                                     model_key=model_key, progress=progress)
    atomic_json(root / 'full-validation-summary.json', {'best_small_step': best['step'], 'evaluated_steps': steps,
        'results': results, 'final_test_accessed': False,
        'note': 'Development set includes selection subset; not an independent final test result. No automatic retraining.'})
    lines = ['# RR RAG OPSD 一轮训练与开发验证', '',
        '从固定 SFT checkpoint-484 起点训练；4,000条RR RAG记录无放回遍历一次、1,000次优化更新。',
        '每100步在冻结200题validation上评测；最佳选择规则：balanced macro All-support@5，再集合F1，再较早step。',
        f"小验证集最佳step：{best['step']}。全量验证预先选定step：{steps}（0表示SFT）。",
        '全量validation为HotpotQA 7,404题与MuSiQue 2,410题。全量数据包含用于选择的200题，因此是开发诊断结果，不能称为独立最终测试。', '',
        '| 范围 | Step | All-support@5 | Support Recall@5 | Support nDCG@5 | 集合精度 | 集合F1 | 平均选择数 | 格式合法率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('balanced_macro', 'pooled', 'hotpot', 'musique'):
        for step in steps:
            m = results[str(step)]['metrics'][group]
            lines.append(f"| {group} | {step} | {m['all_annotated_supports_at_5']:.2%} | {m['support_recall_at_5']:.2%} | {m['support_ndcg_at_5']:.2%} | {m['support_precision_selected']:.2%} | {m['support_f1_selected']:.2%} | {m['selected_count']:.2f} | {m['valid']:.2%} |")
    lines += ['', '封闭候选池、原Student prompt、greedy、batch8、输出上限128；无Generator/Judge。非法输出整条计空，不修复、不补齐。',
        'support-nDCG按标注支持段落定义，非穷尽人工相关性qrels。最终测试集未读取或用于调参。',
        '完整每题预测、分组指标、证据精度/覆盖和答案命中代理保存在各full-validation/step-*目录。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    for p, h in {**plan['original_protected_sha256'], **plan['sft_model_sha256']}.items():
        if sha(p) != h:
            raise ValueError('Original algorithm or SFT weights changed')
    atomic_json(root / 'integrity.json', {'original_algorithm_and_sft_weights_unchanged': True,
        'all_training_records_exactly_once': True, 'final_test_accessed': False})
    atomic_json(root / 'status.json', {'utc': utc(), 'stage': 'complete', 'report': str(root / 'REPORT.md')})


if __name__ == '__main__':
    main()
