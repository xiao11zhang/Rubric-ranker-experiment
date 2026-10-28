"""Inference-only comparison of immutable SFT and existing OPSD adapters."""
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import time
import traceback

from opsd.common import encode_prompt
from .prepare import ROOT, OUT, sha, jsonl
from .metrics import score, aggregate, bm25

RESULTS = ROOT / 'opsd/dev_eval/results/dev-v1'


def report(rows, predictions, protocol):
    import numpy as np
    per_query = {name: [score(row, p['output']) for row, p in zip(rows, values)]
                 for name, values in predictions.items()}
    groups = {}
    for dataset in ('all', 'hotpot', 'musique'):
        indices = [i for i, r in enumerate(rows) if dataset == 'all' or r['dataset'] == dataset]
        groups[dataset] = {name: aggregate([values[i] for i in indices]) for name, values in per_query.items()}
    paired = {}
    for dataset in ('all', 'hotpot', 'musique'):
        indices = [i for i, r in enumerate(rows) if dataset == 'all' or r['dataset'] == dataset]
        paired[dataset] = {}
        # Stratified query bootstrap: resample within each dataset for pooled intervals.
        for checkpoint in ('step10', 'step20'):
            paired[dataset][checkpoint] = {}
            for metric in ('all_annotated_supports_at_5', 'support_recall_at_5', 'support_ndcg_at_5'):
                differences = np.array([per_query[checkpoint][i][metric] - per_query['sft'][i][metric] for i in indices])
                rng = np.random.default_rng(20261026)
                samples = []
                for _ in range(5000):
                    draw = []
                    for name in ('hotpot', 'musique'):
                        positions = [j for j, i in enumerate(indices) if rows[i]['dataset'] == name]
                        if positions:
                            draw.extend(rng.choice(positions, len(positions), replace=True))
                    samples.append(float(differences[draw].mean()))
                paired[dataset][checkpoint][metric] = {'delta': float(differences.mean()),
                    'paired_bootstrap_95ci': np.quantile(samples, [.025, .975]).tolist(),
                    'improved_queries': int((differences > 0).sum()), 'worsened_queries': int((differences < 0).sum())}
    summary = {'protocol': protocol, 'metrics': groups, 'paired_vs_sft': paired,
               'scope': '200 frozen development queries; closed candidate pools; no generator; no final test access',
               'strata': {name: dict(Counter(r['stratum'] for r in rows if r['dataset'] == name)) for name in ('hotpot', 'musique')}}
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2))
    (RESULTS / 'per-query-metrics.json').write_text(json.dumps({r['id']: {name: values[i] for name, values in per_query.items()}
                                                            for i, r in enumerate(rows)}, indent=2))
    lines = ['# 独立开发集：人工支持证据评测', '',
        '固定 SFT checkpoint-484 起点，仅评测已有 SFT / OPSD-10 / OPSD-20，无训练、Generator、Teacher/Judge。',
        '开发子集：HotpotQA distractor validation 100 条 + MuSiQue validation 100 条。不是全量开发集，也不是最终测试集。',
        '先完成全量开发数据重叠/证据映射审计，再按固定盐与原始 ID 的 SHA256 抽样；未根据模型结果选样。',
        '使用原始候选顺序和当前 Student prompt，自然输出可变长子集，贪心解码、最多128 tokens。', '',
        '主指标是 All-support@5：前至多5篇所选文档是否覆盖全部人工标注支持段落。无效格式整条计空，不补齐、不修复。',
        'nDCG 是按人工支持段落定义的二值 support-nDCG，不代表穷尽所有相关文档的标准 qrels。', '',
        '| 数据 | 模型 | N | All-support@5 | Support Recall@5 | Support nDCG@5 | 所选集合 P | 所选集合 R | 平均选择数 | 格式合法率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset, group in groups.items():
        for name, m in group.items():
            lines.append(f"| {dataset} | {name} | {m['queries']} | {m['all_annotated_supports_at_5']:.2%} | {m['support_recall_at_5']:.2%} | {m['support_ndcg_at_5']:.2%} | {m['support_precision_selected']:.2%} | {m['support_recall_selected']:.2%} | {m['selected_count']:.2f} | {m['valid']:.2%} |")
    lines += ['', '## 相对 SFT 的配对变化', '', '| 数据 | 模型 | All-support@5 变化（百分点） | 配对 bootstrap 95% 区间（百分点） | 改善 / 退化 |', '|---|---|---:|---|---|']
    for dataset, values in paired.items():
        for name, metrics in values.items():
            p = metrics['all_annotated_supports_at_5']
            lo, hi = p['paired_bootstrap_95ci']
            lines.append(f"| {dataset} | {name} | {100*p['delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {p['improved_queries']} / {p['worsened_queries']} |")
    lines += ['', '## 数据与边界', '',
        '- HotpotQA：supporting_facts 的 title/sent_id 先验证句子存在，再映射完整候选段落。MuSiQue：is_supporting 与 decomposition 的 paragraph_support_idx 必须完全一致。',
        '- 输入只包含 Query 与带标题正文；答案、别名、支持标签、分解问题和 Rubric 均不进入模型。',
        '- 原顺序与池内 BM25 为固定基线；它们返回全部候选，因此所选集合 P/R 不能与可变长模型输出作公平预算比较，优先比较相同@k。Oracle 按支持段落优先排列，仅用于显示候选池上限。',
        '- 答案字符串命中仅为辅助代理：按 Unicode 规范化后的完整 token 序列匹配，排除 yes/no 和单字符别名；其有效问题数与证据指标不同。完整@2/5/10、集合F1和答案命中见 summary.json。',
        '- 精确规范化与 token Jaccard>=0.8 排除了已知 SFT/RR训练/RR验证问题；不能证明无语义改写重叠或基础模型预训练污染。',
        '- 封闭候选池每题均含全部标注证据，不等同 RR 原论文 BGE 开放检索端到端复现；未标支持的文档不保证真正不相关。',
        '- 200题开发子集、单次贪心输出；置信区间只反映问题抽样差异，不反映训练随机性。无需也未据此改动算法或权重。',
        '- 最终测试集未读取、未推理、未参与选择模型或超参数。', '']
    (RESULTS / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    if RESULTS.exists():
        raise FileExistsError(RESULTS)
    RESULTS.mkdir(parents=True)
    def status(state, **fields):
        value = {'state': state, 'utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), **fields}
        (RESULTS / 'status.json').write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)
    try:
        status('preflight')
        protocol = json.loads((OUT / 'protocol.json').read_text())
        if sha(OUT / 'queries.jsonl') != protocol['queries_sha256']:
            raise ValueError('Frozen data changed')
        protected = protocol['protected_sha256']
        if any(sha(p) != h for p, h in protected.items()):
            raise ValueError('Original OPSD code/config changed')
        rows = jsonl(OUT / 'queries.jsonl')
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        model_path = ROOT / 'opsd/sft/models/rr-qwen3-4b-sft-r16-merged-001'
        run_path = ROOT / 'opsd/runs/post-sft-opsd-pilot-001'
        merge = json.loads((model_path / 'merge-manifest.json').read_text())
        if merge['sft_epochs'] != 1 or merge['sft_steps'] != 484:
            raise ValueError('SFT initialization changed')
        model_hashes = {str(model_path / p): h for p, h in merge['files_sha256'].items()}
        model_hashes.update({str(run_path / name / 'adapter_model.safetensors'): sha(run_path / name / 'adapter_model.safetensors') for name in ('step-10', 'step-20')})
        if any(sha(p) != h for p, h in model_hashes.items()):
            raise ValueError('Fixed model files do not match manifest')
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        prompts = [encode_prompt(tokenizer, row) for row in rows]
        if max(map(len, prompts)) > 12000:
            raise ValueError('Overlong prompt: no silent truncation allowed')
        (RESULTS / 'metadata.json').write_text(json.dumps({'protocol_sha256': sha(OUT / 'protocol.json'),
            'model_sha256': model_hashes, 'protected_sha256_before': protected,
            'prompt_tokens_min_max': [min(map(len, prompts)), max(map(len, prompts))],
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'generation': {'do_sample': False, 'max_new_tokens': 128, 'batch_size': 8},
            'training_performed': False, 'generator_used': False, 'final_test_accessed': False}, indent=2))
        base = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
        model = PeftModel.from_pretrained(base, str(run_path / 'step-10'), adapter_name='step10', is_trainable=False)
        model.load_adapter(str(run_path / 'step-20'), adapter_name='step20', is_trainable=False)
        model.eval()
        predictions = {}
        start_time = time.monotonic()
        for checkpoint in ('sft', 'step10', 'step20'):
            if checkpoint != 'sft':
                model.set_adapter(checkpoint)
            context = model.disable_adapter() if checkpoint == 'sft' else nullcontext()
            records = []
            with context, torch.inference_mode(), (RESULTS / (checkpoint + '.jsonl')).open('x', encoding='utf-8') as stream:
                for start in range(0, len(rows), 8):
                    group = prompts[start:start + 8]
                    width = max(map(len, group))
                    ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p for p in group], device='cuda:0')
                    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in group], device='cuda:0')
                    generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                        temperature=None, top_p=None, top_k=None, max_new_tokens=128,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
                    for row, sequence in zip(rows[start:start + 8], generated[:, width:].tolist()):
                        ended = tokenizer.eos_token_id in sequence
                        if ended:
                            sequence = sequence[:sequence.index(tokenizer.eos_token_id) + 1]
                        record = {'id': row['id'], 'candidate_hash': row['candidate_hash'],
                            'output': tokenizer.decode(sequence, skip_special_tokens=True),
                            'ended_eos': ended, 'response_tokens': len(sequence)}
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stream.flush()
                    status('inference', checkpoint=checkpoint, done=len(records), total=len(rows), seconds=time.monotonic() - start_time)
            predictions[checkpoint] = records
        for baseline in ('original_order', 'bm25_pool', 'oracle_support_first'):
            records = []
            for row in rows:
                order = list(range(1, len(row['docs']) + 1))
                if baseline == 'bm25_pool':
                    order = bm25(row)
                elif baseline == 'oracle_support_first':
                    gold = set(row['gold_support_ids'])
                    order = [i for i in order if i in gold] + [i for i in order if i not in gold]
                records.append({'id': row['id'], 'candidate_hash': row['candidate_hash'], 'output': ' '.join(f'[{i}]' for i in order)})
            predictions[baseline] = records
            (RESULTS / (baseline + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
        status('scoring')
        report(rows, predictions, protocol)
        if any(sha(p) != h for p, h in {**protected, **model_hashes}.items()):
            raise ValueError('Protected training code/config/model files changed during inference')
        (RESULTS / 'integrity.json').write_text(json.dumps({'protected_code_config_models_unchanged': True,
            'final_test_accessed': False, 'training_performed': False}, indent=2))
        status('complete', report=str(RESULTS / 'REPORT.md'))
    except Exception as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
