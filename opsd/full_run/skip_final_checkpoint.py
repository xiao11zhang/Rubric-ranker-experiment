"""User-authorized cancellation of pending step-1000 full validation.

Leave the active step-300 and step-500 inference intact. Stop the old runner only
after the step-500 summary has committed, then finalize already completed results.
No original training/evaluation source or model file is changed.
"""
import json
import os
from pathlib import Path
import signal
import time

from opsd.dev_eval.prepare import ROOT, sha
from .utils import atomic_json, utc


def alive(pid):
    try:
        return '\nState:\tZ' not in Path(f'/proc/{pid}/status').read_text()
    except FileNotFoundError:
        return False


def main():
    root = ROOT / 'opsd/runs/rag-full-epoch-001'
    pipeline = ROOT / 'opsd/full_run/pipeline-status.json'
    current = json.loads(pipeline.read_text())
    child_pid = current['pid']
    supervisor_pid = int((ROOT / 'server-logs/rag-full-epoch-001-controller.pid').read_text())
    for pid, expected in ((child_pid, 'opsd.full_run.full_validation'), (supervisor_pid, 'opsd.full_run.pipeline')):
        if expected not in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode():
            raise ValueError('Unexpected process identity; refusing to signal')
    amendment = {'utc': utc(), 'reason': 'User explicitly cancelled step-1000 full validation',
                 'skip_steps': [1000], 'keep_steps': [0, 300, 500], 'controller_pid': os.getpid(),
                 'active_inference_pid': child_pid, 'weights_preserved': True}
    atomic_json(root / 'full-validation-amendment.json', amendment)
    original = root / 'full-validation-selection.json'
    with (root / 'full-validation-selection.original.json').open('x') as stream:
        stream.write(original.read_text())
    selection = json.loads(original.read_text())
    selection.update(steps=[0, 300, 500], cancelled_steps=[1000], amendment_utc=amendment['utc'])
    atomic_json(original, selection)
    os.kill(supervisor_pid, signal.SIGTERM)  # Preserve its running inference child.
    atomic_json(pipeline, {**current, 'utc': utc(), 'controller_pid': os.getpid(), 'cancelled_steps': [1000]})
    print(json.dumps(amendment), flush=True)
    target = root / 'full-validation/step-500/summary.json'
    while not target.exists():
        if not alive(child_pid):
            raise RuntimeError('Inference exited before step-500 finished')
        time.sleep(.1)
    if alive(child_pid):
        os.kill(child_pid, signal.SIGTERM)
    for _ in range(300):
        if not alive(child_pid):
            break
        time.sleep(.1)
    if alive(child_pid):
        raise RuntimeError('Old inference process did not exit')
    results = {}
    for step in (0, 300, 500):
        folder = root / 'full-validation' / f'step-{step}'
        result = json.loads((folder / 'summary.json').read_text())
        if len(result['query_ids']) != 9814 or sha(folder / 'predictions.jsonl') != result['prediction_sha256']:
            raise ValueError('Incomplete or changed predictions')
        results[str(step)] = result
    plan = json.loads((ROOT / 'opsd/full_run/plan.json').read_text())
    for path, expected in {**plan['original_protected_sha256'], **plan['sft_model_sha256']}.items():
        if sha(path) != expected:
            raise ValueError('Protected algorithm/model changed')
    atomic_json(root / 'full-validation-summary.json', {'evaluated_steps': [0, 300, 500],
        'best_small_step': 300, 'results': results, 'cancelled_steps': [1000], 'final_test_accessed': False,
        'note': 'Full development includes selection subset; user cancelled step-1000 full validation.'})
    lines = ['# RR RAG OPSD 一轮训练与开发验证', '',
        '训练已完成1000步，4000条RAG记录恰好遍历一次。用户随后取消step-1000全量验证，权重保留。',
        '全量开发验证仅比较SFT、step-300、step-500，每个9814题；其中包含用于选模型的200题，不是独立最终测试。', '',
        '| 范围 | Step | All-support@5 | Support Recall@5 | Support nDCG@5 | 集合精度 | 集合F1 | 平均选择数 | 格式合法率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('balanced_macro', 'pooled', 'hotpot', 'musique'):
        for step in (0, 300, 500):
            m = results[str(step)]['metrics'][group]
            lines.append(f"| {group} | {step} | {m['all_annotated_supports_at_5']:.2%} | {m['support_recall_at_5']:.2%} | {m['support_ndcg_at_5']:.2%} | {m['support_precision_selected']:.2%} | {m['support_f1_selected']:.2%} | {m['selected_count']:.2f} | {m['valid']:.2%} |")
    lines += ['', '原Student prompt、greedy、batch8、输出上限128；非法输出计空，不修复、不补齐。无Generator/Judge。',
              '指标基于人工支持段落，不代表穷尽相关性qrels。最终测试集未使用。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(root / 'integrity.json', {'original_algorithm_and_sft_weights_unchanged': True,
        'all_training_records_exactly_once': True, 'final_test_accessed': False})
    atomic_json(root / 'status.json', {'utc': utc(), 'stage': 'complete', 'cancelled_steps': [1000], 'report': str(root / 'REPORT.md')})
    atomic_json(pipeline, {'utc': utc(), 'stage': 'pipeline', 'state': 'complete', 'cancelled_steps': [1000]})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_json(ROOT / 'opsd/full_run/pipeline-status.json', {'utc': utc(), 'stage': 'validation_controller',
                    'state': 'failed', 'error': str(exc)})
        raise
