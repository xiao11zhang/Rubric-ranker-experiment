"""Cancel only the old supervisor; preserve active training, then report dev200 only."""
import json
import os
from pathlib import Path
import signal
import time

from opsd.dev_eval.prepare import ROOT
from opsd.full_run.utils import atomic_json, select_best, utc
from opsd.rubric_swap.prepare import verify
from opsd.rubric_swap.reporting import actual_assignment_report

RUN = ROOT / 'opsd/runs/rag-rubric-same-domain-001'
STATUS = ROOT / 'opsd/rubric_swap/pipeline-status.json'
PLAN = ROOT / 'opsd/rubric_swap/plans/same-domain-001.json'


def alive(pid):
    try:
        return '\nState:\tZ' not in Path(f'/proc/{pid}/status').read_text()
    except FileNotFoundError:
        return False


def write_report(plan, *, completed):
    entries = []
    for line in (RUN / 'train.jsonl').read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            if completed:
                raise
            break
    step = entries[-1]['step'] if entries else 0
    assignment = actual_assignment_report(plan, entries, step)
    new_history = json.loads((RUN / 'small-validation-history.json').read_text())
    baseline = ROOT / 'opsd/runs/rag-full-epoch-001'
    correct_history = [r for r in json.loads((baseline / 'small-validation-history.json').read_text()) if r['step'] <= 500]
    correct_by_step = {r['step']: r for r in correct_history}
    best_new, best_correct = select_best(new_history), select_best(correct_history)
    if completed and (step != 500 or [r['step'] for r in new_history] != [0, 100, 200, 300, 400, 500]):
        raise ValueError('Final training/development history is not complete through step-500')
    # This controller never launches inference. A pre-existing full-stage log would
    # invalidate the requested claim and must be surfaced, not silently ignored.
    if (ROOT / 'server-logs/rubric-mixed-001-full_development.log').exists():
        raise ValueError('Full development stage log exists; cannot claim it never started')
    full_predictions = list((RUN / 'full-validation').glob('**/predictions.jsonl'))
    if full_predictions:
        raise ValueError('Full development predictions exist; cannot claim no full evaluation')
    report = {'utc': utc(), 'completed': completed, 'training_steps': step,
        'actual_training_assignments': assignment, 'dev200_history': new_history,
        'correct_dev200_history_through_500': correct_history,
        'best_mixed_dev200_step': best_new['step'], 'best_correct_dev200_step': best_correct['step'],
        'selection_rule': plan['evaluation_policy']['selection_rule'],
        'full_development_evaluated': False, 'full_development_queries_evaluated': 0,
        'full_development_skip_reason': 'User requested reporting after current training/dev200 only',
        'final_test_evaluated': False, 'baseline_modified': False}
    atomic_json(RUN / 'training-dev200-summary.json', report)
    if completed:
        stored = json.loads((RUN / 'rubric-assignment-summary.json').read_text())
        if stored != assignment:
            raise ValueError('Trainer assignment summary differs from verified actual log')
    lines = ['# 第二组部分同领域 Rubric 错配：训练与 dev200 报告', '',
        '**本组未进行 9,814 条全量开发评估；以下结果仅来自固定 200 题验证集。独立最终测试集也未使用。**', '',
        ('状态：训练、step-500 checkpoint保存及该步dev200验证均已完成。' if completed else
         f'状态：训练进行中，目前已记录{step}/500步；本报告为阶段性记录，结束后自动更新。'),
        '用户已取消后续自动全量开发评估，不再进入该阶段。', '',
        '## 实验条件与实际覆盖', '',
        '- Teacher和Student均从正确组相同的一轮SFT合并模型初始化；Teacher冻结，Student新建LoRA r=16。',
        '- 采用自动领域标签：全量3,383条同领域一对一错配，617条统一标记original_fallback，保留原Rubric；不称人工审核通过。',
        '- 原Query、候选与Student输入不变，原OPSD损失不变。与正确组使用相同seed及训练记录排列。',
        f"- 实际已处理{assignment['records']}条：错配{assignment['counts']['same_domain_mismatch']}条"
        f"（{assignment['ratios']['same_domain_mismatch']:.2%}），保留原Rubric{assignment['counts']['original_fallback']}条"
        f"（{assignment['ratios']['original_fallback']:.2%}）。比例来自核验后的实际日志，不是全量比例代入。",
        '- 本次只训练前2,000条、500次更新，不是完整遍历4,000条的一轮训练。', '',
        '## 固定 dev200 结果与同步数正确组对照', '',
        'HotpotQA与MuSiQue各100题，原prompt及候选顺序；greedy、batch8、输出上限128，非法输出计空，不修复。',
        '下表为两个数据集的宏平均。正确组来自已有dev200记录，未重新运行正确组。', '',
        '| Step | 正确组覆盖@5 | 第二组覆盖@5 | 差值（百分点） | 正确组集合F1 | 第二组集合F1 | 第二组Recall@5 | 第二组格式合法率 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in new_history:
        step_id = row['step']
        m = row['metrics']['balanced_macro']
        c = correct_by_step[step_id]['metrics']['balanced_macro']
        delta = 100 * (m['all_annotated_supports_at_5'] - c['all_annotated_supports_at_5'])
        lines.append(f"| {step_id} | {c['all_annotated_supports_at_5']:.2%} | {m['all_annotated_supports_at_5']:.2%} | {delta:+.2f} | {c['support_f1_selected']:.2%} | {m['support_f1_selected']:.2%} | {m['support_recall_at_5']:.2%} | {m['valid']:.2%} |")
    lines += ['', '## Checkpoint选择与限制', '',
        f"- 沿用原规则：先比较balanced macro All-support@5，再集合F1，仍相同取较早step。第二组{'最终' if completed else '当前'}最佳为step-{best_new['step']}；正确组相同0..500预算内最佳为step-{best_correct['step']}。",
        '- 最佳仅依据dev200，不代表全量开发集最佳，也不能视为独立最终测试结论。',
        '- 未运行本组全量开发评估，未产生本组9,814题结果；此前正确组的全量结果不能替代本组评估。',
        '- 整份Rubric替换同时改变条目、权重与长度；自动领域标签可能有误，617条回退使本实验属于部分错配。',
        '- 映射对4,000条构造，训练只使用前2,000条，因此实际使用的Rubric边际分布不保证与正确组相同。',
        '- 单次训练与固定200题开发集结果不足以建立普遍因果结论；不据此改动算法、追加训练或启用最终测试。', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(RUN / 'full-validation-status.json', {'utc': utc(), 'stage': 'not_run',
        'full_development_evaluated': False, 'queries_evaluated': 0,
        'reason': 'Cancelled by user before full-development stage started'})
    return report


def main():
    initial = json.loads(STATUS.read_text())
    if initial['stage'] != 'training' or initial['state'] != 'running':
        raise ValueError('Expected active training; refusing an unsafe takeover')
    child_pid = int(initial['pid'])
    supervisor_pid = int((ROOT / 'server-logs/rubric-mixed-001-controller.pid').read_text())
    for pid, expected in ((child_pid, 'opsd.rubric_swap.train'), (supervisor_pid, 'opsd.rubric_swap.pipeline')):
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if expected not in cmd:
            raise ValueError('Process identity mismatch; refusing to signal')
    amendment = {'utc': utc(), 'reason': 'User requested training/dev200 report only; no full development evaluation',
        'training_pid': child_pid, 'controller_pid': os.getpid(), 'approved_stop_step': 500,
        'full_development_authorized': False, 'original_frozen_plan_preserved': True}
    with (RUN / 'training-only-amendment.json').open('x') as stream:
        json.dump(amendment, stream, indent=2)
    # Stop only the scheduler. Its training child owns an independent log file FD.
    os.kill(supervisor_pid, signal.SIGTERM)
    atomic_json(STATUS, {**initial, 'utc': utc(), 'controller_pid': os.getpid(), 'full_development_authorized': False})
    print(json.dumps(amendment), flush=True)
    plan = json.loads(PLAN.read_text())
    write_report(plan, completed=False)
    while alive(child_pid):
        time.sleep(1)
    final = json.loads((RUN / 'status.json').read_text())
    if (final.get('stage') != 'training_stopped_at_requested_boundary' or final.get('step') != 500 or
            not final.get('checkpoint_and_dev_complete')):
        raise RuntimeError('Training exited without a complete step-500 checkpoint/dev200: ' + str(final))
    verify(plan)
    report = write_report(plan, completed=True)
    atomic_json(RUN / 'training-only-integrity.json', {'frozen_plan_and_protected_files_unchanged': True,
        'full_development_evaluated': False, 'final_test_evaluated': False})
    atomic_json(STATUS, {'utc': utc(), 'stage': 'training_and_dev200', 'state': 'complete',
        'training_steps': 500, 'records_consumed': 2000, 'full_development_evaluated': False,
        'best_small_step': report['best_mixed_dev200_step'], 'report': str(RUN / 'REPORT.md')})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_json(STATUS, {'utc': utc(), 'stage': 'training_only_controller', 'state': 'failed', 'error': str(exc)})
        raise
