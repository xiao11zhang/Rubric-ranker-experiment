"""Group-local reports based on actual IDs and rubric hashes, never fabricated scores."""
from collections import Counter
from pathlib import Path

from opsd.full_run.utils import atomic_json


def actual_report(plan, entries, derived):
    c = plan['config']
    steps = plan['execution_stop_after_step']
    samples = [s for entry in entries for s in entry['samples']]
    ids = [s['id'] for s in samples]
    if ([e['step'] for e in entries] != list(range(1, steps + 1)) or
            any(len(e['samples']) != c['gradient_accumulation'] for e in entries) or
            ids != plan['order_ids'][:steps * c['gradient_accumulation']] or len(ids) != len(set(ids))):
        raise ValueError('Actual training differs from the baseline prefix')
    for sample in samples:
        item = derived[sample['id']]
        if (sample['teacher_rubric_hash'] != item['rubric_hash'] or
                sample['original_rubric_hash'] != item['original_rubric_hash']):
            raise ValueError('Actual Teacher rubric hash differs from approved data')
    return {'group': '05_deentity', 'verified_actual_log': True, 'steps': steps, 'records': len(ids),
            'unique_records': len(set(ids)), 'matches_baseline_prefix': True,
            'changed_records': sum(bool(derived[i]['changes']) for i in ids),
            'unchanged_no_entities': sum(not derived[i]['changes'] and derived[i].get('processing_mode') != 'original_fallback' for i in ids),
            'replacement_occurrences': sum(len(derived[i]['changes']) for i in ids),
            'entity_groups_by_type': dict(Counter(e['type'] for i in ids for e in derived[i]['entities'])),
            'fallback_records': sum(derived[i].get('processing_mode') == 'original_fallback' for i in ids), 'full_epoch_completed': False}


def training_report(root, plan, actual, history):
    from opsd.full_run.utils import select_best
    root = Path(root)
    reports = root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / 'actual-entities.json', actual)
    best = select_best(history)
    lines = ['# 第五组：原始 Rubric 去实体化', '',
             'Teacher/Student 共用基线相同 SFT 起点；只按标注替换 Rubric description 中的命名实体。剩余抽查由用户明确取消，不能声称全量语义审核完成。',
             f"标注来源：{plan['deentity_summary'].get('annotation_provider', 'recorded_provider')}；审核方式：{plan['deentity_summary'].get('review_method', 'recorded_review')}。Agent审核不得称为人工审核。",
             'Query、候选文档、日期/数值条件、其余描述文字、结构和权重保留。Teacher 仍可从 Query 看见实体。',
             f"训练完成 {actual['steps']} step、{actual['records']} 条，与基线实际输入前缀一致；不是完整4000条遍历。",
             f"有替换 {actual['changed_records']} 条；未标注到目标实体、原样保留 {actual['unchanged_no_entities']} 条；用户授权异常整份回退 {actual['fallback_records']} 条。",
             f"dev200最佳 step：{best['step']}；选点规则：{plan['best_rule']}。", '',
             '| Step | All-support@5（balanced macro） | 集合F1 |', '|---:|---:|---:|']
    for item in sorted(history, key=lambda x: x['step']):
        m = item['metrics']['balanced_macro']
        lines.append(f"| {item['step']} | {m['all_annotated_supports_at_5']:.2%} | {m['support_f1_selected']:.2%} |")
    lines += ['', '9814条全量开发集包含dev200，不能称为独立最终测试集；本报告不虚构尚未执行的全量结果。', '']
    (reports / 'TRAINING.md').write_text('\n'.join(lines), encoding='utf-8')
