"""Create a human review worksheet and report, or apply explicit entity corrections."""
import argparse
from collections import Counter
import json
from pathlib import Path

from opsd.common import digest
from opsd.dev_eval.prepare import ROOT, jsonl, sha
from .entities import candidate, validate_candidate
from .protocol import load, protect


def reviewed(rows, items, reviews):
    expected = [r['id'] for r in rows]
    if [r['id'] for r in items] != expected:
        raise ValueError('Candidates must exactly match baseline first-2000 order')
    if len(reviews) != len(rows) or len({r['id'] for r in reviews}) != len(reviews):
        raise ValueError('Review records missing or duplicated')
    by_id = {r['id']: r for r in reviews}
    if set(by_id) != set(expected):
        raise ValueError('Review IDs differ from training prefix')
    for row, item in zip(rows, items):
        validate_candidate(row, item)
        approval = by_id[row['id']]
        if (approval.get('decision') != 'approved' or not isinstance(approval.get('reviewer'), str)
                or not approval['reviewer'].strip() or approval.get('original_rubric_hash') != item['original_rubric_hash']
                or approval.get('rubric_hash') != item['rubric_hash']):
            raise ValueError('Missing/stale review approval: ' + str(row['id']))
    return {r['id']: r for r in items}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('opsd/group05_deentity/config.toml'))
    p.add_argument('--report', type=Path, default=Path('opsd/group05_deentity/reports/entity-audit.md'))
    p.add_argument('--corrections', type=Path, help='JSONL of {id, entities:[{type, mentions}]}')
    p.add_argument('--corrected-output', type=Path)
    a = p.parse_args()
    c, _, _, rows, baseline_root = load(a.config, check_hashes=False)
    source = ROOT / c['derived_data']
    items = jsonl(source)
    if [r['id'] for r in items] != [r['id'] for r in rows]:
        raise ValueError('Candidate order/coverage differs from baseline prefix')
    if a.corrections:
        if not a.corrected_output:
            p.error('--corrections requires --corrected-output (new file)')
        changes = jsonl(a.corrections)
        mapping = {x['id']: x['entities'] for x in changes}
        if len(mapping) != len(changes) or not set(mapping).issubset({r['id'] for r in rows}):
            raise ValueError('Unknown/duplicate correction ID')
        protect(a.corrected_output, [baseline_root, c['model'], ROOT / c['train_data'], source, a.config, a.corrections])
        fixed = [candidate(row, mapping[row['id']], item['pipeline_fingerprint'], item.get('raw_extraction', ''))
                 if row['id'] in mapping else item for row, item in zip(rows, items)]
        for old, item in zip(items, fixed):
            if item['id'] in mapping:
                item.update(correction_applied=True, previous_candidate_sha256=digest(old),
                            correction_file_sha256=sha(a.corrections))
        a.corrected_output.parent.mkdir(parents=True, exist_ok=True)
        with a.corrected_output.open('x', encoding='utf-8') as stream:
            for item in fixed:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        print('Wrote corrected candidates; set derived_data to this file, then create a new review worksheet.')
        return
    review_path = ROOT / c['review_data']
    for dest in (a.report, review_path):
        protect(dest, [baseline_root, c['model'], ROOT / c['train_data'], source, a.config])
        if dest.exists():
            raise FileExistsError(dest)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Group 05 去实体审核', '',
             '逐项核对漏实体、错误抽象、别名合并、指代、日期/数值条件。无实体记录也需审核。',
             '模型检测不保证穷尽实体；程序只保证所声明替换以外的结构和文字不变。', '']
    decisions = []
    for row, item in zip(rows, items):
        try:
            validate_candidate(row, item)
            check = '结构与声明替换一致；待人工语义审核'
        except (ValueError, KeyError, TypeError) as exc:
            check = '必须修正：' + str(exc)
        lines += ['## ' + str(row['id']), '', 'Query: ' + row['query'], '', check, '',
                  '原始 Rubric：', '```json', json.dumps(row['rubrics'], ensure_ascii=False, indent=2), '```',
                  '转换后 Rubric 与实体：', '```json',
                  json.dumps({'rubrics': item.get('rubrics'), 'entities': item.get('entities'),
                              'error': item.get('error')}, ensure_ascii=False, indent=2), '```', '']
        decisions.append({'id': row['id'], 'original_rubric_hash': item['original_rubric_hash'],
                          'rubric_hash': item.get('rubric_hash'), 'decision': 'pending', 'reviewer': '', 'notes': ''})
    with review_path.open('x', encoding='utf-8') as stream:
        for value in decisions:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    with a.report.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({'review': str(review_path), 'report': str(a.report),
                      'statuses': dict(Counter(r['status'] for r in items)), 'automatic_approval': False}))


if __name__ == '__main__':
    main()
