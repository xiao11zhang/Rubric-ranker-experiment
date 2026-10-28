"""Explicit user-authorized original-rubric fallback; local data operations only.

Keeps the frozen API extraction code and training algorithm unchanged. Fallback
is an identity transform, never a claim that no named entities were detected.
"""
import argparse
import json
from pathlib import Path

from opsd.common import digest
from opsd.dev_eval.prepare import ROOT, sha, jsonl
from opsd.full_run.utils import atomic_json
from . import api_extract as api
from . import api_sampling as sampling
from .entities import candidate, validate_candidate


def identity(row, fingerprint, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Fallback requires a specific uncertainty reason')
    item = candidate(row, [], fingerprint)
    item.update(processing_mode='original_fallback', deentity_success=False,
                fallback_reason=reason, fallback_authorization='user-explicit-2026-10-27')
    validate_candidate(row, item)
    if item['rubrics'] != row['rubrics'] or item['changes']:
        raise ValueError('Fallback must preserve the complete original rubric')
    return item


def apply(config, decisions):
    c, _, rows, _, _ = api.agent.sources(config)
    base = ROOT / Path(c['derived_data']).parent / 'api' / c['api_run_name']
    if (base / 'active.lock').exists():
        raise ValueError('Wait for API workers before applying fallback')
    run = json.loads((base / 'run.json').read_text(encoding='utf-8'))
    api.verify_files(run['spec']['frozen_files_sha256'])
    fingerprint = digest(run['spec'])
    records = api.read_records(base, rows, fingerprint)
    by_id = {r['id']: (i, r) for i, r in enumerate(rows)}
    if not isinstance(decisions, list) or len({v['id'] for v in decisions}) != len(decisions):
        raise ValueError('Fallback decisions require unique IDs')
    # Validate the whole decision file before mutating records.
    for v in decisions:
        if v['id'] not in records or v['id'] in run['imported_agent_records']:
            raise ValueError('Fallback must refer to a current API record')
        identity(by_id[v['id']][1], fingerprint, v['reason'])
    for v in decisions:
        uid = v['id']; i, row = by_id[uid]
        notes = '用户授权原Rubric回退；仅核验原文完整保留，不声称去实体成功。原因：' + v['reason']
        sampling.repair(base, rows, records, fingerprint,
                        [{'id': uid, 'entities': [], 'notes': notes}])
        record = records[uid]
        record.update(processing_mode='original_fallback', fallback_reason=v['reason'])
        record['candidate'] = identity(row, fingerprint, v['reason'])
        atomic_json(base / 'records' / f'{i+1:04d}.json', record)
        review_path = base / 'audit/reviews' / f'{i+1:04d}.json'
        review = json.loads(review_path.read_text(encoding='utf-8'))
        review.update(processing_mode='original_fallback', deentity_success=False,
                      semantic_review_scope='original_identity_and_fallback_decision_only',
                      deentity_semantic_reviewed=False)
        atomic_json(review_path, review)
    api.progress(base, rows, records, run['imported_agent_records'])
    counts = {r['id']: r['fallback_reason'] for r in records.values()
              if r.get('processing_mode') == 'original_fallback'}
    atomic_json(base / 'audit/original-fallback-summary.json',
                {'records': len(counts), 'total': len(rows), 'reasons': counts,
                 'deentity_success_claimed': False, 'api_calls': 0})
    return {'original_fallback_records': len(counts), 'api_calls': 0}


def finalize(config):
    c, _, rows, _, _ = api.agent.sources(config)
    base = ROOT / Path(c['derived_data']).parent / 'api' / c['api_run_name']
    # Retain all original provenance and risk/sample gates; do not bypass them.
    api.finalize(config, base)
    manifest_path = ROOT / c['extraction_manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = api.read_records(base, rows, digest(json.loads((base/'run.json').read_text(encoding='utf-8'))['spec']))
    fallback = {uid: r['fallback_reason'] for uid, r in records.items()
                if r.get('processing_mode') == 'original_fallback'}
    items = jsonl(ROOT / c['derived_data'])
    manifest.update(original_fallback_records=len(fallback), original_fallback_reasons=fallback,
                    original_fallback_fraction=len(fallback)/len(rows),
                    fallback_is_not_deentity_success=True)
    manifest['artifacts_sha256'][api.rel(Path(__file__).resolve())] = sha(Path(__file__))
    summary = base / 'audit/original-fallback-summary.json'
    if summary.exists():
        manifest['artifacts_sha256'][api.rel(summary)] = sha(summary)
    fingerprint = digest(manifest)
    for row, item in zip(rows, items):
        item['pipeline_fingerprint'] = fingerprint
        if row['id'] in fallback:
            fresh = identity(row, fingerprint, fallback[row['id']])
            item.update(fresh, annotation_provider='user_authorized_original_fallback')
        validate_candidate(row, item)
    path = ROOT / c['derived_data']
    temp = path.with_suffix('.fallback.tmp')
    with temp.open('x', encoding='utf-8') as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
    temp.replace(path)
    atomic_json(manifest_path, manifest)
    api.check_api_provenance(manifest, rows)
    report = api.REPORTS / 'API_DATA_PREPARATION.md'
    with report.open('a', encoding='utf-8') as stream:
        stream.write(f'\n## 用户授权的原Rubric回退\n\n{len(fallback)}/{len(rows)}条（{len(fallback)/len(rows):.2%}）明确标为original_fallback，整份保留原Rubric。它们不是无实体样本，不计作去实体成功；此前程序校验/审核计数包含这些原文保留决策，不代表全量去实体语义通过。逐条原因见manifest。\n')
    return {'finalized': True, 'original_fallback_records': len(fallback), 'records': len(rows)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('opsd/group05_deentity/config.api.single.toml'))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--decisions', type=Path)
    g.add_argument('--finalize', action='store_true')
    a = p.parse_args()
    result = finalize(a.config) if a.finalize else apply(a.config, json.loads(a.decisions.read_text(encoding='utf-8')))
    print(json.dumps(result, ensure_ascii=False))
