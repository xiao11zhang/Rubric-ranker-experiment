"""Freeze approved Group 05 data, baseline inputs, code and the 500-step boundary."""
import argparse
from collections import Counter
import json
from pathlib import Path

from opsd.common import digest, encode_prompt
from opsd.dev_eval.prepare import ROOT, jsonl, sha
from .audit import reviewed
from .entities import teacher_row
from .protocol import actual_baseline_prefix, load, protect, verify


def agent_artifact_path(path, root=ROOT):
    value = Path(path)
    if value.is_absolute() or value.drive or '..' in value.parts:
        raise ValueError('Agent provenance paths must be project-relative')
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Agent artifact escapes the project')
    return result


def check_agent_provenance(manifest, selected, root=ROOT):
    if (manifest.get('provider') != 'agent_annotations' or manifest.get('agent_runtime') != 'Codex' or
            manifest.get('review_method') != 'agent_semantic_review_and_deterministic_program_validation' or
            manifest.get('human_review_claimed') is not False or not manifest.get('semantic_review_complete')):
        raise ValueError('Incomplete agent annotation/review provenance')
    if any(k in manifest for k in ('model_files_sha256', 'model_sha256')):
        raise ValueError('Agent annotations must not impersonate local model weight provenance')
    if not manifest.get('rule_version') or not manifest.get('model_identity_note'):
        raise ValueError('Agent rule/model identity disclosure is missing')
    if manifest.get('records') != len(selected):
        raise ValueError('Agent annotation record count differs from selected training data')
    source_path = agent_artifact_path(manifest['source_manifest_path'], root)
    if sha(source_path) != manifest['source_manifest_sha256']:
        raise ValueError('Agent source manifest changed')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if (source.get('provider') != 'agent_annotations' or source['order_ids'] != [r['id'] for r in selected] or
            source['rules_sha256'] != digest(source['rules']) or source['rules_sha256'] != manifest['rules_sha256'] or
            source['actual_baseline_prefix_sha256'] != manifest['baseline_actual_prefix_sha256']):
        raise ValueError('Agent source/rules/order differs from the extraction manifest')
    batches = source['batches']
    if (len(batches) != manifest['batches'] or len({b['batch_id'] for b in batches}) != len(batches) or
            [uid for b in batches for uid in b['ids']] != source['order_ids']):
        raise ValueError('Agent batches do not cover the exact source prefix')
    for field in ('annotation_and_review_artifacts_sha256', 'code_sha256'):
        files = manifest.get(field)
        if not isinstance(files, dict) or not files:
            raise ValueError('Missing agent provenance artifact/code hashes')
        for path, expected in files.items():
            if sha(agent_artifact_path(path, root)) != expected:
                raise ValueError('Agent annotation/review/code artifact changed: ' + path)
    for batch in batches:
        if manifest['annotation_and_review_artifacts_sha256'].get(batch['path']) != batch['sha256']:
            raise ValueError('Source batch is not bound by the final manifest')


def check_provenance(items, manifest, selected, train_hash, root=ROOT):
    fingerprint = digest(manifest)
    if (manifest['train_sha256'] != train_hash or
            manifest['order_ids'] != [r['id'] for r in selected] or
            any(item['pipeline_fingerprint'] != fingerprint for item in items)):
        raise ValueError('Extraction provenance differs from source/order/candidates')
    provider = manifest.get('provider', 'local_causal_model')
    if provider == 'agent_annotations':
        check_agent_provenance(manifest, selected, root)
    elif provider == 'external_api':
        from .api_extract import check_api_provenance
        check_api_provenance(manifest, selected, root)
    elif provider == 'external_api_user_waived':
        from .user_waived import check_provenance as check_waived
        check_waived(manifest, selected, root)
    elif provider != 'local_causal_model':
        raise ValueError('Unsupported annotation provider')


def runtime(plan):
    identity = digest({k: v for k, v in plan.items() if k != 'config_sha256'})
    if identity != plan['config_sha256']:
        raise ValueError('Group 05 plan identity changed')
    verify(plan)
    from opsd.common import read_rows
    c = plan['config']
    rows = read_rows(ROOT / c['train_data'])
    order = plan['order_indices']
    if (len(order) != len(rows) or set(order) != set(range(len(rows))) or
            [rows[i]['id'] for i in order] != plan['order_ids']):
        raise ValueError('Training permutation changed')
    if plan['execution_stop_after_step'] != 500 or c['stop_after_step'] != 500:
        raise ValueError('Group 05 must stop at the frozen 500-step boundary')
    selected = [rows[i] for i in order[:2000]]
    derived = reviewed(selected, jsonl(plan['derived_data_path']), jsonl(plan['review_data_path']))
    return rows, derived


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('opsd/group05_deentity/config.toml'))
    p.add_argument('--plan', type=Path, default=Path('opsd/group05_deentity/plans/deentity-001.json'))
    a = p.parse_args()
    if a.plan.exists():
        raise FileExistsError(a.plan)
    c, baseline, rows, selected, baseline_root = load(a.config)
    guard = baseline_root / 'step-300'
    if not (guard / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError('Protected baseline step-300 adapter missing')
    prefix_hash = actual_baseline_prefix(baseline_root, baseline, c)
    source, review_path = ROOT / c['derived_data'], ROOT / c['review_data']
    items = jsonl(source)
    manifest_path = ROOT / c['extraction_manifest']
    extraction = json.loads(manifest_path.read_text(encoding='utf-8'))
    if c.get('annotation_provider') and extraction.get('provider') != c['annotation_provider']:
        raise ValueError('Configured annotation provider differs from actual source provenance')
    check_provenance(items, extraction, selected, sha(ROOT / c['train_data']))
    derived = reviewed(selected, items, jsonl(review_path))
    root = ROOT / c['output']
    if root.exists():
        raise FileExistsError(root)
    protect(a.plan, [baseline_root, c['model'], ROOT / c['train_data'], a.config, source, review_path])
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c['model'], revision=c['revision'], local_files_only=True)
    model_config = AutoConfig.from_pretrained(c['model'], revision=c['revision'], local_files_only=True)
    maxima = {'student': 0, 'teacher': 0}
    for row in selected:
        pair = {'student': len(encode_prompt(tokenizer, row, top_k=c['top_k'])),
                'teacher': len(encode_prompt(tokenizer, teacher_row(row, derived), teacher=True, mode='correct', top_k=c['top_k']))}
        if max(pair.values()) > c['max_prompt_tokens'] or max(pair.values()) + c['max_response_tokens'] > model_config.max_position_embeddings:
            raise ValueError('Prompt too long; no truncation/filtering: ' + str(row['id']))
        maxima = {k: max(maxima[k], pair[k]) for k in maxima}
    paths = [a.config.resolve(), (ROOT / c['baseline_plan']).resolve(), source.resolve(), review_path.resolve(), manifest_path.resolve(),
             baseline_root / 'train.jsonl', *Path(__file__).parent.glob('*.py')]
    paths += [p for p in guard.iterdir() if p.is_file() and p.name != 'resume.pt']
    if extraction.get('provider') == 'agent_annotations':
        paths += [agent_artifact_path(p) for p in extraction['annotation_and_review_artifacts_sha256']]
    elif extraction.get('provider') == 'external_api':
        paths += [agent_artifact_path(p) for p in extraction['artifacts_sha256']]
    frozen = {str(p.resolve()): sha(p) for p in paths}
    summary = {'records': len(items), 'changed_records': sum(bool(x['changes']) for x in items),
               'unchanged_no_entities': sum(not x['changes'] and x.get('processing_mode') != 'original_fallback' for x in items),
               'entity_groups_by_type': dict(Counter(e['type'] for x in items for e in x['entities'])),
               'replacement_occurrences': sum(len(x['changes']) for x in items),
               'corrected_records': sum(bool(x.get('correction_applied')) for x in items),
               'source': 'deterministic_named_entity_replacement_with_disclosed_review_scope',
               'fallback_records': sum(x.get('processing_mode') == 'original_fallback' for x in items),
               'sample_audit_complete': extraction.get('sample_audit_complete', False),
               'annotation_provider': extraction.get('provider', 'local_causal_model'),
               'review_method': extraction.get('review_method', 'recorded_review'),
               'human_review_claimed': extraction.get('human_review_claimed', None)}
    plan = {**baseline, 'group': '05_deentity', 'config': c,
            'config_file_sha256': sha(a.config), 'execution_stop_after_step': 500,
            'derived_data_path': str(source.resolve()), 'review_data_path': str(review_path.resolve()),
            'baseline_root': str(baseline_root), 'baseline_actual_prefix_sha256': prefix_hash,
            'selected_ids_sha256': digest([r['id'] for r in selected]),
            'maximum_prompt_tokens': maxima, 'deentity_summary': summary,
            'small_eval_steps': [s for s in baseline['small_eval_steps'] if s <= 500],
            'checkpoint_steps': [s for s in baseline['checkpoint_steps'] if s <= 500],
            'full_eval_steps': [0, 300, 500, 'best_small_development'],
            'input_and_execution_sha256': {**baseline['input_and_execution_sha256'], **frozen}}
    plan['config_sha256'] = digest({k: v for k, v in plan.items() if k != 'config_sha256'})
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    with a.plan.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'plan': str(a.plan), 'summary': summary, 'stop_after_step': 500,
                      'baseline_actual_prefix_verified': True, 'training_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
