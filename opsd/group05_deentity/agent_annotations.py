"""Local agent-authored entity annotations, deterministic previews, explicit agent reviews.

No model calls, weight hashes, automatic approval, server writes or training.
"""
import argparse
import json
from pathlib import Path

from opsd.common import digest
from opsd.dev_eval.prepare import ROOT, sha, jsonl
from opsd.full_run.utils import atomic_json, utc
from .entities import candidate, descriptions, validate_candidate
from .protocol import load

BASE = ROOT / 'opsd/group05_deentity/data/agent'
RULE_VERSION = 'group05-agent-annotations-v1'
RULES = {
    'target_types': ['person', 'organization', 'location', 'work'],
    'scope': 'Named people including characters, named organizations, named places, named works only',
    'preserve': ['dates', 'independent years', 'numerical constraints', 'relations', 'negation',
                 'comparison conditions', 'item count/order', 'schema', 'weights', 'all other text'],
    'exclude': ['generic roles/concepts', 'nationality adjectives', 'awards/events outside target types'],
    'aliases': 'Only genuinely identical entities share a placeholder; distinct people remain distinct',
    'numeric_work_titles': 'Require exact unique contexts; never remove homographic independent years',
    'no_entities': 'Keep unchanged, but explicitly review for omissions',
    'source_visibility': 'Agent reads rubric descriptions; no query, answer or candidate documents supplied',
    'review': 'Agent semantic review of source and deterministic preview, plus program reconstruction checks',
}
IDENTITY = {'provider': 'agent_annotations', 'agent_runtime': 'Codex', 'model_id': None,
            'model_identity_note': 'No independently verifiable exact model ID or model weights exposed in this session',
            'review_method': 'agent_semantic_review_and_deterministic_program_validation',
            'human_review_claimed': False}


def fresh_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sources(config):
    c, baseline, _, selected, _ = load(config, check_hashes=False)
    source = ROOT / c['train_data']
    expected = [h for p, h in baseline['input_and_execution_sha256'].items()
                if p.replace('\\', '/').endswith('/' + c['train_data'])]
    if len(expected) != 1 or sha(source) != expected[0]:
        raise ValueError('Local original training data differs from frozen baseline')
    log = ROOT / 'opsd/group05_deentity/data/source/baseline-train.jsonl'
    entries = jsonl(log)[:500]
    if ([r['step'] for r in entries] != list(range(1, 501)) or
            any(len(r['samples']) != 4 for r in entries) or
            [s['id'] for r in entries for s in r['samples']] != [r['id'] for r in selected]):
        raise ValueError('Original actual baseline input sequence does not match 2000 selected rows')
    return c, baseline, selected, log, digest(entries)


def initialize(config, batch_size):
    if batch_size <= 0:
        raise ValueError('Batch size must be positive')
    if (BASE / 'source-manifest.json').exists():
        raise FileExistsError('Source is already initialized')
    c, baseline, rows, log, prefix_hash = sources(config)
    batches = []
    for start in range(0, len(rows), batch_size):
        batch_id = f'batch-{start // batch_size + 1:04d}'
        records = [{'id': r['id'], 'original_rubric_hash': digest(r['rubrics']),
                    'descriptions': [{'path': list(p), 'text': text} for p, text in descriptions(r['rubrics'])]}
                   for r in rows[start:start + batch_size]]
        path = BASE / 'batches' / (batch_id + '.input.json')
        fresh_json(path, {'batch_id': batch_id, 'rule_version': RULE_VERSION, 'records': records})
        batches.append({'batch_id': batch_id, 'path': rel(path), 'sha256': sha(path),
                        'ids': [r['id'] for r in records]})
    manifest = {**IDENTITY, 'utc': utc(), 'rule_version': RULE_VERSION, 'rules': RULES,
        'rules_sha256': digest(RULES), 'config_path': rel(config), 'config_sha256': sha(config),
        'train_path': c['train_data'], 'train_sha256': sha(ROOT / c['train_data']),
        'baseline_plan_path': c['baseline_plan'], 'baseline_plan_sha256': sha(ROOT / c['baseline_plan']),
        'actual_baseline_log_path': rel(log), 'actual_baseline_log_sha256': sha(log),
        'actual_baseline_prefix_sha256': prefix_hash, 'order_ids': [r['id'] for r in rows],
        'tool_code_sha256': {rel(Path(__file__)): sha(__file__),
                             rel(Path(__file__).with_name('entities.py')): sha(Path(__file__).with_name('entities.py'))},
        'batch_size': batch_size, 'batches': batches}
    fresh_json(BASE / 'source-manifest.json', manifest)
    progress()
    print(json.dumps({'records': len(rows), 'batches': len(batches), 'batch_size': batch_size,
                      'provider': 'agent_annotations', 'actual_baseline_prefix_verified': True}))


def context(batch_id):
    source = json.loads((BASE / 'source-manifest.json').read_text(encoding='utf-8'))
    for path, h in source['tool_code_sha256'].items():
        if sha(ROOT / path) != h:
            raise ValueError('Annotation/replacement code changed after source freeze: ' + path)
    for path, h in source.get('amendment_artifacts_sha256', {}).items():
        if sha(ROOT / path) != h:
            raise ValueError('Source amendment evidence changed: ' + path)
    for field, hfield in (('config_path', 'config_sha256'), ('train_path', 'train_sha256'),
                         ('baseline_plan_path', 'baseline_plan_sha256')):
        if sha(ROOT / source[field]) != source[hfield]:
            raise ValueError('Frozen local annotation source changed')
    batch = next((b for b in source['batches'] if b['batch_id'] == batch_id), None)
    if batch is None or sha(ROOT / batch['path']) != batch['sha256']:
        raise ValueError('Unknown or changed source batch')
    _, _, rows, _, _ = sources(ROOT / source['config_path'])
    by_id = {r['id']: r for r in rows}
    inputs = json.loads((ROOT / batch['path']).read_text(encoding='utf-8'))
    for record in inputs['records']:
        if record['original_rubric_hash'] != digest(by_id[record['id']]['rubrics']):
            raise ValueError('Batch source rubric hash changed')
    return source, batch, inputs, by_id


def commit(batch_id, annotations):
    source, batch, inputs, by_id = context(batch_id)
    raw = json.loads(Path(annotations).read_text(encoding='utf-8'))
    if raw['batch_id'] != batch_id or [r['id'] for r in raw['records']] != batch['ids']:
        raise ValueError('Agent annotation coverage/order differs from source batch')
    fingerprint = digest({'provider': 'agent_annotations', 'source_batch_sha256': batch['sha256'],
                          'annotations_sha256': sha(annotations), 'rules_sha256': source['rules_sha256']})
    candidates, preview = [], []
    for annotation in raw['records']:
        row = by_id[annotation['id']]
        item = candidate(row, annotation['entities'], fingerprint, json.dumps(annotation, ensure_ascii=False))
        validate_candidate(row, item)
        item.update(provider='agent_annotations', source_batch_id=batch_id)
        candidates.append(item)
        before = dict(descriptions(row['rubrics']))
        preview.append({'id': row['id'], 'rubric_hash': item['rubric_hash'], 'entities': item['entities'],
            'fields': [{'path': list(p), 'original': before[p], 'converted': text}
                       for p, text in descriptions(item['rubrics'])],
            'changes': item['changes']})
    folder = BASE / 'batches'
    fresh_json(folder / (batch_id + '.candidates.json'), candidates)
    fresh_json(folder / (batch_id + '.preview.json'), preview)
    fresh_json(folder / (batch_id + '.commit.json'), {'batch_id': batch_id, 'provider': 'agent_annotations',
        'source_batch_sha256': batch['sha256'], 'annotations_path': rel(annotations),
        'annotations_sha256': sha(annotations), 'candidates_sha256': sha(folder / (batch_id + '.candidates.json')),
        'preview_sha256': sha(folder / (batch_id + '.preview.json')), 'semantic_review_complete': False})
    progress()
    print(json.dumps({'batch': batch_id, 'converted': len(candidates), 'review_required': True}))


def review(batch_id, decisions):
    _, batch, _, _ = context(batch_id)
    folder = BASE / 'batches'
    committed = json.loads((folder / (batch_id + '.commit.json')).read_text(encoding='utf-8'))
    for suffix, hfield in (('.candidates.json', 'candidates_sha256'), ('.preview.json', 'preview_sha256')):
        if sha(folder / (batch_id + suffix)) != committed[hfield]:
            raise ValueError('Candidate/preview changed after agent annotation')
    items = json.loads((folder / (batch_id + '.candidates.json')).read_text(encoding='utf-8'))
    raw = json.loads(Path(decisions).read_text(encoding='utf-8'))
    if raw['batch_id'] != batch_id or [r['id'] for r in raw['records']] != batch['ids']:
        raise ValueError('Explicit review decisions must cover each batch record in order')
    reviews = []
    for item, decision in zip(items, raw['records']):
        if (decision.get('decision') != 'approved' or not decision.get('notes', '').strip() or
                decision.get('review_method') != IDENTITY['review_method']):
            raise ValueError('Every row requires an explicit documented agent semantic review')
        reviews.append({'id': item['id'], 'original_rubric_hash': item['original_rubric_hash'],
            'rubric_hash': item['rubric_hash'], 'decision': 'approved', 'reviewer': 'Codex agent',
            'review_method': IDENTITY['review_method'], 'human_review_claimed': False, 'notes': decision['notes']})
    fresh_json(folder / (batch_id + '.review.json'), reviews)
    fresh_json(folder / (batch_id + '.review-provenance.json'), {'batch_id': batch_id,
        'review_decisions_path': rel(decisions), 'review_decisions_sha256': sha(decisions),
        'preview_sha256': committed['preview_sha256'], 'review_sha256': sha(folder / (batch_id + '.review.json'))})
    progress()
    print(json.dumps({'batch': batch_id, 'agent_reviewed': len(reviews), 'human_review_claimed': False}))


def progress():
    path = BASE / 'source-manifest.json'
    if not path.exists():
        return
    source = json.loads(path.read_text(encoding='utf-8'))
    committed, reviewed_count, pending = 0, 0, []
    for batch in source['batches']:
        stem = BASE / 'batches' / batch['batch_id']
        if stem.with_suffix('.commit.json').exists():
            committed += len(batch['ids'])
        if stem.with_suffix('.review.json').exists():
            reviewed_count += len(batch['ids'])
        else:
            pending.append(batch['batch_id'])
    atomic_json(BASE / 'progress.json', {'utc': utc(), 'provider': 'agent_annotations',
        'total_records': len(source['order_ids']), 'annotated_records': committed,
        'agent_reviewed_records': reviewed_count, 'total_batches': len(source['batches']),
        'next_pending_batch': pending[0] if pending else None, 'remaining_batches': len(pending),
        'all_agent_reviews_complete': not pending, 'human_review_claimed': False,
        'server_upload_performed': False, 'training_started': False})


def finalize():
    source = json.loads((BASE / 'source-manifest.json').read_text(encoding='utf-8'))
    _, _, rows, _, _ = sources(ROOT / source['config_path'])
    artifacts, annotated, reviews = {}, [], []
    for batch in source['batches']:
        context(batch['batch_id'])
        stem = BASE / 'batches' / batch['batch_id']
        commit_info = json.loads(stem.with_suffix('.commit.json').read_text(encoding='utf-8'))
        review_info = json.loads(stem.with_suffix('.review-provenance.json').read_text(encoding='utf-8'))
        for path, h in ((ROOT / commit_info['annotations_path'], commit_info['annotations_sha256']),
                        (ROOT / review_info['review_decisions_path'], review_info['review_decisions_sha256']),
                        (stem.with_suffix('.candidates.json'), commit_info['candidates_sha256']),
                        (stem.with_suffix('.preview.json'), commit_info['preview_sha256']),
                        (stem.with_suffix('.review.json'), review_info['review_sha256'])):
            if sha(path) != h:
                raise ValueError('Agent annotation/review artifact changed')
            artifacts[rel(path)] = h
        annotated.extend(json.loads((ROOT / commit_info['annotations_path']).read_text(encoding='utf-8'))['records'])
        reviews.extend(json.loads(stem.with_suffix('.review.json').read_text(encoding='utf-8')))
    if [a['id'] for a in annotated] != source['order_ids'] or len(reviews) != 2000:
        raise ValueError('All 2000 records must be explicitly annotated and reviewed')
    artifacts[rel(BASE / 'source-manifest.json')] = sha(BASE / 'source-manifest.json')
    artifacts.update(source.get('amendment_artifacts_sha256', {}))
    artifacts[source['actual_baseline_log_path']] = source['actual_baseline_log_sha256']
    artifacts.update({b['path']: b['sha256'] for b in source['batches']})
    code = {rel(p): sha(p) for p in Path(__file__).parent.glob('*.py')}
    manifest = {**IDENTITY, 'rule_version': RULE_VERSION, 'rules_sha256': source['rules_sha256'],
        'source_manifest_path': rel(BASE / 'source-manifest.json'),
        'source_manifest_sha256': sha(BASE / 'source-manifest.json'),
        'train_sha256': source['train_sha256'], 'order_ids': source['order_ids'],
        'baseline_actual_prefix_sha256': source['actual_baseline_prefix_sha256'],
        'annotation_and_review_artifacts_sha256': artifacts, 'code_sha256': code,
        'batches': len(source['batches']), 'records': 2000, 'semantic_review_complete': True}
    fingerprint = digest(manifest)
    items = [candidate(row, annotation['entities'], fingerprint, json.dumps(annotation, ensure_ascii=False))
             for row, annotation in zip(rows, annotated)]
    for item in items:
        item['provider'] = 'agent_annotations'
    from .audit import reviewed
    reviewed(rows, items, reviews)
    c = json.loads((BASE / 'source-manifest.json').read_text())
    config, _, _, _, _ = load(ROOT / c['config_path'], check_hashes=False)
    for path, values in ((ROOT / config['derived_data'], items), (ROOT / config['review_data'], reviews)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    fresh_json(ROOT / config['extraction_manifest'], manifest)
    progress()
    print(json.dumps({'provider': 'agent_annotations', 'records': 2000, 'agent_review_complete': True,
                      'uploaded': False, 'training_started': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'commit', 'review', 'progress', 'finalize'))
    parser.add_argument('--config', type=Path, default=Path('opsd/group05_deentity/config.toml'))
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--batch')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--decisions', type=Path)
    args = parser.parse_args()
    if args.action == 'init':
        initialize(args.config, args.batch_size)
    elif args.action == 'commit':
        commit(args.batch, args.annotations)
    elif args.action == 'review':
        review(args.batch, args.decisions)
    elif args.action == 'finalize':
        finalize()
    else:
        progress()


if __name__ == '__main__':
    main()
