"""Archive and regenerate one unreviewed batch; never approves its replacement."""
import argparse
import json
from pathlib import Path

from opsd.dev_eval.prepare import ROOT, sha
from opsd.full_run.utils import atomic_json, utc
from .agent_annotations import BASE, commit, context, fresh_json, rel
from .entities import candidate, validate_candidate


def revise(batch_id, annotations, reason):
    source, batch, _, rows = context(batch_id)
    folder = BASE / 'batches'
    if (folder / (batch_id + '.review.json')).exists():
        raise ValueError('Reviewed batches require a separate review revocation workflow')
    incoming = json.loads(annotations.read_text(encoding='utf-8'))
    if incoming['batch_id'] != batch_id or [r['id'] for r in incoming['records']] != batch['ids']:
        raise ValueError('Revised annotations must cover the unchanged complete batch')
    old_commit = json.loads((folder / (batch_id + '.commit.json')).read_text(encoding='utf-8'))
    for suffix, field in (('.candidates.json', 'candidates_sha256'), ('.preview.json', 'preview_sha256')):
        if sha(folder / (batch_id + suffix)) != old_commit[field]:
            raise ValueError('Old batch evidence changed')
    if sha(ROOT / old_commit['annotations_path']) != old_commit['annotations_sha256']:
        raise ValueError('Old annotations changed')
    for item in incoming['records']:
        row = rows[item['id']]
        validate_candidate(row, candidate(row, item['entities'], 'revision-precheck'))
    archive = BASE / 'amendments' / (batch_id + '-revision-001')
    archive.mkdir(parents=True, exist_ok=False)
    fresh_json(archive / 'source-manifest.before.json', source)
    for suffix in ('.candidates.json', '.preview.json', '.commit.json'):
        original = (folder / (batch_id + suffix)).resolve()
        target = (archive / original.name).resolve()
        if not original.is_relative_to(BASE.resolve()) or not target.is_relative_to(BASE.resolve()):
            raise ValueError('Revision paths must remain inside the local annotation workspace')
        original.rename(target)
    commit(batch_id, annotations)
    fresh_json(archive / 'revision.json', {'utc': utc(), 'reason': reason,
        'original_annotations_path': old_commit['annotations_path'],
        'original_annotations_sha256': old_commit['annotations_sha256'],
        'revised_annotations_path': rel(annotations), 'revised_annotations_sha256': sha(annotations),
        'new_commit_sha256': sha(folder / (batch_id + '.commit.json')),
        'semantic_review_complete': False, 'human_review_claimed': False})
    source.setdefault('amendment_artifacts_sha256', {}).update(
        {rel(path): sha(path) for path in archive.iterdir() if path.is_file()})
    source['amendment_artifacts_sha256'][old_commit['annotations_path']] = old_commit['annotations_sha256']
    atomic_json(BASE / 'source-manifest.json', source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    revise(args.batch, args.annotations, args.reason)


if __name__ == '__main__':
    main()
