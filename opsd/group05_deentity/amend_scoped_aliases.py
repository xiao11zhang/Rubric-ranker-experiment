"""One-time, evidenced preprocessing amendment; preserves existing review outputs."""
import json
from pathlib import Path

from opsd.dev_eval.prepare import ROOT, sha
from opsd.full_run.utils import atomic_json, utc
from .agent_annotations import BASE, rel, fresh_json
from .entities import transform, validate_candidate


def main():
    archive = BASE / 'amendments/scoped-aliases-001'
    old_path = archive / 'source-manifest.before.json'
    current = BASE / 'source-manifest.json'
    if sha(old_path) != sha(current):
        raise ValueError('Source already amended or differs from archived source')
    source = json.loads(old_path.read_text(encoding='utf-8'))
    for name in ('entities', 'agent_annotations'):
        before = archive / (name + '.before.py')
        if sha(before) != source['tool_code_sha256']['opsd/group05_deentity/' + name + '.py']:
            raise ValueError('Archived original code does not match frozen source')
    rows = {r['id']: r for r in (json.loads(line) for line in
            (ROOT / source['train_path']).read_text(encoding='utf-8').splitlines())}
    evidence, count = {}, 0
    for batch in source['batches']:
        folder = BASE / 'batches'
        commit_path = folder / (batch['batch_id'] + '.commit.json')
        if not commit_path.exists():
            continue
        committed = json.loads(commit_path.read_text(encoding='utf-8'))
        candidates_path = folder / (batch['batch_id'] + '.candidates.json')
        preview_path = folder / (batch['batch_id'] + '.preview.json')
        if sha(candidates_path) != committed['candidates_sha256'] or sha(preview_path) != committed['preview_sha256']:
            raise ValueError('Existing candidate or preview changed')
        items = json.loads(candidates_path.read_text(encoding='utf-8'))
        for item in items:
            validate_candidate(rows[item['id']], item)
            entities = [{k: v for k, v in e.items() if k != 'placeholder'} for e in item['entities']]
            result, normalized, changes = transform(rows[item['id']]['rubrics'], entities)
            if (result, normalized, changes) != (item['rubrics'], item['entities'], item['changes']):
                raise ValueError('Previously reviewed transformation would change')
            count += 1
        for path in folder.glob(batch['batch_id'] + '.*.json'):
            evidence[rel(path)] = sha(path)
    code_before = source['tool_code_sha256'].copy()
    code_after = {path: sha(ROOT / path) for path in code_before}
    proof = archive / 'amendment.json'
    fresh_json(proof, {'utc': utc(), 'scope': 'Preprocessing only; no OPSD algorithm/config/input order change',
        'reason': 'Permit identical spellings for distinct entities only with disjoint exact contexts',
        'prior_source_sha256': sha(old_path), 'code_before_sha256': code_before,
        'code_after_sha256': code_after, 'existing_records_revalidated_unchanged': count,
        'existing_batch_artifacts_sha256': evidence, 'human_review_claimed': False})
    source['tool_code_sha256'] = code_after
    source['amendment_artifacts_sha256'] = {rel(path): sha(path) for path in archive.iterdir() if path.is_file()}
    source['amendment_note'] = 'Scoped homonyms supported; existing candidates/previews/reviews unchanged; see amendment evidence'
    atomic_json(current, source)
    print(json.dumps({'existing_records_unchanged': count, 'amendment': rel(proof)}))


if __name__ == '__main__':
    main()
