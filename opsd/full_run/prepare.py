"""Freeze full-epoch inputs and preflight all training/development prompt lengths."""
import json
from pathlib import Path

from opsd.common import digest, encode_prompt, read_rows, check_disjoint
from opsd.dev_eval.prepare import ROOT, sha, jsonl
from opsd.train import load_config
from .utils import atomic_json, order_for_epoch, utc

PLAN = ROOT / 'opsd/full_run/plan.json'
CONFIG = ROOT / 'opsd/full_run/config.toml'
FULL_DEV = ROOT / 'opsd/full_run/data/full-validation-v1.jsonl'


def main():
    if PLAN.exists():
        raise FileExistsError('Plan already frozen')
    c = load_config(CONFIG)
    original_config = load_config(ROOT / 'opsd/sft/configs/opsd_after_sft.toml')
    changed = {k for k in set(c) | set(original_config) if c.get(k) != original_config.get(k)}
    if changed != {'steps', 'save_every', 'output'}:
        raise ValueError('Unexpected OPSD hyperparameter changes: ' + str(changed))
    train = read_rows(ROOT / c['train_data'])
    val = read_rows(ROOT / c['val_data'])
    check_disjoint(train, val)
    if len(train) != 4000 or c['gradient_accumulation'] != 4 or c['steps'] != 1000:
        raise ValueError('Expected 4000 records / accumulation4 / 1000 steps')
    dev_root = ROOT / 'opsd/dev_eval/data/dev-v1'
    source_protocol = json.loads((dev_root / 'protocol.json').read_text())
    if sha(dev_root / 'queries.jsonl') != source_protocol['queries_sha256']:
        raise ValueError('Small development data changed')
    if any(sha(p) != h for p, h in source_protocol['input_sha256'].items()):
        raise ValueError('Audited training input changed')
    if any(sha(p) != h for p, h in source_protocol['protected_sha256'].items()):
        raise ValueError('Original OPSD code/config changed')
    full_paths = [dev_root / (name + '-eligible.jsonl') for name in ('hotpot', 'musique')]
    full = [r for path in full_paths for r in jsonl(path)]
    small = jsonl(dev_root / 'queries.jsonl')
    if len(full) != 9814 or len({r['id'] for r in full}) != len(full):
        raise ValueError('Full development query count/IDs differ')
    by_id = {r['id']: r for r in full}
    if any(by_id.get(r['id']) != r for r in small):
        raise ValueError('Frozen small-development records must be exact full-development subset')
    if any(r['split'] != 'validation' or r['candidate_hash'] != digest(r['docs']) for r in full):
        raise ValueError('Non-validation split or candidate hash mismatch')
    model_path = Path(c['model'])
    merge = json.loads((model_path / 'merge-manifest.json').read_text())
    if merge['sft_steps'] != 484 or merge['sft_epochs'] != 1:
        raise ValueError('Expected fixed one-epoch SFT initialization')
    for path, h in merge['files_sha256'].items():
        if sha(model_path / path) != h:
            raise ValueError('SFT model changed: ' + path)
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    maxima = {'student_train': 0, 'teacher_train': 0, 'full_development': 0}
    for i, row in enumerate(train):
        for teacher, key in ((False, 'student_train'), (True, 'teacher_train')):
            n = len(encode_prompt(tokenizer, row, teacher=teacher, mode=c['pi_mode'], top_k=c['top_k']))
            if n > c['max_prompt_tokens'] or n + c['max_response_tokens'] > model_config.max_position_embeddings:
                raise ValueError('Training prompt exceeds original limits: ' + row['id'])
            maxima[key] = max(maxima[key], n)
        if (i + 1) % 1000 == 0:
            print(json.dumps({'preflight': 'train', 'done': i + 1, 'total': len(train)}), flush=True)
    for i, row in enumerate(full):
        n = len(encode_prompt(tokenizer, row))
        if n + 128 > model_config.max_position_embeddings:
            raise ValueError('Development prompt exceeds model context: ' + row['id'])
        maxima['full_development'] = max(maxima['full_development'], n)
        if (i + 1) % 2000 == 0:
            print(json.dumps({'preflight': 'full_validation', 'done': i + 1, 'total': len(full)}), flush=True)
    FULL_DEV.parent.mkdir(parents=True, exist_ok=True)
    with FULL_DEV.open('x', encoding='utf-8') as stream:
        for row in full:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    paths = [CONFIG, ROOT / c['train_data'], ROOT / c['val_data'], dev_root / 'queries.jsonl',
             dev_root / 'protocol.json', FULL_DEV, model_path / 'merge-manifest.json', *full_paths]
    paths += list((ROOT / 'opsd/full_run').glob('*.py'))
    paths += [ROOT / 'opsd/dev_eval/metrics.py']
    order = order_for_epoch(len(train), c['seed'])
    plan = {'utc': utc(), 'config': c, 'config_sha256': sha(CONFIG), 'train_records': len(train),
        'steps': 1000, 'epochs': 1, 'sampling': 'fixed seeded shuffle without replacement, one rollout per record',
        'order_indices': order, 'order_ids': [train[i]['id'] for i in order],
        'small_dev_records': len(small), 'small_dev_path': str(dev_root / 'queries.jsonl'),
        'full_dev_records': len(full), 'full_dev_path': str(FULL_DEV),
        'maximum_prompt_tokens': maxima,
        'small_eval_steps': [0] + list(range(100, 1001, 100)),
        'checkpoint_steps': [10, 20] + list(range(100, 1001, 100)),
        'full_eval_steps': [0, 500, 1000, 'best_small_development'],
        'best_rule': 'balanced-macro all-support@5, then balanced-macro set F1, then earliest step',
        'evaluation_rng': 'Python/Torch CPU/all CUDA RNG and model train/eval mode restored after validation',
        'optimizer': 'same constant-lr AdamW; no new schedule, reward, ATO or GRPO',
        'original_protected_sha256': source_protocol['protected_sha256'],
        'input_and_execution_sha256': {str(p): sha(p) for p in paths},
        'sft_model_sha256': {str(model_path / p): h for p, h in merge['files_sha256'].items()},
        'final_test_accessed': False, 'generator_used': False,
        'full_dev_note': 'Full validation includes the 200 selection queries; development diagnostics, not an independent final test'}
    atomic_json(PLAN, plan)
    print(json.dumps({k: v for k, v in plan.items() if k not in ('order_ids', 'order_indices', 'sft_model_sha256', 'input_and_execution_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
