"""Atomic adapter + optimizer + RNG + traversal cursor checkpoints."""
import json
from pathlib import Path
import time

from opsd.dev_eval.prepare import sha
from .utils import atomic_json, rng_state


def save_checkpoint(model, tokenizer, optimizer, root, step, plan):
    import torch
    root = Path(root)
    final = root / f'step-{step}'
    temporary = root / f'step-{step}.incomplete'
    if final.exists():
        raise FileExistsError(final)
    if temporary.exists():
        temporary.rename(root / f'step-{step}.incomplete-{time.time_ns()}')
    temporary.mkdir()
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    state = {'step': step, 'cursor': step * plan['config']['gradient_accumulation'],
             'config_sha256': plan['config_sha256'], 'order_ids': plan['order_ids'],
             'optimizer': optimizer.state_dict(), 'rng': rng_state()}
    torch.save(state, temporary / 'resume.pt')
    atomic_json(temporary / 'checkpoint.json', {'step': step, 'cursor': state['cursor'],
        'adapter_sha256': sha(temporary / 'adapter_model.safetensors'), 'resume_sha256': sha(temporary / 'resume.pt'),
        'config_sha256': plan['config_sha256']})
    temporary.rename(final)
    atomic_json(root / 'latest-checkpoint.json', {'step': step, 'path': str(final)})
    return final


def load_checkpoint(path, plan):
    import torch
    path = Path(path).resolve()
    record = json.loads((path / 'checkpoint.json').read_text())
    if sha(path / 'resume.pt') != record['resume_sha256'] or sha(path / 'adapter_model.safetensors') != record['adapter_sha256']:
        raise ValueError('Resume checkpoint integrity failure')
    state = torch.load(path / 'resume.pt', map_location='cpu', weights_only=False)
    if state['config_sha256'] != plan['config_sha256'] or state['order_ids'] != plan['order_ids']:
        raise ValueError('Resume configuration/permutation differs')
    if state['cursor'] != state['step'] * plan['config']['gradient_accumulation']:
        raise ValueError('Resume cursor is not an optimizer-step boundary')
    later = [p for p in path.parent.glob('step-*') if p.name[5:].isdigit() and int(p.name[5:]) > state['step']]
    if later:
        raise ValueError('Resume from the latest complete checkpoint only')
    return state


def reconcile_log(root, step):
    root = Path(root)
    path = root / 'train.jsonl'
    content = path.read_text(encoding='utf-8')
    kept = []
    for line in content.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row['step'] <= step:
            kept.append(row)
    if [r['step'] for r in kept] != list(range(1, step + 1)):
        raise ValueError('Training log does not cover checkpoint prefix exactly once')
    (root / f'pre-resume-train-{time.time_ns()}.jsonl').write_text(content, encoding='utf-8')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    temporary.replace(path)
