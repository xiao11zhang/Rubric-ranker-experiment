from datetime import datetime, timezone
import json
from pathlib import Path
import random


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def order_for_epoch(count, seed):
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


def rng_state():
    import torch
    return {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def utc():
    return datetime.now(timezone.utc).isoformat()


def select_best(items):
    # SFT step-0 participates; exact ties prefer set F1, then the earlier step.
    return max(items, key=lambda r: (r['metrics']['balanced_macro']['all_annotated_supports_at_5'],
                                    r['metrics']['balanced_macro']['support_f1_selected'], -r['step']))
