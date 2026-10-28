"""Group-local inputs and immutable baseline-prefix checks."""
import json
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from opsd.common import digest, read_rows
from opsd.dev_eval.prepare import ROOT, sha, jsonl


def verify(plan):
    for field in ('original_protected_sha256', 'input_and_execution_sha256', 'sft_model_sha256'):
        for path, expected in plan[field].items():
            if sha(path) != expected:
                raise ValueError('Frozen input/code/model changed: ' + path)


def protect(path, protected):
    path = Path(path).resolve()
    for item in protected:
        item = Path(item).resolve()
        if path == item or path in item.parents or item in path.parents:
            raise ValueError('Output overlaps a protected source: ' + str(item))


def load(config_path, check_hashes=True):
    c = tomllib.loads(Path(config_path).read_text(encoding='utf-8'))
    baseline = json.loads((ROOT / c['baseline_plan']).read_text(encoding='utf-8'))
    for key, value in baseline['config'].items():
        if key not in ('output', 'pi_mode') and c.get(key) != value:
            raise ValueError('Baseline training setting differs: ' + key)
    if c['pi_mode'] != 'deentity' or c['stop_after_step'] != 500:
        raise ValueError('Group 05 is frozen to deentity and 500 optimizer steps')
    if check_hashes:
        verify(baseline)
    rows = read_rows(ROOT / c['train_data'])
    order = baseline['order_indices']
    if (len(order) != len(rows) or set(order) != set(range(len(rows))) or
            [rows[i]['id'] for i in order] != baseline['order_ids']):
        raise ValueError('Baseline order is not the exact source-data permutation')
    n = c['stop_after_step'] * c['gradient_accumulation']
    selected = [rows[i] for i in order[:n]]
    if n != 2000 or len(selected) != n:
        raise ValueError('Expected the same first 2000 baseline records')
    baseline_root = ROOT / baseline['config']['output']
    protect(ROOT / c['output'], [baseline_root, c['model'], ROOT / c['train_data']])
    return c, baseline, rows, selected, baseline_root


def actual_baseline_prefix(baseline_root, baseline, c):
    entries = jsonl(baseline_root / 'train.jsonl')
    steps = c['stop_after_step']
    prefix = entries[:steps]
    if [e['step'] for e in prefix] != list(range(1, steps + 1)):
        raise ValueError('Baseline actual log does not cover steps 1..500')
    ids = [s['id'] for e in prefix for s in e['samples']]
    if (any(len(e['samples']) != c['gradient_accumulation'] for e in prefix) or
            ids != baseline['order_ids'][:2000]):
        raise ValueError('Baseline actual training differs from the planned 2000-record prefix')
    return digest(prefix)
