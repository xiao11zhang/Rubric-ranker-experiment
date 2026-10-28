"""Greedy development inference, with training RNG and mode restored afterward."""
from collections import defaultdict
import json
from pathlib import Path
import time

from opsd.common import encode_prompt
from opsd.dev_eval.metrics import score, aggregate
from opsd.dev_eval.prepare import sha
from .utils import atomic_json, rng_state, restore_rng, utc


def evaluate(model, tokenizer, rows, destination, *, model_key, progress=None):
    import torch
    destination = Path(destination)
    if (destination / 'summary.json').exists():
        result = json.loads((destination / 'summary.json').read_text())
        if result['model_key'] != model_key or result['query_ids'] != [r['id'] for r in rows]:
            raise ValueError('Existing evaluation belongs to a different model or dataset')
        return result
    if destination.exists():
        # Preserve incomplete inference attempts; never treat a partial file as complete.
        destination.rename(destination.with_name(destination.name + '-incomplete-' + str(time.time_ns())))
    destination.mkdir(parents=True)
    state, was_training = rng_state(), model.training
    records, scores = [], defaultdict(list)
    try:
        model.eval()
        with torch.inference_mode(), (destination / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
            for start in range(0, len(rows), 8):
                group = rows[start:start + 8]
                prompts = [encode_prompt(tokenizer, row) for row in group]
                width = max(map(len, prompts))
                if width + 128 > model.config.max_position_embeddings:
                    raise ValueError('Development input exceeds context; no truncation performed')
                ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p for p in prompts], device=model.device)
                mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=model.device)
                generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                    temperature=None, top_p=None, top_k=None, max_new_tokens=128,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
                for row, seq in zip(group, generated[:, width:].tolist()):
                    ended = tokenizer.eos_token_id in seq
                    if ended:
                        seq = seq[:seq.index(tokenizer.eos_token_id) + 1]
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    value = score(row, text)
                    value.update(ended_eos=float(ended), response_tokens=len(seq))
                    scores[row['dataset']].append(value)
                    scores['pooled'].append(value)
                    record = {'id': row['id'], 'candidate_hash': row['candidate_hash'], 'output': text,
                              'ended_eos': ended, 'response_tokens': len(seq), 'metrics': value}
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                if progress and (len(records) % 200 == 0 or len(records) == len(rows)):
                    progress(len(records), len(rows))
        metrics = {name: aggregate(values) for name, values in scores.items()}
        datasets = sorted(set(r['dataset'] for r in rows))
        metrics['balanced_macro'] = {k: sum(metrics[name][k] for name in datasets) / len(datasets)
            for k in metrics[datasets[0]] if not k.endswith('_queries') and k != 'queries'
            and all(metrics[name][k] is not None for name in datasets)}
        result = {'utc': utc(), 'model_key': model_key, 'query_ids': [r['id'] for r in rows],
                  'metrics': metrics, 'prediction_sha256': sha(destination / 'predictions.jsonl'),
                  'protocol': 'frozen dev-v1; greedy; batch8; max_new_tokens128; strict invalid-as-empty',
                  'final_test_accessed': False, 'generator_used': False}
        atomic_json(destination / 'summary.json', result)
        return result
    finally:
        model.train(was_training)
        restore_rng(state)
