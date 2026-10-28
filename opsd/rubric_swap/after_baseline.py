"""Wait for baseline validation/report, then classify queries only; never starts training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from opsd.dev_eval.prepare import ROOT, sha
from opsd.full_run.utils import atomic_json, utc


def main():
    status_path = ROOT / 'opsd/rubric_swap/classification-pipeline-status.json'
    def status(stage, **extra):
        value = {'utc': utc(), 'stage': stage, **extra}
        atomic_json(status_path, value)
        print(json.dumps(value), flush=True)
    baseline = ROOT / 'opsd/runs/rag-full-epoch-001'
    try:
        status('waiting_for_baseline_validation', gpu_work_started=False)
        while True:
            pipeline = json.loads((ROOT / 'opsd/full_run/pipeline-status.json').read_text())
            if pipeline['state'] == 'failed':
                raise RuntimeError('Baseline validation failed; classification not started')
            if pipeline['state'] == 'complete':
                break
            time.sleep(5)
        integrity = json.loads((baseline / 'integrity.json').read_text())
        if not integrity['original_algorithm_and_sft_weights_unchanged'] or integrity['final_test_accessed']:
            raise ValueError('Baseline integrity mismatch')
        if not (baseline / 'REPORT.md').is_file():
            raise ValueError('Baseline report not committed')
        summary = json.loads((baseline / 'full-validation-summary.json').read_text())
        if summary['evaluated_steps'] != [0, 300, 500]:
            raise ValueError('Unexpected completed baseline checkpoints')
        completed = {step: summary['results'][str(step)]['metrics']['balanced_macro'] for step in (0, 300, 500)}
        atomic_json(ROOT / 'opsd/rubric_swap/baseline-completed-summary.json', completed)
        guard = baseline / 'step-300/adapter_model.safetensors'
        guard_hash = sha(guard)
        status('baseline_complete', baseline_macro_metrics=completed, step300_adapter_sha256=guard_hash)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            status('waiting_for_free_gpu', gpu_work_started=False)
            time.sleep(5)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        tests = [sys.executable, '-m', 'unittest', 'opsd.rubric_swap.test_rubrics', 'opsd.rubric_swap.test_classify',
                 'opsd.rubric_swap.test_stop', 'opsd.full_run.test_execution', 'opsd.dev_eval.test_metrics']
        status('server_checks')
        subprocess.run(tests, cwd=ROOT, env=env, check=True)
        model = Path('/ai/yaodong/selector/models/Qwen3-4B')
        if not model.is_dir():
            raise FileNotFoundError(model)
        command = [sys.executable, '-u', '-m', 'opsd.rubric_swap.classify', '--model', str(model),
                   '--device', 'cuda:0', '--execute']
        with (ROOT / 'server-logs/rubric-swap-classify-001.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            status('classifying_queries', pid=child.pid, model=str(model), training_started=False)
            code = child.wait()
        if code:
            raise RuntimeError('Classification exited ' + str(code))
        if sha(guard) != guard_hash:
            raise ValueError('Protected baseline step-300 changed')
        status('classification_complete_review_required', training_started=False,
               labels='opsd/rubric_swap/data/domain-labels.jsonl')
    except Exception as exc:
        status('failed', error=str(exc), training_started=False)
        raise


if __name__ == '__main__':
    main()
