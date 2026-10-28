"""Authorized mixed-rubric run: checks -> real preflight -> 500 updates -> full dev."""
import json
import os
import subprocess
import sys

from opsd.dev_eval.prepare import ROOT
from opsd.full_run.utils import atomic_json, utc

PLAN = 'opsd/rubric_swap/plans/same-domain-001.json'


def main():
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    status = ROOT / 'opsd/rubric_swap/pipeline-status.json'
    commands = [
        ('tests', [sys.executable, '-m', 'unittest', 'opsd.rubric_swap.test_rubrics',
            'opsd.rubric_swap.test_classify', 'opsd.rubric_swap.test_stop', 'opsd.rubric_swap.test_mixed',
            'opsd.rubric_swap.test_evaluation', 'opsd.full_run.test_execution', 'opsd.dev_eval.test_metrics']),
        ('preflight', [sys.executable, '-u', '-m', 'opsd.rubric_swap.prepare', '--config',
            'opsd/rubric_swap/same-domain.toml', '--plan', PLAN]),
        ('training', [sys.executable, '-u', '-m', 'opsd.rubric_swap.train', '--plan', PLAN, '--execute']),
        ('full_development', [sys.executable, '-u', '-m', 'opsd.rubric_swap.full_validation',
            '--plan', PLAN, '--execute']),
    ]
    try:
        for stage, command in commands:
            log = ROOT / f'server-logs/rubric-mixed-001-{stage}.log'
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                atomic_json(status, {'utc': utc(), 'stage': stage, 'state': 'running',
                    'pid': child.pid, 'log': str(log), 'approved_stop_step': 500})
                code = child.wait()
            if code:
                raise RuntimeError(f'{stage} exited {code}: {log}')
            print(json.dumps({'utc': utc(), 'stage': stage, 'state': 'complete'}), flush=True)
        atomic_json(status, {'utc': utc(), 'stage': 'pipeline', 'state': 'complete',
            'run': 'opsd/runs/rag-rubric-same-domain-001', 'approved_stop_step': 500})
    except Exception as exc:
        atomic_json(status, {'utc': utc(), 'stage': 'pipeline', 'state': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
