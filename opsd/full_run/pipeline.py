import json
import os
import subprocess
import sys

from opsd.dev_eval.prepare import ROOT
from .utils import atomic_json, utc


def main():
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    status = ROOT / 'opsd/full_run/pipeline-status.json'
    try:
        for stage, module in (('preflight', 'opsd.full_run.prepare'), ('training', 'opsd.full_run.train'),
                              ('full_validation', 'opsd.full_run.full_validation')):
            log = ROOT / f'server-logs/rag-full-epoch-001-{stage}.log'
            with log.open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', module], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                atomic_json(status, {'utc': utc(), 'stage': stage, 'state': 'running', 'pid': child.pid, 'log': str(log)})
                code = child.wait()
            if code:
                raise RuntimeError(f'{stage} exited {code}: {log}')
        atomic_json(status, {'utc': utc(), 'stage': 'pipeline', 'state': 'complete'})
    except Exception as exc:
        atomic_json(status, {'utc': utc(), 'stage': 'pipeline', 'state': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
