"""Run extraction only after existing tasks finish; audit pending, never auto-train."""
import json
import os
from pathlib import Path
import subprocess
import sys

from opsd.dev_eval.prepare import ROOT
from opsd.full_run.utils import atomic_json, utc


def main():
    status_path = ROOT / 'opsd/group05_deentity/extraction-status.json'
    def status(stage, **fields):
        record = {'utc': utc(), 'stage': stage, **fields}
        atomic_json(status_path, record)
        print(json.dumps(record), flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1')
    try:
        previous = json.loads((ROOT / 'opsd/rubric_swap/pipeline-status.json').read_text())
        previous_report = json.loads((ROOT / 'opsd/runs/rag-rubric-same-domain-001/training-dev200-summary.json').read_text())
        if previous.get('state') != 'complete' or not previous_report.get('completed'):
            raise ValueError('Existing second-group tasks/report are not complete')
        if previous_report.get('full_development_evaluated'):
            raise ValueError('Second-group execution differs from the user-requested training-only scope')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('GPU is occupied; refusing to compete with current work: ' + active)
        model = Path('/ai/yaodong/selector/models/Qwen3-4B')
        if not model.is_dir():
            raise FileNotFoundError(model)
        commands = [
            ('server_checks', [sys.executable, '-m', 'unittest', 'opsd.group05_deentity.test_group05',
                               'opsd.full_run.test_execution', 'opsd.dev_eval.test_metrics']),
            ('extraction_preview', [sys.executable, '-u', '-m', 'opsd.group05_deentity.extract', '--model', str(model)]),
            ('extracting', [sys.executable, '-u', '-m', 'opsd.group05_deentity.extract', '--model', str(model), '--execute']),
            ('building_audit', [sys.executable, '-u', '-m', 'opsd.group05_deentity.audit']),
        ]
        for stage, command in commands:
            log = ROOT / f'server-logs/group05-entity-001-{stage}.log'
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                status(stage, pid=child.pid, log=str(log), extraction_model=str(model), training_started=False)
                code = child.wait()
            if code:
                raise RuntimeError(f'{stage} exited {code}: {log}')
        status('extraction_complete_review_required', training_started=False,
               candidates='opsd/group05_deentity/data/candidates.jsonl',
               audit='opsd/group05_deentity/reports/entity-audit.md')
    except Exception as exc:
        status('failed', error=str(exc), training_started=False)
        raise


if __name__ == '__main__':
    main()
