"""Versioned extraction jobs; no training, review approval or silent overwrite."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from opsd.dev_eval.prepare import ROOT
from opsd.full_run.utils import atomic_json, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--config', default='opsd/group05_deentity/config.toml')
    parser.add_argument('--model', default='/ai/yaodong/selector/models/Qwen3-4B')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--json-constrained', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--audit-after', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+', args.name) or (args.audit_after and args.limit is not None):
        raise ValueError('Invalid job name or diagnostic audit request')
    if not args.worker:
        log = ROOT / f'server-logs/group05-entity-{args.name}-controller.log'
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'opsd.group05_deentity.launch_job',
                *sys.argv[1:], '--worker'], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'log': str(log)}))
        return
    status_path = ROOT / 'opsd/group05_deentity/extraction-status.json'
    def status(stage, **fields):
        atomic_json(status_path, {'utc': utc(), 'stage': stage, 'job': args.name,
            'training_started': False, **fields})
    try:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('GPU occupied; refusing to compete: ' + active)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        command = [sys.executable, '-u', '-m', 'opsd.group05_deentity.extract', '--config', args.config,
                   '--model', args.model, '--execute']
        if args.limit is not None:
            command += ['--limit', str(args.limit)]
        if args.thinking:
            command += ['--thinking', '--max-new-tokens', '1024' if args.json_constrained else '4096']
        if args.json_constrained:
            command += ['--json-constrained', '--reasoning-budget', '256']
            if not args.thinking:
                command += ['--max-new-tokens', '1024']
        log = ROOT / f'server-logs/group05-entity-{args.name}-extracting.log'
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
            status('extracting_pilot' if args.limit else 'extracting', pid=child.pid,
                   config=args.config, expected_records=args.limit or 2000, log=str(log))
            code = child.wait()
        if code:
            raise RuntimeError('Extraction exited ' + str(code))
        if args.audit_after:
            status('building_audit', config=args.config)
            with (ROOT / f'server-logs/group05-entity-{args.name}-audit.log').open('x') as stream:
                subprocess.run([sys.executable, '-u', '-m', 'opsd.group05_deentity.audit', '--config', args.config],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status('pilot_complete' if args.limit else 'extraction_complete_review_required', config=args.config)
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
