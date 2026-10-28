"""Launch only frozen development inference, never training."""
import os
import subprocess
import sys
from .prepare import ROOT

log = ROOT / 'server-logs/dev-eval-inference-001.log'
with log.open('x') as stream:
    child = subprocess.Popen([sys.executable, '-u', '-m', 'opsd.dev_eval.run'], cwd=ROOT,
        env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false'),
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / 'server-logs/dev-eval-inference-001.pid').write_text(str(child.pid))
print({'pid': child.pid, 'log': str(log)})
