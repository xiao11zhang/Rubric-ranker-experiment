import os
import subprocess
import sys
from opsd.dev_eval.prepare import ROOT

log = ROOT / 'server-logs/rag-full-epoch-001-controller.log'
with log.open('x') as stream:
    child = subprocess.Popen([sys.executable, '-u', '-m', 'opsd.full_run.pipeline'], cwd=ROOT,
        env=dict(os.environ, PYTHONUNBUFFERED='1'), stdin=subprocess.DEVNULL,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / 'server-logs/rag-full-epoch-001-controller.pid').write_text(str(child.pid))
print({'pid': child.pid, 'log': str(log)})
