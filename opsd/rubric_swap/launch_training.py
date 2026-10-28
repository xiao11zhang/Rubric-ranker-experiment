"""Launch only after explicit execution authorization; never overwrite an old run."""
import subprocess
import sys
from opsd.dev_eval.prepare import ROOT

log = ROOT / 'server-logs/rubric-mixed-001-controller.log'
with log.open('x') as stream:
    child = subprocess.Popen([sys.executable, '-u', '-m', 'opsd.rubric_swap.pipeline'], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / 'server-logs/rubric-mixed-001-controller.pid').write_text(str(child.pid))
print({'pid': child.pid, 'log': str(log)})
