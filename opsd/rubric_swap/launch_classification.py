import subprocess
import sys
from opsd.dev_eval.prepare import ROOT

log = ROOT / 'server-logs/rubric-swap-after-baseline-001.log'
with log.open('x') as stream:
    child = subprocess.Popen([sys.executable, '-u', '-m', 'opsd.rubric_swap.after_baseline'], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / 'server-logs/rubric-swap-after-baseline-001.pid').write_text(str(child.pid))
print({'pid': child.pid, 'log': str(log)})
