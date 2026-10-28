import subprocess
import sys
from opsd.dev_eval.prepare import ROOT

log = ROOT / 'server-logs/rag-full-epoch-001-skip1000.log'
with log.open('x') as stream:
    child = subprocess.Popen([sys.executable, '-u', '-m', 'opsd.full_run.skip_final_checkpoint'],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / 'server-logs/rag-full-epoch-001-skip1000.pid').write_text(str(child.pid))
print({'pid': child.pid, 'log': str(log)})
