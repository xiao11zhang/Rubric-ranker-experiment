"""Launch the authorized sequential workflow once, preserving all old runs."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
log = root / "server-logs/sft-pipeline-001.log"
with log.open("x") as stream:
    process = subprocess.Popen([sys.executable, "-u", "-m", "opsd.sft.pipeline"], cwd=root,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(root / "server-logs/sft-pipeline-001.pid").write_text(str(process.pid))
print(json.dumps({"pid": process.pid, "log": str(log), "status": str(root / "opsd/sft/pipeline-status.json")}))
