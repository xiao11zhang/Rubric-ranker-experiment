"""Launch the one-epoch boundary controller without restarting the live SFT."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
log = root / "server-logs/sft-epoch-one-controller-001.log"
with log.open("x") as stream:
    process = subprocess.Popen([sys.executable, "-u", "-m", "opsd.sft.limit_epoch"], cwd=root,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(root / "server-logs/sft-epoch-one-controller-001.pid").write_text(str(process.pid))
print(json.dumps({"pid": process.pid, "log": str(log)}))
