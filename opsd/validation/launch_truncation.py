"""Launch the inference-only length-limit counterfactual."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
log = root / "server-logs/truncation-001.log"
command = [sys.executable, "-u", "-m", "opsd.validation.truncation_audit", "--root", str(root),
           "--source", str(root / "opsd/validation/results/server-002"),
           "--output", str(root / "opsd/validation/results/truncation-001"), "--max-new-tokens", "256"]
with log.open("x", encoding="utf-8") as stream:
    process = subprocess.Popen(command, cwd=root,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1"),
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(root / "server-logs/truncation-001.pid").write_text(str(process.pid))
print(json.dumps({"pid": process.pid, "log": str(log), "command": command}))
