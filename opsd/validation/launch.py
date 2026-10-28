"""Launch bounded inference diagnostics in the background on the server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--run-id", default="001")
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--resume", action="store_true")
p.add_argument("--skip-pi", action="store_true", default=True)
args = p.parse_args()
if not args.run_id.isdigit() or args.batch_size < 1:
    raise ValueError("Invalid run ID or batch size")
suffix = "-resume" if args.resume else ""
log = root / f"server-logs/validation-{args.run_id}{suffix}.log"
command = [sys.executable, "-u", "-m", "opsd.validation.run_checks", "--root", str(root),
           "--output", str(root / f"opsd/validation/results/server-{args.run_id}"), "--batch-size", str(args.batch_size), "--pi-queries", "32"]
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
if args.resume:
    command.append("--resume")
if args.skip_pi:
    command.append("--skip-pi")
with log.open("x", encoding="utf-8") as stream:
    process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(root / f"server-logs/validation-{args.run_id}{suffix}.pid").write_text(str(process.pid))
print(json.dumps({"pid": process.pid, "log": str(log), "command": command}))
