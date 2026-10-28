"""Launch RR-style SFT into a new directory; does not run OPSD."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("stage", choices=["smoke", "full", "preflight", "lora", "lora-smoke"])
p.add_argument("--run-id", default="001")
p.add_argument("--config")
args = p.parse_args()
if not args.run_id.isdigit():
    raise ValueError("run-id must be numeric")
root = Path(__file__).resolve().parents[2]
log = root / f"server-logs/rr-sft-{args.stage}-{args.run_id}.log"
command = [sys.executable, "-u", "-m", "opsd.sft.preflight", "--root", str(root)] if args.stage == "preflight" else [
    sys.executable, "-u", "-m", "opsd.sft.entry", "--config", args.config or str(root / f"opsd/sft/configs/{args.stage}.json")]
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
           TOKENIZERS_PARALLELISM="false", WANDB_DISABLED="true", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
with log.open("x") as stream:
    process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(root / f"server-logs/rr-sft-{args.stage}-{args.run_id}.pid").write_text(str(process.pid))
print(json.dumps({"pid": process.pid, "log": str(log), "command": command}))
