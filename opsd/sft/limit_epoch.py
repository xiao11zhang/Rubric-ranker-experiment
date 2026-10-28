"""Select the first complete epoch checkpoint of the live job, then run unchanged follow-ups.

The active optimizer/scheduler is preserved until the epoch boundary. No restart,
gradient changes or unsafe in-process injection is performed.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
status_path = root / "opsd/sft/pipeline-status.json"
initial = json.loads(status_path.read_text())
current = initial["current"]
if current["stage"] != "lora_sft" or current["state"] != "running":
    raise ValueError("Expected the original SFT stage to be running")
training_pid = int(current["pid"])
supervisor_pid = int((root / "server-logs/sft-pipeline-001.pid").read_text())
train_cmd = Path(f"/proc/{training_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
parent_cmd = Path(f"/proc/{supervisor_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
if "opsd.sft.entry" not in train_cmd or "opsd.sft.pipeline" not in parent_cmd:
    raise ValueError("Process identities differ; refusing to signal")
config_path = root / "opsd/sft/configs/lora.json"
config = json.loads(config_path.read_text())
run = Path(config["output_dir"])
original = config_path.with_name("lora.original-two-epoch.json")
with original.open("x") as stream:
    json.dump(config, stream, indent=2)
config["num_train_epochs"] = 1.0
config_path.write_text(json.dumps(config, indent=2))
os.kill(supervisor_pid, signal.SIGTERM)  # Only the old controller; the live SFT child continues.
history = initial["history"]


def update(stage, state, **extra):
    record = {"stage": stage, "state": state, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    history.append(record)
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"current": record, "history": history}, indent=2))
    temporary.replace(status_path)
    print(json.dumps(record), flush=True)


def alive(pid):
    try:
        return "\nState:\tZ" not in Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return False


env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
           PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WANDB_DISABLED="true", MPLBACKEND="Agg")
try:
    update("lora_sft", "running", pid=training_pid, log=current["log"], controller_pid=os.getpid(),
           requested_epochs=1, action="stop after first complete epoch checkpoint",
           scheduler_note="Existing two-epoch LR trajectory retained; active training was not restarted")
    checkpoint = None
    while checkpoint is None:
        for candidate in sorted(run.glob("checkpoint-*")):
            state_path = candidate / "trainer_state.json"
            adapter_path = candidate / "adapter_model.safetensors"
            if not state_path.is_file() or not adapter_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                continue
            if abs(float(state.get("epoch", 0)) - 1.0) < 0.01 and adapter_path.stat().st_size > 1_000_000:
                checkpoint = candidate
                break
        if checkpoint is None:
            if not alive(training_pid):
                raise RuntimeError("SFT exited before a complete first-epoch checkpoint was found")
            time.sleep(0.5)
    # All files needed by merge have been written before trainer_state.json.
    from safetensors import safe_open
    with safe_open(str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as stream:
        if not list(stream.keys()):
            raise ValueError("Empty epoch-one adapter")
    if alive(training_pid):
        os.kill(training_pid, signal.SIGTERM)
    for _ in range(60):
        if not alive(training_pid):
            break
        time.sleep(0.5)
    if alive(training_pid):
        raise RuntimeError("SFT process has not exited; refusing to overlap GPU stages")
    selection = {"requested_epochs": 1, "checkpoint": str(checkpoint), "trainer_epoch": state["epoch"],
                 "global_step": state["global_step"], "original_scheduler_total_steps": state.get("max_steps"),
                 "scheduler_note": "First epoch of the already-running two-epoch schedule; no state reset",
                 "adapter_sha256": hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()}
    (root / "opsd/sft/selected-epoch-one.json").write_text(json.dumps(selection, indent=2))
    update("lora_sft", "complete", **selection)
    commands = [
        ("merge_sft", [sys.executable, "-u", "-m", "opsd.sft.merge", "--adapter", str(checkpoint)]),
        ("opsd_20_steps", [sys.executable, "-u", "-m", "opsd.train", "--config", str(root / "opsd/sft/configs/opsd_after_sft.toml"), "--execute"]),
        ("format_comparison", [sys.executable, "-u", "-m", "opsd.sft.compare_formats"]),
    ]
    for stage, command in commands:
        log = root / f"server-logs/sft-pipeline-{stage}-001.log"
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            update(stage, "running", pid=child.pid, log=str(log), command=command)
            code = child.wait()
        if code != 0:
            raise RuntimeError(f"{stage} failed with code {code}: {log}")
        update(stage, "complete", log=str(log))
    update("pipeline", "complete", report=str(root / "opsd/sft/results/post-sft-format-001/REPORT.md"))
except Exception as exc:
    update("pipeline", "failed", error=str(exc))
    raise
