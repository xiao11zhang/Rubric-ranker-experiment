"""Bounded workflow: r16 LoRA SFT -> merge -> original OPSD 20 -> fixed validation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
status_path = root / "opsd/sft/pipeline-status.json"
if status_path.exists():
    raise FileExistsError(status_path)
meta = json.loads((root / "opsd/validation/results/server-002/metadata.json").read_text())
protected = meta["protected_sha256_before"]
if {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in protected} != protected:
    raise ValueError("Original OPSD experiment files changed")
smoke = root / "opsd/sft/runs/smoke-lora-r16-001"
state = json.loads((smoke / "trainer_state.json").read_text())
if state["global_step"] != 2 or not (smoke / "adapter_model.safetensors").is_file():
    raise ValueError("LoRA smoke checkpoint incomplete")
lora_config = json.loads((root / "opsd/sft/configs/lora.json").read_text())
if lora_config["finetuning_type"] != "lora" or lora_config["lora_rank"] != 16:
    raise ValueError("This workflow requires the requested r=16 LoRA SFT")
from safetensors.torch import load_file
smoke_weights = load_file(str(smoke / "adapter_model.safetensors"))
if not any(t.float().abs().max().item() > 0 for k, t in smoke_weights.items() if "lora_B" in k):
    raise ValueError("LoRA smoke did not update B weights")
del smoke_weights
history = []


def status(stage, state, **extra):
    record = {"stage": stage, "state": state, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    history.append(record)
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"current": record, "history": history}, indent=2))
    temporary.replace(status_path)
    print(json.dumps(record), flush=True)


commands = [
    ("lora_sft", [sys.executable, "-u", "-m", "opsd.sft.entry", "--config", str(root / "opsd/sft/configs/lora.json")]),
    ("merge_sft", [sys.executable, "-u", "-m", "opsd.sft.merge"]),
    ("opsd_20_steps", [sys.executable, "-u", "-m", "opsd.train", "--config", str(root / "opsd/sft/configs/opsd_after_sft.toml"), "--execute"]),
    ("format_comparison", [sys.executable, "-u", "-m", "opsd.sft.compare_formats"]),
]
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
           PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WANDB_DISABLED="true", MPLBACKEND="Agg")
try:
    for stage, command in commands:
        log = root / f"server-logs/sft-pipeline-{stage}-001.log"
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            status(stage, "running", pid=child.pid, log=str(log), command=command)
            code = child.wait()
        if code != 0:
            raise RuntimeError(f"{stage} exited with code {code}; see {log}")
        status(stage, "complete", log=str(log))
    if {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in protected} != protected:
        raise ValueError("Original experiment files changed")
    status("pipeline", "complete", report=str(root / "opsd/sft/results/post-sft-format-001/REPORT.md"))
except Exception as exc:
    status("pipeline", "failed", error=str(exc))
    raise
