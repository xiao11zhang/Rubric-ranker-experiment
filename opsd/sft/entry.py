"""Call the pinned official RR LLaMA-Factory SFT implementation unchanged."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "opsd/sft/backend/src"))
import torch
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp

config = json.loads(Path(args.config).read_text(encoding="utf-8"))
output = Path(config["output_dir"])
if output.exists():
    raise FileExistsError("SFT output must be a new directory")


class ResourceLog(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        entry = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                 "trainable_dtypes": dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)),
                 "training_algorithm": f"official LLaMA-Factory {config['finetuning_type']} SFT cross entropy",
                 "lora_rank": config.get("lora_rank"), "lora_alpha": config.get("lora_alpha")}
        output.mkdir(parents=True, exist_ok=True)
        (output / "resource-metadata.json").write_text(json.dumps(entry, indent=2))
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = {"step": state.global_step, **(logs or {}),
                  "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                  "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        with (output / "resource-log.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


run_exp(args=config, callbacks=[ResourceLog()])
