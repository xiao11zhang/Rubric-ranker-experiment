"""Configure the requested r=16 LoRA SFT and unchanged post-SFT OPSD."""
import json
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

root = Path(__file__).resolve().parents[2]
configs = root / "opsd/sft/configs"
reference = json.loads((configs / "full.json").read_text())
lora = dict(reference, finetuning_type="lora", lora_rank=16, lora_alpha=32, lora_target="all", lora_dropout=0.0,
            learning_rate=1e-4, num_train_epochs=1.0, output_dir=str(root / "opsd/sft/runs/rr-qwen3-4b-lora-r16-001"),
            logging_dir=str(root / "opsd/runs/rr-sft-lora-r16-001/tensorboard"))
smoke = dict(lora, dataset="rr_rag_smoke", max_steps=2, dataloader_num_workers=0, save_strategy="no", plot_loss=False,
             output_dir=str(root / "opsd/sft/runs/smoke-lora-r16-001"),
             logging_dir=str(root / "opsd/runs/rr-sft-lora-r16-smoke-001/tensorboard"))
for name, config in (("lora", lora), ("lora-smoke", smoke)):
    with (configs / f"{name}.json").open("x") as stream:
        json.dump(config, stream, indent=2)
merged = root / "opsd/sft/models/rr-qwen3-4b-sft-r16-merged-001"
old_path = root / "opsd/configs/server-pilot.toml"
old = tomllib.loads(old_path.read_text())
text = old_path.read_text().replace(old["model"], str(merged)).replace(old["output"], "opsd/runs/post-sft-opsd-pilot-001")
new = tomllib.loads(text)
if {k: v for k, v in old.items() if k not in {"model", "output"}} != {k: v for k, v in new.items() if k not in {"model", "output"}}:
    raise ValueError("Unexpected OPSD parameter change")
if new["steps"] != 20 or new["save_every"] != 10:
    raise ValueError("Expected one 20-step OPSD run saving steps 10/20")
with (configs / "opsd_after_sft.toml").open("x") as stream:
    stream.write(text)
print(json.dumps({"sft": lora, "opsd_changed_keys_only": ["model", "output"], "merged_checkpoint": str(merged)}, indent=2))
