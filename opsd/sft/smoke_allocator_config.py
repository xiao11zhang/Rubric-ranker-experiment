"""Make a new output-only smoke config; keep all training parameters."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "opsd/sft/configs/smoke.json"
config = json.loads(path.read_text())
config["output_dir"] = str(root / "opsd/sft/runs/smoke-full-002")
config["logging_dir"] = str(root / "opsd/runs/rr-sft-smoke-002/tensorboard")
target = path.with_name("smoke-allocator.json")
with target.open("x") as stream:
    json.dump(config, stream, indent=2)
print(target)
