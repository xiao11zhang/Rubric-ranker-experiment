"""Publish fixed-validation diagnostics as separate TensorBoard curves."""
import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--summary", required=True)
p.add_argument("--logdir", required=True)
args = p.parse_args()
target = Path(args.logdir)
if target.exists():
    raise FileExistsError(target)
summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
with SummaryWriter(str(target)) as writer:
    for checkpoint, step in (("base", 0), ("step10", 10), ("step20", 20)):
        for decoding in ("greedy", "sampled"):
            values = summary["generation"][checkpoint + "-" + decoding]
            for key in ("valid", "complete_valid", "ended_eos", "duplicate", "incomplete_syntax", "out_of_range", "tokens", "all_candidates"):
                writer.add_scalar(f"validation/{decoding}/{key}", values[key], step)
    writer.flush()
print(target.resolve())
