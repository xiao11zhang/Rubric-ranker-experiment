"""Probe the pinned official RR SFT backend without loading a model."""
import importlib.util
import sys

sys.path.insert(0, "/ai/yaodong/selector/code/RubricRanker/LLaMA-Factory/src")
import torch
import transformers
import llamafactory

print("torch", torch.__version__, "transformers", transformers.__version__, "llamafactory", llamafactory.__file__, flush=True)
from llamafactory.train.tuner import run_exp
print("run_exp imported", flush=True)
print({x: bool(importlib.util.find_spec(x)) for x in ("deepspeed", "flash_attn", "liger_kernel")}, flush=True)
