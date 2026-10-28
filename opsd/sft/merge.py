"""Merge the completed SFT adapter; export the original OPSD-compatible tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
base_path = "/ai/yaodong/selector/models/Qwen3-4B"
p = argparse.ArgumentParser()
p.add_argument("--adapter", default=str(root / "opsd/sft/runs/rr-qwen3-4b-lora-r16-001"))
p.add_argument("--output", default=str(root / "opsd/sft/models/rr-qwen3-4b-sft-r16-merged-001"))
args = p.parse_args()
adapter = Path(args.adapter)
target = Path(args.output)
if target.exists():
    raise FileExistsError(target)
if json.loads((adapter / "adapter_config.json").read_text())["r"] != 16:
    raise ValueError("Expected the requested r=16 SFT adapter")
state = json.loads((adapter / "trainer_state.json").read_text())
requested_epochs = float(json.loads((root / "opsd/sft/configs/lora.json").read_text())["num_train_epochs"])
if abs(float(state.get("epoch", 0)) - requested_epochs) > 0.01:
    raise ValueError(f"Expected an epoch-{requested_epochs} SFT checkpoint")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from opsd.common import encode_prompt, read_rows

original_tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
sft_tok = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
if original_tok.get_vocab() != sft_tok.get_vocab():
    raise ValueError("SFT changed tokenizer vocabulary")
base = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
model = model.merge_and_unload(safe_merge=True)
model.to(torch.bfloat16)
target.mkdir(parents=True)
model.save_pretrained(str(target), safe_serialization=True, max_shard_size="4GB")
# SFT token preflight proved this original chat template matches training source IDs.
original_tok.save_pretrained(str(target))
restored_tok = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
for row in read_rows(root / "opsd/data/rr_rag/val.jsonl"):
    if encode_prompt(original_tok, row) != encode_prompt(restored_tok, row):
        raise ValueError("Merged model changes OPSD input token IDs")
manifest = {"adapter": str(adapter), "sft_epochs": state["epoch"], "sft_steps": state["global_step"],
            "requested_sft_epochs": requested_epochs,
            "lora_rank": 16, "merged_dtype": "bfloat16", "tokenizer_vocab_unchanged": True,
            "all_val_prompt_tokens_unchanged": True,
            "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}}
(target / "merge-manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
