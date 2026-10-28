"""Check RR Qwen3 SFT encoding against existing OPSD, then create configs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
args = p.parse_args()
root = Path(args.root).resolve()
sft = root / "opsd/sft"
sys.path.insert(0, str(sft / "backend/src"))
from transformers import AutoTokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

model = "/ai/yaodong/selector/models/Qwen3-4B"
data_dir = sft / "data/rr_rag_v1"
records = json.loads((data_dir / "rr_rag_opsd_prompt.json").read_text(encoding="utf-8"))
raw_tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
train_tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
template = get_template_and_fix_tokenizer(train_tok, DataArguments(template="qwen3", enable_thinking=False, cutoff_len=16384))
lengths = []
for index, row in enumerate(records):
    msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["instruction"]}]
    opsd_ids = raw_tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    source, target = template.encode_oneturn(train_tok,
        [{"role": "user", "content": row["instruction"]}, {"role": "assistant", "content": row["output"]}], system=row["system"])
    if source != opsd_ids:
        raise ValueError(f"SFT/OPSD prompt token mismatch at row {index}")
    expected = train_tok.encode(row["output"], add_special_tokens=False)
    if target[:len(expected)] != expected or train_tok.eos_token_id not in target[len(expected):]:
        raise ValueError(f"SFT label/EOS mismatch at row {index}")
    if len(source) + len(target) > 16384:
        raise ValueError(f"SFT would truncate source/target at row {index}")
    lengths.append({"index": index, "prompt": len(source), "target": len(target), "total": len(source) + len(target)})
    if (index + 1) % 500 == 0:
        print(f"checked {index + 1}/{len(records)}", flush=True)
longest = sorted(lengths, key=lambda r: r["total"], reverse=True)[:8]
(data_dir / "rr_rag_smoke_longest.json").write_text(json.dumps([records[r["index"]] for r in longest], ensure_ascii=False, indent=2), encoding="utf-8")
registry = json.loads((data_dir / "dataset_info.json").read_text())
registry["rr_rag_smoke"] = {"file_name": "rr_rag_smoke_longest.json", "columns": registry["rr_rag_sft"]["columns"]}
(data_dir / "dataset_info.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
configs = sft / "configs"
configs.mkdir(exist_ok=True)
full = {"model_name_or_path": model, "trust_remote_code": True, "stage": "sft", "do_train": True,
        "finetuning_type": "full", "dataset": "rr_rag_sft", "dataset_dir": str(data_dir), "template": "qwen3",
        "enable_thinking": False, "cutoff_len": 16384, "train_on_prompt": False,
        "overwrite_cache": True, "preprocessing_num_workers": 4, "dataloader_num_workers": 4,
        "output_dir": str(sft / "runs/rr-qwen3-4b-full-001"), "logging_steps": 1, "logging_first_step": True,
        "plot_loss": True, "overwrite_output_dir": False, "save_only_model": True, "report_to": ["tensorboard"],
        "logging_dir": str(root / "opsd/runs/rr-sft-full-001/tensorboard"),
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 5e-6,
        "num_train_epochs": 2.0, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1, "bf16": True,
        "save_strategy": "epoch", "gradient_checkpointing": True, "flash_attn": "sdpa",
        "optim": "adamw_torch_fused", "seed": 42, "data_seed": 42, "disable_tqdm": True,
        "ddp_timeout": 180000000}
smoke = dict(full, dataset="rr_rag_smoke", max_steps=2, dataloader_num_workers=0,
             output_dir=str(sft / "runs/smoke-full-001"), save_strategy="no", plot_loss=False,
             logging_dir=str(root / "opsd/runs/rr-sft-smoke-001/tensorboard"))
for name, config in (("full", full), ("smoke", smoke)):
    config_path = configs / f"{name}.json"
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
report = {"records": len(records), "all_prompt_prefixes_equal_to_existing_opsd": True,
          "all_response_labels_preserved_with_eos": True, "all_fit_cutoff_16384": True,
          "max_prompt_tokens": max(r["prompt"] for r in lengths), "max_target_tokens": max(r["target"] for r in lengths),
          "max_total_tokens": max(r["total"] for r in lengths), "longest_smoke_records": longest,
          "dataset_sha256": hashlib.sha256((data_dir / "rr_rag_opsd_prompt.json").read_bytes()).hexdigest(),
          "backend_commit": "f464f083bd54cf6b1464a4347d7e9811a2c6de50"}
(sft / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
