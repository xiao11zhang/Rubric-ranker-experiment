"""Single-process, one frozen teacher and one trainable student. No ATO/GRPO."""
import argparse
import hashlib
import json
import random
import time
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

from .common import check_disjoint, encode_prompt, parse_output, read_rows
from .objective import loss, sampled_log_probs


def load_config(path):
    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("steps", "gradient_accumulation", "max_prompt_tokens", "max_response_tokens", "save_every"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["dtype"] not in {"bfloat16", "float32"} or config["pi_mode"] not in {"correct", "none"}:
        raise ValueError("Unsupported dtype or PI mode")
    if config["top_k"] < 0 or config["learning_rate"] <= 0:
        raise ValueError("Invalid top_k or learning_rate")
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--execute", action="store_true", help="Actually load models and update parameters")
    args = p.parse_args()
    c = load_config(args.config)
    train, val = read_rows(c["train_data"]), read_rows(c["val_data"])
    check_disjoint(train, val)
    if any(c["top_k"] > len(r["docs"]) for r in train + val):
        raise ValueError("top_k exceeds a candidate pool")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "train": len(train), "val": len(val), "config": c}, indent=2))
        return
    root = Path(c["output"])
    if root.exists():
        raise FileExistsError("Output must not exist; resuming is not implemented")
    import torch
    import transformers
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["revision"])
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(c["model"], revision=c["revision"])
    # Preflight all lengths: no silent filtering/truncation or paid/model training yet.
    lengths = []
    for row in train + val:
        pair = [len(encode_prompt(tokenizer, row, teacher=t, mode=c["pi_mode"], top_k=c["top_k"]))
                for t in (False, True)]
        if max(pair) > c["max_prompt_tokens"]:
            raise ValueError(f"Prompt too long for {row['id']}: {pair}; increase budget explicitly")
        if max(pair) + c["max_response_tokens"] > model_config.max_position_embeddings:
            raise ValueError(f"Prompt plus response exceeds model context for {row['id']}")
        lengths.append(pair)
    dtype = getattr(torch, c["dtype"])
    kwargs = dict(revision=c["revision"], torch_dtype=dtype, attn_implementation="sdpa")
    student = AutoModelForCausalLM.from_pretrained(c["model"], **kwargs).to(c["student_device"])
    teacher = AutoModelForCausalLM.from_pretrained(c["model"], **kwargs).to(c["teacher_device"])
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    if c["use_lora"]:
        from peft import LoraConfig, get_peft_model
        student = get_peft_model(student, LoraConfig(r=c["lora_rank"], lora_alpha=c["lora_alpha"],
            lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
        student.enable_input_require_grads()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Dropout-free training preserves sampling probabilities and enables checkpointing.
    for module in student.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=c["learning_rate"])
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    metadata = {"torch": torch.__version__, "transformers": transformers.__version__,
                "model_commit": getattr(student.config, "_commit_hash", None),
                "teacher_commit": getattr(teacher.config, "_commit_hash", None),
                "max_student_teacher_prompt_lengths": [max(x[i] for x in lengths) for i in (0, 1)],
                "data_sha256": {k: hashlib.sha256(Path(c[k]).read_bytes()).hexdigest() for k in ("train_data", "val_data")}}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    writer = SummaryWriter(log_dir=str(root / "tensorboard"))
    start = time.monotonic()
    for step in range(1, c["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        records = []
        for _ in range(c["gradient_accumulation"]):
            row = random.choice(train)
            sp = encode_prompt(tokenizer, row, top_k=c["top_k"])
            tp = encode_prompt(tokenizer, row, teacher=True, mode=c["pi_mode"], top_k=c["top_k"])
            s_input = torch.tensor([sp], device=c["student_device"])
            student.eval()
            with torch.no_grad():
                generated = student.generate(input_ids=s_input, attention_mask=torch.ones_like(s_input),
                    do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                    max_new_tokens=c["max_response_tokens"], pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, use_cache=True)
            student.train()
            response = generated[0, len(sp):].tolist()
            if not response:
                raise RuntimeError("Empty response")
            t_input = torch.tensor([tp + response], device=c["teacher_device"])
            with torch.no_grad():
                tlogits = teacher(input_ids=t_input, use_cache=False).logits
                tlp = sampled_log_probs(tlogits, t_input, len(tp)).to(c["student_device"])
                del tlogits
            slogits = student(input_ids=generated, use_cache=False).logits
            slp = sampled_log_probs(slogits, generated, len(sp))
            objective, adv = loss(slp, tlp)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite distillation loss")
            (objective / c["gradient_accumulation"]).backward()
            text = tokenizer.decode(response, skip_special_tokens=True)
            records.append({"id": row["id"], "output": text,
                            "valid": parse_output(text, len(row["docs"]), c["top_k"])["valid"],
                            "tokens": len(response), "ended_eos": response[-1] == tokenizer.eos_token_id,
                            "loss": objective.item(), "mean_logp_gap": adv.mean().item()})
            del slogits, slp, tlp, objective, adv
        grad_norm = torch.nn.utils.clip_grad_norm_(params, c["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        entry = {"step": step, "seconds": time.monotonic() - start, "grad_norm": float(grad_norm), "samples": records}
        with (root / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for tag, value in {
            "train/loss": sum(r["loss"] for r in records) / len(records),
            "train/teacher_student_logp_gap": sum(r["mean_logp_gap"] for r in records) / len(records),
            "train/valid_fraction": sum(r["valid"] for r in records) / len(records),
            "train/eos_fraction": sum(r["ended_eos"] for r in records) / len(records),
            "train/mean_response_tokens": sum(r["tokens"] for r in records) / len(records),
            "train/grad_norm": float(grad_norm),
            "train/elapsed_seconds": entry["seconds"],
        }.items():
            writer.add_scalar(tag, value, step)
        writer.flush()
        print(json.dumps({"step": step, "loss": sum(r["loss"] for r in records) / len(records),
                          "valid_fraction": sum(r["valid"] for r in records) / len(records)}), flush=True)
        if step % c["save_every"] == 0 or step == c["steps"]:
            destination = root / f"step-{step}"
            student.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
    # Checkpoints are inference/continued-initialization artifacts, not exact optimizer resumes.
    writer.close()


if __name__ == "__main__":
    main()
