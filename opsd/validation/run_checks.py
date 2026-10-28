"""Paired validation of base / step-10 / step-20. No optimizer or backward."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import time

from opsd.common import encode_prompt, parse_output, read_rows
from opsd.objective import sampled_log_probs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(record, count):
    ids = record["ids"]
    return {"valid": record["valid"], "ended_eos": record["ended_eos"],
            "complete_valid": record["valid"] and record["ended_eos"],
            "duplicate": len(ids) != len(set(ids)),
            "incomplete_syntax": not bool(re.fullmatch(r"\s*(?:\[\d+\]\s*)+", record["output"])),
            "out_of_range": any(i < 1 or i > count for i in ids),
            "all_candidates": record["valid"] and len(ids) == count,
            "tokens": len(record["response_tokens"]),
            "selected": len(ids) if record["valid"] else 0}


def summarize(records, rows):
    by_id = {r["id"]: r for r in rows}
    details = [describe(r, len(by_id[r["id"]]["docs"])) for r in records]
    return {"queries": len(details), **{key: sum(d[key] for d in details) / len(details)
            for key in details[0]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--pi-queries", type=int, default=32)
    p.add_argument("--skip-pi", action="store_true", default=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model_path = "/ai/yaodong/selector/models/Qwen3-4B"
    data_path = root / "opsd/data/rr_rag/val.jsonl"
    run = root / "opsd/runs/server-pilot-001"
    protected = [root / "opsd" / name for name in ("train.py", "objective.py", "common.py", "predict.py")]
    protected += [run / f"step-{n}" / "adapter_model.safetensors" for n in (10, 20)]
    protected += [run / "config.json", data_path]
    before = {str(path): sha(path) for path in protected}
    if args.resume:
        previous = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if previous["protected_sha256_before"] != before or previous["batch_size"] != args.batch_size:
            raise ValueError("Protected inputs changed or resume batch differs")
    rows = read_rows(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    prompts = [encode_prompt(tokenizer, row) for row in rows]
    if max(map(len, prompts)) > 12000:
        raise ValueError("Validation prompt exceeds original limit")
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(base, str(run / "step-10"), adapter_name="step10", is_trainable=False)
    model.load_adapter(str(run / "step-20"), adapter_name="step20", is_trainable=False)
    model.eval()
    metadata = {"seed": 20261026, "queries": len(rows), "batch_size": args.batch_size,
                "max_new_tokens": 128, "sample_temperature": 1.0, "sample_top_p": 1.0,
                "sample_top_k": 0, "torch": torch.__version__, "transformers": transformers.__version__,
                "peft": peft.__version__, "protected_sha256_before": before,
                "validation_ids": [r["id"] for r in rows],
                "note": "No updates. Format diagnostics are not retrieval relevance metrics."}
    metadata["skip_pi"] = args.skip_pi
    (output / ("resume-metadata.json" if args.resume else "metadata.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    all_results, summary = {}, {}
    started = time.monotonic()
    for checkpoint in ("base", "step10", "step20"):
        if checkpoint != "base":
            model.set_adapter(checkpoint)
        for decoding in ("greedy", "sampled"):
            records = []
            filename = output / f"{checkpoint}-{decoding}.jsonl"
            if args.resume and filename.exists():
                lines = filename.read_text(encoding="utf-8").splitlines()
                for index, line in enumerate(lines):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        if index != len(lines) - 1:
                            raise
                keep = len(records) if len(records) == len(rows) else len(records) // args.batch_size * args.batch_size
                records = records[:keep]
                if any(r["id"] != row["id"] or r["candidate_hash"] != row["candidate_hash"]
                       for r, row in zip(records, rows)):
                    raise ValueError("Resume records do not match data prefix")
                shutil.copy2(filename, str(filename) + ".before-resume")
                filename.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            context = model.disable_adapter() if checkpoint == "base" else nullcontext()
            with context, torch.inference_mode(), filename.open("a" if args.resume else "x", encoding="utf-8") as stream:
                for i in range(len(records), len(rows), args.batch_size):
                    torch.manual_seed(20261026 + i)
                    group = prompts[i:i + args.batch_size]
                    width = max(map(len, group))
                    tokens = torch.tensor([[tokenizer.pad_token_id] * (width - len(s)) + s for s in group], device="cuda:0")
                    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in group], device="cuda:0")
                    kwargs = {"do_sample": decoding == "sampled", "max_new_tokens": 128,
                              "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                              "use_cache": True}
                    if decoding == "sampled":
                        kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
                    else:
                        kwargs.update(temperature=None, top_p=None, top_k=None)
                    generated = model.generate(input_ids=tokens, attention_mask=mask, **kwargs)
                    for row, seq in zip(rows[i:i + args.batch_size], generated[:, width:].tolist()):
                        ended = tokenizer.eos_token_id in seq
                        if ended:
                            seq = seq[:seq.index(tokenizer.eos_token_id) + 1]
                        text = tokenizer.decode(seq, skip_special_tokens=True)
                        record = {"id": row["id"], "candidate_hash": row["candidate_hash"],
                                  "output": text, "response_tokens": seq, "ended_eos": ended,
                                  **parse_output(text, len(row["docs"]))}
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    if (i + args.batch_size) % 32 == 0 or i + args.batch_size >= len(rows):
                        print(json.dumps({"phase": "generation", "checkpoint": checkpoint, "decoding": decoding,
                                          "done": len(records), "elapsed": time.monotonic() - started}), flush=True)
            key = checkpoint + "-" + decoding
            all_results[key] = records
            summary[key] = summarize(records, rows)
            (output / "generation-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Exact paired transitions on the same 160 validation queries.
    paired = {}
    for decoding in ("greedy", "sampled"):
        baseline = all_results["base-" + decoding]
        for checkpoint in ("step10", "step20"):
            current = all_results[checkpoint + "-" + decoding]
            paired[checkpoint + "-" + decoding] = {
                "valid_to_invalid": sum(a["valid"] and not b["valid"] for a, b in zip(baseline, current)),
                "invalid_to_valid": sum(not a["valid"] and b["valid"] for a, b in zip(baseline, current)),
                "identical_id_sequence": sum(a["ids"] == b["ids"] for a, b in zip(baseline, current))}
    # Frozen base teacher re-scores exactly the same original-student sampled tokens.
    pi_records = []
    chosen = [] if args.skip_pi else random.Random(20261026).sample(range(len(rows)), min(args.pi_queries, len(rows)))
    pi_stream = nullcontext(None) if args.skip_pi else (output / "teacher-pi.jsonl").open("x", encoding="utf-8")
    with model.disable_adapter(), torch.inference_mode(), pi_stream as stream:
        for i in chosen:
            row = rows[i]
            response = all_results["base-sampled"][i]["response_tokens"]
            logps = []
            for teacher in (False, True):
                prompt = encode_prompt(tokenizer, row, teacher=teacher)
                tokens = torch.tensor([prompt + response], device="cuda:0")
                logits = model(input_ids=tokens, use_cache=False).logits
                lp = sampled_log_probs(logits, tokens, len(prompt)).float().cpu()
                logps.append(lp)
                del logits
            delta = logps[1] - logps[0]
            record = {"id": row["id"], "student_output": all_results["base-sampled"][i]["output"],
                      "without_pi_logp": logps[0].tolist(), "with_pi_logp": logps[1].tolist(),
                      "mean_delta": delta.mean().item(), "mean_absolute_delta": delta.abs().mean().item(),
                      "max_absolute_delta": delta.abs().max().item()}
            pi_records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    after = {str(path): sha(path) for path in protected}
    if before != after:
        raise RuntimeError("Protected source/checkpoint/data files changed during validation")
    final = {"generation": summary, "paired": paired,
             "pi_probe": {"status": "skipped_at_user_request", "queries": 0} if args.skip_pi else {"queries": len(pi_records),
                          "mean_delta": sum(r["mean_delta"] for r in pi_records) / len(pi_records),
                          "mean_absolute_delta": sum(r["mean_absolute_delta"] for r in pi_records) / len(pi_records),
                          "changed_queries_gt_1e-3": sum(r["max_absolute_delta"] > 1e-3 for r in pi_records),
                          "note": "Sensitivity only; no quality or relevance conclusions."},
             "protected_files_unchanged": before == after, "elapsed_seconds": time.monotonic() - started}
    (output / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
