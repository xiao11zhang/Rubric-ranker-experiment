"""Read-only paired-error audit: the sole decoding change is 128 -> 256 tokens."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from opsd.common import encode_prompt, parse_output, read_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def pattern(ids):
    best = None
    for start in range(len(ids)):
        for width in range(1, (len(ids) - start) // 2 + 1):
            block = ids[start:start + width]
            repeats = 1
            while ids[start + repeats * width:start + (repeats + 1) * width] == block:
                repeats += 1
            if repeats > 1 and (best is None or width * repeats > best["coverage"]):
                best = {"start_id_position": start + 1, "block": block,
                        "repeats": repeats, "coverage": width * repeats}
    return best


def diagnose(record, candidate_count):
    ids = record["ids"]
    duplicates = {str(k): v for k, v in Counter(ids).items() if v > 1}
    out_of_range = sorted({i for i in ids if not 1 <= i <= candidate_count})
    syntax_error = not bool(re.fullmatch(r"\s*(?:\[\d+\]\s*)+", record["output"]))
    at_cap = not record["ended_eos"] and len(record["response_tokens"]) == 128
    seen, first_duplicate = set(), None
    for position, doc_id in enumerate(ids, 1):
        if doc_id in seen and first_duplicate is None:
            first_duplicate = {"position": position, "id": doc_id}
        seen.add(doc_id)
    matches = list(re.finditer(r"\[\d+\]", record["output"]))
    tail = record["output"][matches[-1].end():] if matches else record["output"]
    if duplicates and at_cap:
        category = "duplicate_before_cap"
    elif duplicates and record["ended_eos"]:
        category = "duplicate_with_eos"
    elif at_cap and syntax_error and not out_of_range:
        category = "tail_syntax_only_at_cap"
    else:
        category = "other"
    return {"category": category, "suspected_truncation": at_cap, "duplicate_counts": duplicates,
            "out_of_range_ids": out_of_range, "syntax_error": syntax_error,
            "repeated_block": pattern(ids), "first_duplicate": first_duplicate, "tail_after_last_complete_id": tail,
            "complete_id_count": len(ids), "unique_id_count": len(set(ids))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()
    if args.max_new_tokens <= 128:
        raise ValueError("The counterfactual output limit must exceed 128")
    root, source, output = Path(args.root).resolve(), Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original_meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    protected = original_meta["protected_sha256_before"]
    before = {path: sha(path) for path in protected}
    if before != protected:
        raise ValueError("Original protected source/weights/data changed")
    rows = read_rows(root / "opsd/data/rr_rag/val.jsonl")
    if [r["id"] for r in rows] != original_meta["validation_ids"]:
        raise ValueError("Validation order changed")
    batch_size = original_meta["batch_size"]
    old, cases = {}, []
    for decoding in ("greedy", "sampled"):
        base = read_jsonl(source / f"base-{decoding}.jsonl")
        old[decoding] = read_jsonl(source / f"step20-{decoding}.jsonl")
        for index, (a, b, row) in enumerate(zip(base, old[decoding], rows)):
            if a["id"] != row["id"] or b["id"] != row["id"]:
                raise ValueError("Paired IDs differ")
            if a["valid"] and not b["valid"]:
                cases.append({"id": row["id"], "query": row["query"], "decoding": decoding,
                    "validation_index": index, "batch_start": index // batch_size * batch_size,
                    "candidate_count": len(row["docs"]), "base_record": a, "step20_128": b,
                    **diagnose(b, len(row["docs"]))})
    (output / "cases-before.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    plan = {"paired_error_records": len(cases), "unique_queries": len({c["id"] for c in cases}),
            "suspected_truncation_records": sum(c["suspected_truncation"] for c in cases),
            "original_limit": 128, "new_limit": args.max_new_tokens, "batch_size": batch_size,
            "seed": original_meta["seed"], "protected_sha256_before": before,
            "source_prediction_sha256": {str(source / f"{checkpoint}-{mode}.jsonl"): sha(source / f"{checkpoint}-{mode}.jsonl")
                for checkpoint in ("base", "step20") for mode in ("greedy", "sampled")},
            "only_changed_decoding_parameter": "max_new_tokens",
            "note": "Full original batches preserve padding, sample RNG shapes and per-batch seeds."}
    (output / "metadata.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model_path = "/ai/yaodong/selector/models/Qwen3-4B"
    run = root / "opsd/runs/server-pilot-001"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(base_model, str(run / "step-10"), adapter_name="step10", is_trainable=False)
    model.load_adapter(str(run / "step-20"), adapter_name="step20", is_trainable=False)
    model.set_adapter("step20")
    model.eval()
    prompts = [encode_prompt(tokenizer, row) for row in rows]
    extended, finished_batches = {}, 0
    started = time.monotonic()
    with torch.inference_mode(), (output / "extended-targets.jsonl").open("x", encoding="utf-8") as stream:
        for decoding in ("greedy", "sampled"):
            targets = {c["validation_index"] for c in cases if c["decoding"] == decoding and c["suspected_truncation"]}
            batches = sorted({i // batch_size * batch_size for i in targets})
            for start in batches:
                torch.manual_seed(original_meta["seed"] + start)
                group = prompts[start:start + batch_size]
                width = max(map(len, group))
                tokens = torch.tensor([[tokenizer.pad_token_id] * (width - len(x)) + x for x in group], device="cuda:0")
                mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in group], device="cuda:0")
                kwargs = {"do_sample": decoding == "sampled", "max_new_tokens": args.max_new_tokens,
                          "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                          "use_cache": True}
                if decoding == "sampled":
                    kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
                else:
                    kwargs.update(temperature=None, top_p=None, top_k=None)
                generated = model.generate(input_ids=tokens, attention_mask=mask, **kwargs)
                for offset, seq in enumerate(generated[:, width:].tolist()):
                    index = start + offset
                    if index not in targets:
                        continue
                    ended = tokenizer.eos_token_id in seq
                    if ended:
                        seq = seq[:seq.index(tokenizer.eos_token_id) + 1]
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    previous = old[decoding][index]
                    record = {"id": rows[index]["id"], "decoding": decoding, "batch_start": start,
                              "output": text, "response_tokens": seq, "ended_eos": ended,
                              "prefix_128_identical": seq[:128] == previous["response_tokens"],
                              **parse_output(text, len(rows[index]["docs"]))}
                    extended[(decoding, record["id"])] = record
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                finished_batches += 1
                print(json.dumps({"finished_batches": finished_batches, "decoding": decoding,
                                  "batch_start": start, "extended_targets": len(extended),
                                  "seconds": time.monotonic() - started}), flush=True)
    totals = {}
    for decoding in ("greedy", "sampled"):
        subset = [c for c in cases if c["decoding"] == decoding]
        for case in subset:
            if not case["suspected_truncation"]:
                case["counterfactual_status"] = "not_rerun_already_ended_eos"
                continue
            new = extended[(decoding, case["id"])]
            case["step20_extended"] = new
            if not new["prefix_128_identical"]:
                case["counterfactual_status"] = "prefix_mismatch_not_attributable"
            elif new["valid"] and new["ended_eos"]:
                case["counterfactual_status"] = "confirmed_cap_only_restored"
            elif new["valid"]:
                case["counterfactual_status"] = "syntax_restored_but_still_no_eos"
            else:
                case["counterfactual_status"] = "still_invalid_after_extension"
            case["extended_duplicate_counts"] = {str(k): v for k, v in Counter(new["ids"]).items() if v > 1}
        totals[decoding] = {"paired_error_records": len(subset),
            "initial_categories": dict(Counter(c["category"] for c in subset)),
            "counterfactual_outcomes": dict(Counter(c["counterfactual_status"] for c in subset)),
            "extended_targets": sum(c["suspected_truncation"] for c in subset)}
    after = {path: sha(path) for path in protected}
    if after != before:
        raise RuntimeError("Protected files changed")
    (output / "cases-after.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"groups": totals, "paired_error_records": len(cases), "unique_queries": len({c["id"] for c in cases}),
        "all_extended_prefixes_identical": all(r["prefix_128_identical"] for r in extended.values()),
        "confirmed_cap_only_restored": sum(c["counterfactual_status"] == "confirmed_cap_only_restored" for c in cases),
        "unique_queries_restored": len({c["id"] for c in cases if c["counterfactual_status"] == "confirmed_cap_only_restored"}),
        "original_limit": 128, "new_limit": args.max_new_tokens,
        "protected_files_unchanged": before == after, "seconds": time.monotonic() - started,
        "scope": "Only base-valid/step20-invalid cases, not a full validation rerun at the extended limit."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
