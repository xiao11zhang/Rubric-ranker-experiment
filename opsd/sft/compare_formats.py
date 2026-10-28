"""Compare post-SFT OPSD steps 10/20 to the preserved previous experiment."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

from opsd.common import encode_prompt, parse_output, read_rows
from opsd.validation.run_checks import summarize


def main():
    root = Path(__file__).resolve().parents[2]
    old = root / "opsd/validation/results/server-002"
    old_summary = json.loads((old / "summary.json").read_text())
    old_meta = json.loads((old / "metadata.json").read_text())
    output = root / "opsd/sft/results/post-sft-format-001"
    output.mkdir(parents=True, exist_ok=False)
    protected = old_meta["protected_sha256_before"]
    before = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in protected}
    if before != protected:
        raise ValueError("Original algorithm, old checkpoints or validation data changed")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model_path = root / "opsd/sft/models/rr-qwen3-4b-sft-r16-merged-001"
    run = root / "opsd/runs/post-sft-opsd-pilot-001"
    rows = read_rows(root / "opsd/data/rr_rag/val.jsonl")
    if [r["id"] for r in rows] != old_meta["validation_ids"]:
        raise ValueError("Validation order changed")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    prompts = [encode_prompt(tokenizer, row) for row in rows]
    base = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(base, str(run / "step-10"), adapter_name="step10", is_trainable=False)
    model.load_adapter(str(run / "step-20"), adapter_name="step20", is_trainable=False)
    model.eval()
    metadata = {"validation_queries": len(rows), "validation_ids": [r["id"] for r in rows],
                "batch_size": old_meta["batch_size"], "seed": old_meta["seed"], "max_new_tokens": 128,
                "sample_temperature": 1.0, "sample_top_p": 1.0, "sample_top_k": 0,
                "initialization": str(model_path), "torch": torch.__version__, "transformers": transformers.__version__,
                "old_results": str(old), "protected_sha256_before": before, "teacher_scoring": "not_performed"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    results, metrics = {}, {}
    started = time.monotonic()
    batch_size = old_meta["batch_size"]
    for checkpoint in ("sft", "step10", "step20"):
        if checkpoint != "sft":
            model.set_adapter(checkpoint)
        for decoding in ("greedy", "sampled"):
            records = []
            context = model.disable_adapter() if checkpoint == "sft" else nullcontext()
            with context, torch.inference_mode(), (output / f"{checkpoint}-{decoding}.jsonl").open("x", encoding="utf-8") as stream:
                for start in range(0, len(rows), batch_size):
                    torch.manual_seed(old_meta["seed"] + start)
                    group = prompts[start:start + batch_size]
                    width = max(map(len, group))
                    tokens = torch.tensor([[tokenizer.pad_token_id] * (width - len(x)) + x for x in group], device="cuda:0")
                    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in group], device="cuda:0")
                    kwargs = {"do_sample": decoding == "sampled", "max_new_tokens": 128,
                              "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id, "use_cache": True}
                    kwargs.update(temperature=1.0, top_p=1.0, top_k=0) if decoding == "sampled" else kwargs.update(temperature=None, top_p=None, top_k=None)
                    generated = model.generate(input_ids=tokens, attention_mask=mask, **kwargs)
                    for row, seq in zip(rows[start:start + batch_size], generated[:, width:].tolist()):
                        ended = tokenizer.eos_token_id in seq
                        if ended:
                            seq = seq[:seq.index(tokenizer.eos_token_id) + 1]
                        text = tokenizer.decode(seq, skip_special_tokens=True)
                        record = {"id": row["id"], "candidate_hash": row["candidate_hash"], "output": text,
                                  "response_tokens": seq, "ended_eos": ended, **parse_output(text, len(row["docs"]))}
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    if (start + batch_size) % 32 == 0 or start + batch_size >= len(rows):
                        print(json.dumps({"checkpoint": checkpoint, "decoding": decoding, "done": len(records),
                                          "seconds": time.monotonic() - started}), flush=True)
            key = checkpoint + "-" + decoding
            results[key], metrics[key] = records, summarize(records, rows)
            (output / "generation-summary.json").write_text(json.dumps(metrics, indent=2))
    paired = {}
    for decoding in ("greedy", "sampled"):
        for checkpoint in ("step10", "step20"):
            source = [json.loads(s) for s in (old / f"{checkpoint}-{decoding}.jsonl").read_text(encoding="utf-8").splitlines()]
            new = results[checkpoint + "-" + decoding]
            if len(source) != len(rows) or any(a["id"] != b["id"] or a["candidate_hash"] != b["candidate_hash"]
                                               for a, b in zip(source, new)):
                raise ValueError("Previous/new predictions do not match query/candidate pairs")
            paired[checkpoint + "-" + decoding] = {
                "previous_invalid_now_valid": sum(not a["valid"] and b["valid"] for a, b in zip(source, new)),
                "previous_valid_now_invalid": sum(a["valid"] and not b["valid"] for a, b in zip(source, new))}
    read_train = lambda path: [sample["id"] for row in (json.loads(s) for s in Path(path).read_text().splitlines()) for sample in row["samples"]]
    same_training_queries = read_train(root / "opsd/runs/server-pilot-001/train.jsonl") == read_train(run / "train.jsonl")
    after = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in protected}
    if after != before:
        raise ValueError("Original protected files changed")
    summary = {"previous": old_summary["generation"], "after_sft": metrics, "paired_against_previous": paired,
               "training_query_sequence_matches_previous": same_training_queries,
               "original_algorithm_and_files_unchanged": True, "max_new_tokens": 128,
               "retrieval_quality_evaluated": False, "teacher_scoring": "not_performed"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# LoRA SFT r=16 后 OPSD step-10/20 格式对比", "",
             "固定 RR RAG val 160 条；同一 prompt、候选顺序、batch=16、种子和 128-token 上限。没有混用 256-token 截断对照。", "",
             "| 起点/阶段 | 解码 | 合法率 | EOS 结束率 | 重复 ID | 平均 tokens |", "|---|---|---:|---:|---:|---:|"]
    for label, group, prefix in (("原始4B", summary["previous"], "base"), ("之前OPSD-10", summary["previous"], "step10"),
            ("之前OPSD-20", summary["previous"], "step20"), ("LoRA SFT起点", metrics, "sft"),
            ("SFT后OPSD-10", metrics, "step10"), ("SFT后OPSD-20", metrics, "step20")):
        for decoding in ("greedy", "sampled"):
            value = group[prefix + "-" + decoding]
            lines.append(f"| {label} | {decoding} | {value['valid']:.3%} | {value['ended_eos']:.3%} | {value['duplicate']:.3%} | {value['tokens']:.2f} |")
    lines += ["", f"两轮 OPSD 训练 Query 采样顺序一致：{same_training_queries}。原 OPSD 核心源码、配置、旧 checkpoint 和验证数据 hash 保持一致。",
              "Teacher 与 Student 从同一合并后的 SFT checkpoint 初始化。未加入 GRPO、ATO，也未比较 Teacher 有/无 Rubric 打分。",
              "格式合法不代表检索相关性正确；本次未评估 qrels 或 RAG 答案质量，采样仅使用一组固定种子。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(root / "opsd/runs/post-sft-format-001/tensorboard")) as writer:
        for checkpoint, step in (("sft", 0), ("step10", 10), ("step20", 20)):
            for decoding in ("greedy", "sampled"):
                value = metrics[checkpoint + "-" + decoding]
                for key in ("valid", "ended_eos", "duplicate", "tokens"):
                    writer.add_scalar(f"validation/{decoding}/{key}", value[key], step)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
