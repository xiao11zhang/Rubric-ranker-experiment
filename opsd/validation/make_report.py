"""Create a human-readable report from completed inference diagnostics."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--data", required=True)
args = p.parse_args()
root = Path(args.results)
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
data = {r["id"]: r for r in map(json.loads, Path(args.data).read_text(encoding="utf-8").splitlines())}
lines = ["# 固定验证集检查结果", "", "2026-10-26；官方 RR RAG val 全部 160 条，三个 checkpoint 使用相同 prompt、候选顺序、解码设置与输出上限。仅推理，无参数更新。", "",
         "| Checkpoint | 解码 | 格式合法率 | EOS 结束率 | 重复 ID 比例 | 平均输出 tokens |",
         "|---|---|---:|---:|---:|---:|"]
for checkpoint in ("base", "step10", "step20"):
    for decoding in ("greedy", "sampled"):
        v = summary["generation"][checkpoint + "-" + decoding]
        lines.append(f"| {checkpoint} | {decoding} | {v['valid']:.2%} | {v['ended_eos']:.2%} | {v['duplicate']:.2%} | {v['tokens']:.2f} |")
lines += ["", "## 与原始模型的逐题配对", "", "| Checkpoint/解码 | 合法→非法 | 非法→合法 | ID 顺序完全相同 |", "|---|---:|---:|---:|"]
for key, v in summary["paired"].items():
    lines.append(f"| {key} | {v['valid_to_invalid']} | {v['invalid_to_valid']} | {v['identical_id_sequence']} |")
lines += ["", "## 结论边界", "",
          "格式合法不等于检索相关性正确；EOS 与格式分开统计。未正常 EOS 结束的输出达到 128-token 上限。",
          "采样仅采用一组固定随机种子，不能把小幅差异宣称为统计显著。没有 qrels，因此没有评估 nDCG、Recall 或 RAG 答案质量。",
          "Teacher 有/无 Rubric 的打分比较已按用户要求取消，没有执行。",
          f"核心源码、训练配置、验证数据与 checkpoint 文件的前后 SHA-256 核对：{summary['protected_files_unchanged']}。",
          "验证结果只用于定位，不自动更改算法、解码参数、训练配置或继续训练。", "",
          "原始输出见同目录六个 checkpoint-decoding.jsonl；逐题失败示例见 paired-errors.jsonl。"]
with (root / "REPORT.md").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
with (root / "paired-errors.jsonl").open("x", encoding="utf-8") as stream:
    for decoding in ("greedy", "sampled"):
        baseline = {r["id"]: r for r in map(json.loads, (root / f"base-{decoding}.jsonl").read_text(encoding="utf-8").splitlines())}
        for checkpoint in ("step10", "step20"):
            rows = map(json.loads, (root / f"{checkpoint}-{decoding}.jsonl").read_text(encoding="utf-8").splitlines())
            for row in rows:
                old = baseline[row["id"]]
                if old["valid"] and not row["valid"]:
                    record = {"id": row["id"], "query": data[row["id"]]["query"], "checkpoint": checkpoint,
                              "decoding": decoding, "base_output": old["output"], "checkpoint_output": row["output"],
                              "base_ended_eos": old["ended_eos"], "checkpoint_ended_eos": row["ended_eos"]}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
print(root / "REPORT.md")
