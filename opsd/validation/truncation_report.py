"""Write a per-case diagnosis from the completed length-only counterfactual."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

CATEGORIES = {
    "duplicate_with_eos": "已 EOS，但重复 ID",
    "duplicate_before_cap": "截断前已重复，同时达到上限",
    "tail_syntax_only_at_cap": "达到上限，原前缀仅尾部语法不完整",
    "other": "其他",
}


def syntax_ok(record):
    return bool(re.fullmatch(r"\s*(?:\[\d+\]\s*)+", record["output"]))


def explanation(case):
    if case["category"] == "duplicate_with_eos":
        text = "已正常生成 EOS；非法来自重复编号，原输出上限不是其重复错误的原因。"
    elif case["category"] == "duplicate_before_cap":
        text = "达到 128-token 上限，但在此前已重复选择文档；保持前缀时，延长输出不能删除已有重复。"
    elif case["category"] == "tail_syntax_only_at_cap":
        text = f"原前缀有 {case['complete_id_count']} 个互异完整编号，尾部 {case['tail_after_last_complete_id']!r} 未完成；需要长度对照才能判断能否完整恢复。"
    else:
        text = "需要结合原始文本检查。"
    first = case["first_duplicate"]
    if first:
        text += f"首次重复发生在第 {first['position']} 个完整编号 [{first['id']}]。"
    block = case["repeated_block"]
    if block:
        text += f"存在连续重复片段 {block['block']}，连续 {block['repeats']} 次。"
    elif case["duplicate_counts"]:
        text += "主要表现为回头选择先前出现过的编号。"
    return text


def outcome(case):
    status = case["counterfactual_status"]
    if status == "not_rerun_already_ended_eos":
        return "未复跑：原输出已 EOS，错误是重复编号。"
    record = case["step20_extended"]
    tokens = len(record["response_tokens"])
    if not record["prefix_128_identical"]:
        return "前 128 tokens 不一致，不能归因于只改上限。"
    if status == "confirmed_cap_only_restored":
        return f"在 {tokens} tokens 正常 EOS，格式合法；确认本例可仅靠提高上限恢复。"
    if record["valid"]:
        return f"到 {tokens} tokens 格式合法，但仍未 EOS；不计作完整恢复。"
    end = "正常 EOS" if record["ended_eos"] else "仍未 EOS"
    reasons = []
    if case["extended_duplicate_counts"]:
        reasons.append("仍有重复编号")
    if not syntax_ok(record):
        reasons.append("仍有不完整/非法语法")
    if any(not 1 <= x <= case["candidate_count"] for x in record["ids"]):
        reasons.append("出现越界编号")
    return f"{tokens} tokens，{end}；{'、'.join(reasons)}，未恢复合法。"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    args = p.parse_args()
    root = Path(args.results)
    cases = json.loads((root / "cases-after.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    extended = [c for c in cases if c["suspected_truncation"]]
    comparable = [c for c in extended if c["step20_extended"]["prefix_128_identical"]]
    restored = [c for c in comparable if c["counterfactual_status"] == "confirmed_cap_only_restored"]
    strict = [c for c in cases if c["base_record"]["ended_eos"]]
    detail = {
        "case_records": len(cases), "unique_queries": len({c["id"] for c in cases}),
        "suspected_truncation_records": len(extended),
        "prefix_identical_records": len(comparable),
        "fully_restored_records": len(restored),
        "fully_restored_unique_queries": len({c["id"] for c in restored}),
        "restored_cases": [{"id": c["id"], "decoding": c["decoding"],
                            "extended_tokens": len(c["step20_extended"]["response_tokens"]),
                            "selected_ids": len(c["step20_extended"]["ids"]),
                            "candidate_count": c["candidate_count"]} for c in restored],
        "remaining_cases_with_duplicates": sum(
            c["counterfactual_status"] != "confirmed_cap_only_restored"
            and bool(c.get("extended_duplicate_counts", c["duplicate_counts"])) for c in cases),
        "initial_repetition_pattern_counts": dict(Counter(
            "single_id_repeat" if c["repeated_block"] and len(c["repeated_block"]["block"]) == 1
            else "multi_id_block_repeat" if c["repeated_block"]
            else "noncontiguous_reselection" if c["duplicate_counts"]
            else "no_complete_id_repeat_before_cap" for c in cases)),
        "syntax_fixed_but_still_invalid": sum(c["syntax_error"] and syntax_ok(c["step20_extended"])
                                             and not c["step20_extended"]["valid"] for c in comparable),
        "still_at_extended_cap": sum(not c["step20_extended"]["ended_eos"] for c in comparable),
        "tail_only_cases_developed_duplicates": sum(c["category"] == "tail_syntax_only_at_cap"
            and bool(c["extended_duplicate_counts"]) for c in comparable),
        "base_syntax_valid_but_no_eos": [{"id": c["id"], "decoding": c["decoding"]} for c in cases if not c["base_record"]["ended_eos"]],
        "strict_base_complete_records": len(strict),
        "strict_base_complete_restored": sum(c["base_record"]["ended_eos"] for c in restored),
        "remaining_unresolved_without_duplicate_or_range_error": [
            {"id": c["id"], "decoding": c["decoding"]} for c in comparable
            if not c["step20_extended"]["ended_eos"] and not c["extended_duplicate_counts"]
            and not any(not 1 <= i <= c["candidate_count"] for i in c["step20_extended"]["ids"])],
        "protected_files_unchanged": summary["protected_files_unchanged"],
    }
    lines = ["# Step-20 新增非法输出：逐条分类与截断对照", "",
        "范围：原始模型格式合法、step-20 格式非法的配对；greedy 与 sampled 分开计数。同一 Query 可在两种解码下各出现一次。", "",
        f"共 {len(cases)} 个配对、{detail['unique_queries']} 个不同 Query；疑似截断 {len(extended)} 个。仅将推理上限从 128 提高至 {summary['new_limit']} tokens。训练算法、配置、权重与 prompt 均保持不变。", "",
        "## 初始错误分类（互斥分类）", "", "| 分类 | Greedy | Sampled | 总计 |", "|---|---:|---:|---:|"]
    for category, title in CATEGORIES.items():
        counts = [sum(c["category"] == category and c["decoding"] == mode for c in cases) for mode in ("greedy", "sampled")]
        if sum(counts):
            lines.append(f"| {title} | {counts[0]} | {counts[1]} | {sum(counts)} |")
    lines += ["", "## 长度对照结论", "",
        f"{len(comparable)}/{len(extended)} 个对照的前 128 tokens 完全相同。只有前缀完全相同的对照用于归因。",
        f"仅延长上限后，{len(restored)} 个配对（{detail['fully_restored_unique_queries']} 个不同 Query）恢复格式合法且正常 EOS。",
        f"另有 {detail['syntax_fixed_but_still_invalid']} 个补齐了语法，但仍因其他格式错误非法；{detail['still_at_extended_cap']} 个在新上限仍未 EOS。以上数目可能重叠。",
        f"原先仅尾部不完整的样本中，{detail['tail_only_cases_developed_duplicates']} 个延长后出现重复，不能把它们算作纯截断修复。", "",
        "| 解码 | 原配对错误数 | 延长对照数 | 完整恢复数 |", "|---|---:|---:|---:|"]
    for mode in ("greedy", "sampled"):
        subset = [c for c in cases if c["decoding"] == mode]
        lines.append(f"| {mode} | {len(subset)} | {sum(c['suspected_truncation'] for c in subset)} | {sum(c['counterfactual_status'] == 'confirmed_cap_only_restored' for c in subset)} |")
    lines += ["", f"未完整恢复的 {len(cases) - len(restored)} 个配对中，{detail['remaining_cases_with_duplicates']} 个存在重复编号；保持这些已观察到的前缀继续生成，不能删除已有重复。",
        "", "完整恢复名单：", "", "| 解码 | Query ID | 结束 tokens | 选取数/候选数 |", "|---|---|---:|---:|"]
    for case in restored:
        record = case["step20_extended"]
        lines.append(f"| {case['decoding']} | {case['id']} | {len(record['response_tokens'])} | {len(record['ids'])}/{case['candidate_count']} |")
    lines += ["", f"初始重复模式细分：{detail['initial_repetition_pattern_counts']}。这些是输出形态分类，不是训练算法内部根因的证明。"]
    lines += ["", "## 基准口径与边界", "",
        f"原始模型有 {len(detail['base_syntax_valid_but_no_eos'])} 个配对仅格式合法、但也未 EOS，分别为 {detail['base_syntax_valid_but_no_eos']}。沿用上一轮格式合法定义，不把它们隐藏或剔除。",
        f"若严格要求原始模型也正常 EOS，则范围为 {len(strict)} 个配对，其中仅延长上限后完整恢复 {detail['strict_base_complete_restored']} 个。",
        "重复已出现在保留前缀中时，提高上限不能消除该重复。对延长后仍截断且没有不可逆错误的样本，不能排除更大上限的影响；请查看下列 unresolved 列表。",
        f"尚无重复/越界但仍未结束：{detail['remaining_unresolved_without_duplicate_or_range_error']}。",
        "本次只复跑疑似截断错误所在的原始完整批次，以保持 padding、随机种子和 RNG 形状；非目标样本仅作批次上下文。没有重算全部 160 条在新上限下的总体合法率。",
        "未做 Teacher 打分或相关性评价，恢复格式不等于排序质量改善。",
        f"核心文件与权重前后 hash 一致：{summary['protected_files_unchanged']}。", "", "## 逐条分析", ""]
    for index, case in enumerate(cases, 1):
        lines += [f"### {index}. {case['decoding']} / {case['id']}", "", case["query"], "",
            f"候选数：{case['candidate_count']}；原始模型 EOS：{case['base_record']['ended_eos']}；step-20 原输出长度：{len(case['step20_128']['response_tokens'])}。",
            "", "**初始分类：**" + CATEGORIES[case["category"]], "", explanation(case), "",
            "**对照结论：**" + outcome(case), "", "原始模型输出：", "```text", case["base_record"]["output"], "```", "",
            "step-20 / 128：", "```text", case["step20_128"]["output"], "```", ""]
        if case["suspected_truncation"]:
            lines += [f"step-20 / {summary['new_limit']}（前 128 tokens 相同：{case['step20_extended']['prefix_128_identical']}）：", "```text", case["step20_extended"]["output"], "```", ""]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "diagnosis-summary.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(detail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
