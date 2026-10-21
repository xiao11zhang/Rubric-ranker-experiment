from __future__ import annotations

import argparse
import json
from pathlib import Path

from setwise.config import assert_paired, load, validate
from setwise.data import prepare, read_jsonl, student_prompt, validate_row


def dump_toml(config):
    lines = []
    for section, fields in config.items():
        lines.append(f"[{section}]")
        for key, value in fields.items():
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Controlled setwise baselines; no training without --execute")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify-upstream", help="Verify vendored files against pinned Git blob hashes")
    check = sub.add_parser("check")
    check.add_argument("config")
    pair = sub.add_parser("compare")
    pair.add_argument("b1")
    pair.add_argument("b2")
    data = sub.add_parser("prepare")
    data.add_argument("source")
    data.add_argument("destination")
    data.add_argument("--expected-k", type=int)
    sft = sub.add_parser("export-sft")
    sft.add_argument("source")
    sft.add_argument("destination")
    train = sub.add_parser("train")
    train.add_argument("config")
    train.add_argument("--ato-root", required=True)
    train.add_argument("--execute", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("config")
    evaluate.add_argument("data")
    evaluate.add_argument("predictions", help="JSONL {id, output} from a frozen checkpoint")
    evaluate.add_argument("destination")
    evaluate.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    if args.action == "verify-upstream":
        from setwise.upstream import verify_sources
        manifest = verify_sources()
        print(json.dumps({"status": "verified", "file_count": len(manifest["files"]),
            "commits": sorted({f["commit"] for f in manifest["files"]})}, indent=2))
    elif args.action == "check":
        config = load(args.config)
        validate(config)
        print(json.dumps(config, indent=2))
    elif args.action == "compare":
        a, b = load(args.b1), load(args.b2)
        validate(a)
        validate(b)
        assert_paired(a, b)
        print("PASS: all shared settings match")
    elif args.action == "prepare":
        print(json.dumps(prepare(args.source, args.destination, args.expected_k), indent=2))
    elif args.action == "export-sft":
        output = []
        for row in read_jsonl(args.source):
            validate_row(row)
            if not row.get("silver_ids"):
                raise ValueError("Every SFT sample needs checked silver_ids")
            output.append({"messages": [*student_prompt(row), {"role": "assistant", "content": " ".join(f"[{i}]" for i in row["silver_ids"])}]})
        Path(args.destination).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.action == "train":
        from setwise.backend import launch
        print(json.dumps(launch(load(args.config), args.config, args.ato_root, args.execute), indent=2))
    elif args.action == "evaluate":
        from setwise.rewards import score_selection
        config = load(args.config)
        if config["reward"]["backend"] != "mock" and not args.allow_api:
            raise ValueError("Real Judge evaluation requires --allow-api")
        predictions = list(read_jsonl(args.predictions))
        pred = {p["id"]: p["output"] for p in predictions}
        rows = list(read_jsonl(args.data))
        if len(pred) != len(predictions) or set(pred) != {r["id"] for r in rows}:
            raise ValueError("Predictions must match dataset IDs exactly once")
        results = []
        for row in rows:
            results.append({"id": row["id"], **score_selection(pred[row["id"]],
                {"query_id": row["id"], "query": row["query"], "split": row["split"],
                 "initial_list": row["docs"], "rubrics": row["rubrics"],
                 "debug_gold_ids": row.get("debug_gold_ids", [])}, config)})
        if not results:
            raise ValueError("Empty evaluation")
        Path(args.destination).write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
        print(json.dumps({"count": len(results), "reward_mean": sum(r["score"] for r in results) / len(results),
                          "format_valid_rate": sum(r["format_valid"] for r in results) / len(results),
                          "mean_selected": sum(r["num_selected"] for r in results) / len(results)}))


if __name__ == "__main__":
    main()
