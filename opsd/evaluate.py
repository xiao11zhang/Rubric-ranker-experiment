"""Format diagnostics; optional ranking metrics ONLY against supplied qrels."""
import argparse
import json
import math
from pathlib import Path

from .common import parse_output, read_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--metric-k", type=int, default=5)
    p.add_argument("--qrels", help='Optional JSONL: {"id": ..., "candidate_hash": ..., "grades": {"1": 2, "3": 1}}')
    args = p.parse_args()
    if args.metric_k <= 0:
        raise ValueError("metric-k must be positive")
    rows = read_rows(args.data)
    predictions = [json.loads(s) for s in Path(args.predictions).read_text(encoding="utf-8").splitlines() if s.strip()]
    pred = {r["id"]: r for r in predictions}
    ids = {r["id"] for r in rows}
    if len(pred) != len(predictions) or set(pred) != ids:
        raise ValueError("Prediction IDs must match data exactly once")
    labels = None
    if args.qrels:
        source = [json.loads(s) for s in Path(args.qrels).read_text(encoding="utf-8").splitlines() if s.strip()]
        labels = {r["id"]: r for r in source}
        if len(labels) != len(source) or set(labels) != ids:
            raise ValueError("qrels must cover the data exactly once")
    valid, counts, ndcg, recall = [], [], [], []
    for row in rows:
        prediction = pred[row["id"]]
        if prediction["candidate_hash"] != row["candidate_hash"]:
            raise ValueError("Prediction candidates do not match")
        parsed = parse_output(prediction["output"], len(row["docs"]), args.top_k)
        valid.append(parsed["valid"])
        counts.append(len(parsed["ids"]) if parsed["valid"] else 0)
        if labels is None:
            continue
        label = labels[row["id"]]
        if label["candidate_hash"] != row["candidate_hash"]:
            raise ValueError("qrels candidates do not match")
        grades = {int(k): float(v) for k, v in label["grades"].items()}
        if any(not 1 <= k <= len(row["docs"]) or not math.isfinite(v) or v < 0 for k, v in grades.items()):
            raise ValueError("Invalid qrels")
        # Explicit supplied-qrels convention: unlisted documents have relevance 0.
        ranked = parsed["ids"][:args.metric_k] if parsed["valid"] else []
        relevant = {k for k, v in grades.items() if v > 0}
        if relevant:
            dcg = lambda values: sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(values))
            ideal = dcg(sorted(grades.values(), reverse=True)[:args.metric_k])
            ndcg.append(dcg([grades.get(k, 0) for k in ranked]) / ideal)
            recall.append(len(set(ranked) & relevant) / len(relevant))
    result = {"queries": len(rows), "valid_fraction": sum(valid) / len(valid),
              "mean_selected_count_invalid_as_zero": sum(counts) / len(counts),
              "ranking_metrics_available": labels is not None}
    if labels is not None:
        result.update({"metric_k": args.metric_k, "judged_positive_queries": len(ndcg),
                       "ndcg": sum(ndcg) / len(ndcg) if ndcg else None,
                       "candidate_pool_recall": sum(recall) / len(recall) if recall else None})
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
