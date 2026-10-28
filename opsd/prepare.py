"""Convert official RR Parquet without generating rubrics or changing candidates."""
import argparse
import hashlib
import json
from pathlib import Path

from .common import check_disjoint, digest


def convert(path, split):
    import pyarrow.parquet as pq
    result = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for offset, row in enumerate(batch.to_pylist()):
            info = row["extra_info"]
            if info["scenario"] != "rag":
                continue
            if info["split"] != split:
                raise ValueError(f"Unexpected source split: {info['split']}")
            docs = info["initial_list"]
            if not docs or any(not d["content"].strip() for d in docs):
                raise ValueError("Empty candidate")
            rubrics = info["rubrics"]
            if set(rubrics) != {"set_level_rubrics", "doc_level_rubrics"}:
                raise ValueError("Expected RR five-dimension rubric schema")
            result.append({"id": f"rr:{split}:{info['index']}", "split": split,
                           "source_index": info["index"], "query": info["query"],
                           "docs": docs, "rubrics": rubrics, "candidate_hash": digest(docs),
                           "rubric_hash": digest(rubrics)})
    if not result or len({r["id"] for r in result}) != len(result):
        raise ValueError("Empty RAG split or duplicate source index")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("Use a new output directory")
    train, val = convert(args.train, "train"), convert(args.val, "val")
    check_disjoint(train, val)
    root.mkdir(parents=True)
    manifest = {}
    for name, rows, source in (("train", train, args.train), ("val", val, args.val)):
        target = root / f"{name}.jsonl"
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        manifest[name] = {"count": len(rows), "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                          "jsonl_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
