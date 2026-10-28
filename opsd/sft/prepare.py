"""Losslessly parse official RR RAG inputs and retain original silver outputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from opsd.common import digest, messages, parse_output, read_rows

EXPECTED_SHA256 = "ae75a131d68826ffc36a76de6e29b302ea900c6db281637113568ec3ba5cdd40"
RAG_SYSTEM = "You are a high-quality document set selector for a retrieval-augmented QA system."
HEADER = (r"I will provide you with (\d+) passages, each indicated by a numerical identifier \[\]\. "
          r"Select and rank the passages based on their joint usefulness for the user question: (.*)\.")
TAIL = ("\n\nUser Question: {query}.\nSelect the best subset of the {n} passages above so that, judged by the 5 rubrics, "
        "the selected set best answers the question. Then rank the selected passages from most to least useful. "
        "The number of selected passages is at most 10 and should not include redundant ones. "
        "The output must be only the selected identifiers in ranked order, e.g. [2] [1] [4]. "
        "Only respond with the selection; do not say any other word and do not explain.")


def normalize(query):
    return " ".join(query.lower().split())


def parse_rag(record):
    header, body = record["instruction"].split("\n\n[PASSAGES]\n", 1)
    match = re.fullmatch(HEADER, header, re.DOTALL)
    if not match:
        raise ValueError("unrecognized_header")
    count, query = int(match[1]), match[2]
    footer = TAIL.format(query=query, n=count)
    if not body.endswith(footer):
        raise ValueError("unrecognized_footer")
    body = body[:-len(footer)]
    starts = list(re.finditer(r"(?:\A|\n\n)\[(\d+)\] ", body))
    if [int(m[1]) for m in starts] != list(range(1, count + 1)):
        raise ValueError("ambiguous_candidate_boundaries")
    contents = [body[m.end():starts[i + 1].start() if i + 1 < len(starts) else len(body)]
                for i, m in enumerate(starts)]
    if not contents or any(not s.strip() for s in contents):
        raise ValueError("empty_candidate")
    if "\n\n".join(f"[{i}] {s}" for i, s in enumerate(contents, 1)) != body:
        raise ValueError("candidate_roundtrip_failed")
    return {"query": query, "docs": [{"content": s} for s in contents]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--exclude-val", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source, target = Path(args.source), Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != EXPECTED_SHA256:
        raise ValueError("Official source hash mismatch")
    raw = json.loads(source.read_text(encoding="utf-8"))
    held_out = {normalize(row["query"]) for row in read_rows(args.exclude_val)}
    official, aligned, audit, excluded = [], [], [], []
    for index, record in enumerate(raw):
        if not record["system"].startswith(RAG_SYSTEM):
            continue
        try:
            parsed = parse_rag(record)
        except (ValueError, KeyError) as exc:
            excluded.append({"source_index": index, "reason": str(exc)})
            continue
        labels = parse_output(record["output"], len(parsed["docs"]))
        if not labels["valid"] or len(labels["ids"]) > 10:
            excluded.append({"source_index": index, "query": parsed["query"], "reason": "invalid_silver_ids",
                             "output": record["output"]})
            continue
        if normalize(parsed["query"]) in held_out:
            excluded.append({"source_index": index, "query": parsed["query"], "reason": "rr_rag_val_overlap"})
            continue
        prompt = messages(parsed)
        official.append(record)
        aligned.append({"system": prompt[0]["content"], "instruction": prompt[1]["content"], "output": record["output"]})
        audit.append({"id": f"rr_sft:{index}", "source_index": index, "query": parsed["query"],
                      "candidate_count": len(parsed["docs"]), "candidate_hash": digest(parsed["docs"]),
                      "silver_ids": labels["ids"], "output_unchanged": aligned[-1]["output"] == record["output"]})
    if not aligned:
        raise ValueError("No valid RAG records")
    target.mkdir(parents=True)
    for name, data in (("rr_rag_official_prompt.json", official), ("rr_rag_opsd_prompt.json", aligned),
                       ("records.json", audit), ("excluded.json", excluded)):
        (target / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = {"rr_rag_sft": {"file_name": "rr_rag_opsd_prompt.json",
        "columns": {"system": "system", "prompt": "instruction", "response": "output"}}}
    (target / "dataset_info.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    manifest = {"official_source_sha256": source_sha, "official_total": len(raw),
        "official_rag": sum(r["system"].startswith(RAG_SYSTEM) for r in raw), "prepared_train": len(aligned),
        "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
        "candidate_counts": dict(Counter(r["candidate_count"] for r in audit)),
        "silver_selection_counts": dict(Counter(len(r["silver_ids"]) for r in audit)),
        "unique_queries": len({normalize(r["query"]) for r in audit}),
        "current_opsd_common_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("common.py").read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256((target / "rr_rag_opsd_prompt.json").read_bytes()).hexdigest(),
        "adaptations": ["RAG only", "exclude existing RR RAG val queries", "same Student messages as current OPSD",
                        "original candidates/order and silver output unchanged", "no extra random validation split"]}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
