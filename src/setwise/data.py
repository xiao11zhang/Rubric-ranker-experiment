from __future__ import annotations

import hashlib
import json
import math
import copy
from pathlib import Path

SET_DIMS = ("Relevance", "Conciseness", "Consistency")
DOC_DIMS = ("Source Authority", "Timeliness")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def validate_row(row, expected_k=None):
    for key in ("id", "query", "split", "docs", "rubrics"):
        if key not in row:
            raise ValueError(f"Missing {key}")
    if not isinstance(row["query"], str) or not row["query"].strip():
        raise ValueError("Empty query")
    docs = row["docs"]
    if not docs or (expected_k is not None and len(docs) != expected_k):
        raise ValueError(f"{row['id']}: incorrect candidate count")
    if len({d["document_id"] for d in docs}) != len(docs):
        raise ValueError("Duplicate global document ID")
    for position, doc in enumerate(docs, 1):
        if doc["position"] != position or not isinstance(doc["content"], str) or not doc["content"].strip():
            raise ValueError("Candidate order/content invalid")
    active = 0
    for tier, names in (("set_level_rubrics", SET_DIMS), ("doc_level_rubrics", DOC_DIMS)):
        for name, rubrics in row["rubrics"].get(tier, {}).items():
            if name not in names:
                raise ValueError(f"Unexpected RR rubric: {name}")
            for rubric in rubrics:
                weight = rubric["weight"]
                if not math.isfinite(weight) or weight <= 0 or not rubric["description"].strip():
                    raise ValueError("Invalid rubric")
                active += 1
            if tier == "doc_level_rubrics" and rubrics:
                for doc in docs:
                    value = doc.get(name)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Missing/invalid pre-score: {name}")
    if not active:
        raise ValueError("No active rubrics")
    labels = row.get("silver_ids")
    if labels is not None and (not labels or len(labels) != len(set(labels)) or any(type(i) is not int or not 1 <= i <= len(docs) for i in labels)):
        raise ValueError("Invalid silver IDs")
    if {"rubric_set", "rubric_set_reward"} & row.keys():
        raise ValueError("Precomputed ATO proposals are forbidden")


def student_prompt(row):
    # Preserve the frozen official-data prompt, including role/template choices.
    # Do not silently replace missing research prompts with a hand-written one.
    if "prompt" in row:
        messages = row["prompt"]
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in {"system", "user"}
            or not isinstance(m.get("content"), str) or not m["content"].strip()
            for m in messages
        ):
            raise ValueError("Frozen Student prompt must contain nonempty system/user messages")
        return copy.deepcopy(messages)
    if row.get("split") != "debug":
        raise ValueError("Provide the frozen official-data 'prompt'; automatic research prompt synthesis is disabled")
    # This hand-written prompt is restricted to the explicit offline fixture.
    docs = "\n\n".join(f"[{d['position']}] {d['content']}" for d in row["docs"])
    return [{"role": "user", "content": f"Select a complementary, concise document subset that supports answering the query. Output only document IDs, such as [2] [5].\n\n### Query\n{row['query']}\n\n### Documents\n{docs}\n\n### Output"}]


def prepare(source, destination, expected_k=None):
    rows = list(read_jsonl(source))
    seen = set()
    for row in rows:
        validate_row(row, expected_k)
        if row["id"] in seen:
            raise ValueError("Duplicate query ID")
        seen.add(row["id"])
    if not rows:
        raise ValueError("Empty dataset")
    import pyarrow as pa
    import pyarrow.parquet as pq
    converted = []
    for row in rows:
        converted.append({"data_source": "set_selection", "prompt": student_prompt(row),
                          "reward_model": {"style": "rule", "ground_truth": ""},
                          "extra_info": {"query_id": row["id"], "query": row["query"], "scenario": "rag",
                                         "initial_list": row["docs"], "rubrics": row["rubrics"], "split": row["split"]}})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {"source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "parquet_sha256": digest, "query_ids": sorted(seen), "count": len(rows),
                "splits": sorted({r["split"] for r in rows})}
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
