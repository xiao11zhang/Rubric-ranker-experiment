from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty data or duplicate IDs")
    for row in rows:
        if not row["query"].strip() or not row["docs"] or not row["rubrics"]:
            raise ValueError("Missing query, candidates or rubrics")
        if row["candidate_hash"] != digest(row["docs"]):
            raise ValueError("Candidate hash mismatch")
        if row["rubric_hash"] != digest(row["rubrics"]):
            raise ValueError("Rubric hash mismatch")
    return rows


def messages(row, *, teacher=False, mode="correct", top_k=0):
    if top_k < 0 or top_k > len(row["docs"]):
        raise ValueError("Invalid top_k")
    instruction = (f"Select exactly {top_k} passages" if top_k else "Select a subset of passages")
    system = ("You select evidence for retrieval-augmented question answering. " + instruction
              + " and rank the selected passages from most to least useful for answering the question. "
              "Output only distinct passage identifiers in square brackets separated by spaces, "
              "for example [2] [5]. Do not output explanations.")
    content = "Question:\n" + row["query"] + "\n\nPassages:\n" + "\n\n".join(
        f"[{i}] {doc['content']}" for i, doc in enumerate(row["docs"], 1))
    if teacher and mode == "correct":
        content += "\n\nAdditional evaluation criteria:\n" + json.dumps(row["rubrics"], ensure_ascii=False, sort_keys=True)
    elif mode not in {"correct", "none"}:
        raise ValueError("Only correct/none PI are implemented in the initial experiment")
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def parse_output(text, n_docs, top_k=0):
    ids = [int(s) for s in re.findall(r"\[(\d+)\]", text)]
    valid = bool(ids) and bool(re.fullmatch(r"\s*(?:\[\d+\]\s*)+", text))
    valid = valid and len(ids) == len(set(ids)) and all(1 <= i <= n_docs for i in ids)
    valid = valid and (not top_k or len(ids) == top_k)
    return {"ids": ids, "valid": bool(valid)}


def encode_prompt(tokenizer, row, *, teacher=False, mode="correct", top_k=0):
    return tokenizer.apply_chat_template(messages(row, teacher=teacher, mode=mode, top_k=top_k),
                                         tokenize=True, add_generation_prompt=True, enable_thinking=False)


def check_disjoint(train, val):
    normalize = lambda q: " ".join(q.lower().split())
    if {normalize(r["query"]) for r in train} & {normalize(r["query"]) for r in val}:
        raise ValueError("Train/validation query overlap")
