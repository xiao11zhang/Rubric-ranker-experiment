"""Reuse audited RR function bodies unchanged; inject only prompt/API I/O.

Complete original files and licenses live in _upstream. AST selection avoids
upstream's import-time .env loading and optional SDK dependencies. Each request
gets its own namespace, so parallel rollouts never patch shared globals.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path
import re
import types

ROOT = Path(__file__).parent / "_upstream"
RR_FUNCTIONS = (
    "_parse_selected_ids", "_filter_ids", "_parse_rubric_scores",
    "_flatten_rubrics", "_format_rubrics_block", "_format_doc_set",
    "_build_set_level_prompt", "_score_set_level", "_doc_level_mean", "compute_score",
)
RR_CONSTANTS = {"_BRACKET_INT_RE", "_INT_RE", "SET_LEVEL_RUBRIC_NAMES", "DOC_LEVEL_RUBRIC_NAMES"}


def verify_sources():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        content = (ROOT / item["local_path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != item["git_blob_sha1"]:
            raise ValueError(f"Upstream snapshot changed: {item['local_path']}")
    return manifest


def load_block_templates(path):
    """Read only the audited YAML subset: top-level '|-' literal strings.

    Unsupported YAML fails instead of silently changing prompt text. These
    snapshots use no aliases, tags, folding or scalar interpolation.
    """
    templates = {}
    key, block = None, []

    def finish():
        if key is not None:
            templates[key] = "\n".join(block).rstrip("\n")

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace():
            if line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*): \|-", line)
            if not match:
                raise ValueError(f"Unsupported template YAML: {line}")
            finish()
            key, block = match[1], []
        elif line.startswith("  "):
            if key is None:
                raise ValueError("Unexpected YAML indentation")
            block.append(line[2:])
        elif not line.strip() and key is not None:
            block.append("")
        elif line.strip():
            raise ValueError("Unsupported YAML line")
    finish()
    return templates


def rr_module(call_gpt, max_attempts=4):
    """Instantiate original RR functions with a caller-provided text transport."""
    verify_sources()
    source_path = ROOT / "rr/set_selection.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in RR_FUNCTIONS:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in RR_CONSTANTS for t in node.targets):
            selected.append(node)
    if {n.name for n in selected if isinstance(n, ast.FunctionDef)} != set(RR_FUNCTIONS):
        raise ValueError("Upstream RR function contract changed")
    module = types.ModuleType("audited_rr_request")
    templates = load_block_templates(ROOT / "rr/set_level_rubric_scoring.yaml")
    module.__dict__.update(re=re, logger=logging.getLogger("setwise.upstream.rr"),
        MAX_ATTEMPTS=max_attempts, _load_prompts=lambda: templates, _call_gpt=call_gpt)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), module.__dict__)
    return module
