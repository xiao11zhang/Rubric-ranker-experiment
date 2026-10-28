"""Inspect official RR SFT records without transforming or training them."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("path")
args = p.parse_args()
path = Path(args.path)
rows = json.loads(path.read_text(encoding="utf-8"))
systems = Counter(row.get("system", "") for row in rows)
print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "key_sets": {str(k): v for k, v in Counter(tuple(sorted(r)) for r in rows).items()},
                  "system_templates": len(systems)}, ensure_ascii=False, indent=2))
for index, (system, count) in enumerate(systems.items()):
    example = next(r for r in rows if r.get("system", "") == system)
    print(json.dumps({"system_template": index, "count": count, "system": system,
                      "instruction_head": example["instruction"][:1400], "output": example["output"]},
                     ensure_ascii=False, indent=2))
