"""Format agent-authored repair specs; no NER, model calls or review decisions.

An agent explicitly names entities, aliases and containing names to exclude.
Only exact span/context construction is automated. Original entities.py validates
the expanded result; this helper never writes API records or approves reviews.
"""
import argparse
import json
import re
from pathlib import Path

from . import agent_annotations as agent
from .entities import descriptions, occurrences, candidate

KINDS = {'p': 'person', 'o': 'organization', 'l': 'location', 'w': 'work'}


def scoped(entity, rubrics, excluded):
    fields = list(descriptions(rubrics))
    contexts = []
    for path, text in fields:
        blocked = [(m.start(), m.end()) for name in excluded for m in occurrences(text, name)]
        for alias in entity['mentions']:
            for hit in occurrences(text, alias):
                if any(start <= hit.start() and hit.end() <= end for start, end in blocked):
                    continue
                for width in range(len(text) + 1):
                    start, end = max(0, hit.start()-width), min(len(text), hit.end()+width)
                    quote = text[start:end]
                    if len(list(re.finditer(re.escape(quote), text))) != 1:
                        continue
                    if any(len(occurrences(quote, name)) > 1 for name in entity['mentions']):
                        continue
                    value = {'path': list(path), 'quote': quote}
                    if value not in contexts:
                        contexts.append(value)
                    break
                else:
                    raise ValueError('Agent must supply manual exact contexts for ' + alias)
    return {**entity, 'contexts': contexts}


def expand(spec_path):
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    _, _, selected, _, _ = agent.sources(Path('opsd/group05_deentity/config.api.single.toml'))
    by_id = {row['id']: row for row in selected}
    result = []
    for row in spec['records']:
        entities = []
        for cell in row['entities']:
            if isinstance(cell, dict):
                entities.append(cell)
                continue
            aliases = cell[1] if isinstance(cell[1], list) else [cell[1]]
            entity = {'type': KINDS[cell[0]], 'mentions': aliases}
            if len(cell) > 2:
                entity = scoped(entity, by_id[row['id']]['rubrics'], cell[2])
            entities.append(entity)
        candidate(by_id[row['id']], entities, 'local-precheck-only')
        result.append({'id': row['id'], 'entities': entities, 'notes': row['notes']})
    target = spec_path.with_suffix('.expanded.json')
    agent.fresh_json(target, result)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    print(expand(parser.parse_args().spec))
