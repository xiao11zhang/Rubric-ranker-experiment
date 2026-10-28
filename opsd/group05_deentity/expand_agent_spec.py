"""Expand compact, agent-authored notation; performs no entity identification or review."""
import argparse
import json
from pathlib import Path

from .agent_annotations import fresh_json

KINDS = {'p': 'person', 'o': 'organization', 'l': 'location', 'w': 'work'}


def expand(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    records = []
    for row in value['records']:
        entities = []
        for cell in row[1:]:
            if isinstance(cell, dict):
                entities.append(cell)
            elif isinstance(cell[1], list):
                sections = {'R': 'Relevance', 'C': 'Conciseness', 'S': 'Consistency'}
                entities.append({'type': KINDS[cell[0]], 'mentions': cell[1],
                    'contexts': [{'path': ['set_level_rubrics', sections[section], index, 'description'],
                                  'quote': quote} for section, index, quote in cell[2]]})
            else:
                entities.append({'type': KINDS[cell[0]], 'mentions': cell[1:]})
        records.append({'id': row[0], 'entities': entities})
    output = path.with_name(value['batch_id'] + '.annotations.json')
    fresh_json(output, {'batch_id': value['batch_id'], 'records': records})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    print(expand(parser.parse_args().spec))


if __name__ == '__main__':
    main()
