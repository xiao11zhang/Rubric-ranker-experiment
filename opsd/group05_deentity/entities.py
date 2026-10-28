"""Deterministic replacements in description VALUES; no free-form rubric rewriting."""
from copy import deepcopy
import re

from opsd.common import digest

KINDS = {'person': 'PERSON', 'organization': 'ORG', 'location': 'PLACE', 'work': 'WORK'}


def descriptions(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'description' and isinstance(child, str):
                yield path + (key,), child
            else:
                yield from descriptions(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from descriptions(child, path + (index,))


def letter(index):
    result = ''
    while True:
        index, remainder = divmod(index, 26)
        result = chr(65 + remainder) + result
        if not index:
            return result
        index -= 1


def occurrences(text, mention):
    # Do not replace Ann inside Anna, US inside MUSIC, etc.
    pattern = (r'(?<!\w)' if mention[0].isalnum() else '') + re.escape(mention)
    pattern += r'(?!\w)' if mention[-1].isalnum() else ''
    return list(re.finditer(pattern, text))


def transform(rubrics, entities):
    if not isinstance(entities, list):
        raise ValueError('entities must be a list')
    strings = list(descriptions(rubrics))
    if not strings:
        raise ValueError('No description fields found; unsupported rubric schema')
    if any(re.search(r'<(?:PERSON|ORG|PLACE|WORK)_[A-Z]+>', s) for _, s in strings):
        raise ValueError('Source already contains reserved placeholders')
    normalized, counts, seen, scoped = [], {}, {}, {}
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) not in ({'type', 'mentions'}, {'type', 'mentions', 'contexts'}):
            raise ValueError('Each entity requires type/mentions and optional exact contexts')
        kind, mentions = entity['type'], entity['mentions']
        if kind not in KINDS or not isinstance(mentions, list) or not mentions:
            raise ValueError('Unsupported entity type or empty mentions')
        local_seen = set()
        for mention in mentions:
            if not isinstance(mention, str) or not mention.strip() or mention != mention.strip():
                raise ValueError('Invalid entity surface')
            if mention in local_seen or (mention in seen and not (seen[mention] and entity.get('contexts'))):
                raise ValueError('Repeated/ambiguous alias: ' + mention)
            local_seen.add(mention)
            if re.fullmatch(r'[\d\s.,:/+%\-–]+', mention) and kind != 'work':
                raise ValueError('Standalone date/number is not a named entity: ' + mention)
            if re.fullmatch(r'[\d\s.,:/+%\-–]+', mention) and not entity.get('contexts'):
                raise ValueError('Numeric work names require exact occurrence contexts, not global replacement: ' + mention)
            if not any(occurrences(text, mention) for _, text in strings):
                raise ValueError('Mention absent from rubric descriptions: ' + mention)
            # A shared spelling is allowed only if EVERY entity scopes it.
            # The span-overlap check below still rejects ambiguous occurrences.
            seen[mention] = bool(entity.get('contexts'))
        index = counts.get(kind, 0)
        counts[kind] = index + 1
        item = {'type': kind, 'mentions': mentions, 'placeholder': '<' + KINDS[kind] + '_' + letter(index) + '>'}
        if 'contexts' in entity:
            contexts = entity['contexts']
            if not isinstance(contexts, list) or not contexts:
                raise ValueError('contexts must be a nonempty list')
            allowed, used_mentions = set(), set()
            field_text = dict(strings)
            for context in contexts:
                if not isinstance(context, dict) or set(context) != {'path', 'quote'} or not isinstance(context['path'], list):
                    raise ValueError('Each context requires exact description path and quote')
                path, quote = tuple(context['path']), context['quote']
                if path not in field_text or not isinstance(quote, str) or not quote:
                    raise ValueError('Unknown context path or empty quote')
                positions = list(re.finditer(re.escape(quote), field_text[path]))
                if len(positions) != 1:
                    raise ValueError('Context quote must identify exactly one occurrence in its description')
                found = False
                for mention in mentions:
                    hits = occurrences(quote, mention)
                    if len(hits) > 1:
                        raise ValueError('Context repeats an alias; provide a narrower unambiguous quote')
                    for hit in hits:
                        if re.fullmatch(r'[\d\s.,:/+%\-–]+', mention):
                            work_cue = re.search(r'\b(?:novel|book|film|movie|song|album|title|titled|named|work|poem|play|series|artwork|painting)\b|[《》"“”‘’]|小说|电影|歌曲|作品|书名|专辑', quote, re.I)
                            date_prefix = re.search(r'\b(?:year|in|since|before|after|during|until|by|from|between|on)\s+(?:the\s+)?$', quote[:hit.start()], re.I)
                            if not work_cue or date_prefix or quote[hit.end():].startswith('年'):
                                raise ValueError('Numeric alias context is a year/condition or lacks a work-title cue')
                        allowed.add((path, positions[0].start() + hit.start(), positions[0].start() + hit.end(), mention))
                        used_mentions.add(mention)
                        found = True
                if not found:
                    raise ValueError('Context contains no declared entity alias')
            if used_mentions != set(mentions):
                raise ValueError('Each scoped alias needs an exact context')
            item['contexts'] = deepcopy(contexts)
            scoped[item['placeholder']] = allowed
        normalized.append(item)
    output, changes = deepcopy(rubrics), []
    for path, text in strings:
        matches = []
        for entity in normalized:
            for mention in entity['mentions']:
                for match in occurrences(text, mention):
                    allowed = scoped.get(entity['placeholder'])
                    if allowed is not None and (path, match.start(), match.end(), mention) not in allowed:
                        continue
                    matches.append((match.start(), match.end(), mention, entity))
        matches.sort(key=lambda x: (x[0], -x[1]))
        distinct = []
        for match in matches:
            if distinct and distinct[-1][1] > match[0]:
                previous = distinct[-1]
                if match[1] <= previous[1] and match[3]['placeholder'] == previous[3]['placeholder']:
                    continue  # Same entity's short alias inside its full name.
                raise ValueError('Overlapping entities require review: ' + previous[2] + ' / ' + match[2])
            distinct.append(match)
        matches = distinct
        replaced = text
        for start, end, mention, entity in reversed(matches):
            replaced = replaced[:start] + entity['placeholder'] + replaced[end:]
        parent = output
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = replaced
        for start, end, mention, entity in matches:
            changes.append({'path': list(path), 'start': start, 'end': end, 'source': mention,
                            'replacement': entity['placeholder'], 'type': entity['type']})
    # All keys, list order, weights, dates/numbers and wording OUTSIDE the
    # reviewed entity spans remain byte-for-byte unchanged by construction.
    return output, normalized, changes


def candidate(row, entities, fingerprint, raw=''):
    transformed, normalized, changes = transform(row['rubrics'], entities)
    return {'id': row['id'], 'query_hash': digest(row['query']),
            'original_rubric_hash': digest(row['rubrics']), 'rubric_hash': digest(transformed),
            'rubrics': transformed, 'entities': normalized, 'changes': changes,
            'status': 'needs_review' if changes else 'no_entities_detected',
            'pipeline_fingerprint': fingerprint, 'raw_extraction': raw}


def validate_candidate(row, item):
    if (item['id'] != row['id'] or item['query_hash'] != digest(row['query']) or
            item['original_rubric_hash'] != digest(row['rubrics'])):
        raise ValueError('Candidate ID/query/source mismatch')
    if item['status'] not in ('needs_review', 'no_entities_detected'):
        raise ValueError('Failed conversion requires correction: ' + str(row['id']))
    extracted = [{k: e[k] for k in ('type', 'mentions', 'contexts') if k in e} for e in item['entities']]
    expected = candidate(row, extracted, item['pipeline_fingerprint'], item.get('raw_extraction', ''))
    for key in ('rubrics', 'rubric_hash', 'entities', 'changes', 'status'):
        if item[key] != expected[key]:
            raise ValueError('Candidate transformation changed outside declared entity spans: ' + key)


def teacher_row(row, derived):
    item = derived[row['id']]
    return {**row, 'rubrics': item['rubrics'], 'rubric_hash': item['rubric_hash']}
