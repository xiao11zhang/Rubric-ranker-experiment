"""Bounded local extraction reasoning + grammar-constrained JSON; never used in OPSD."""
from opsd.common import digest

SCHEMA = {'type': 'object', 'properties': {'entities': {'type': 'array', 'items': {
    'type': 'object', 'properties': {
        'type': {'type': 'string', 'enum': ['person', 'organization', 'location', 'work']},
        'mentions': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1},
        'contexts': {'type': 'array', 'minItems': 1, 'items': {'type': 'object', 'properties': {
            'path': {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}},
            'quote': {'type': 'string'}}, 'required': ['path', 'quote'], 'additionalProperties': False}}},
    'required': ['type', 'mentions'], 'additionalProperties': False}}},
    'required': ['entities'], 'additionalProperties': False}


def prepare_decoder(tokenizer):
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    return build_token_enforcer_tokenizer_data(tokenizer)


def generate(model, tokenizer, tokens, device, decoder, *, reasoning_budget, max_json_tokens):
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    common = {'do_sample': False, 'temperature': None, 'top_p': None, 'top_k': None,
              'pad_token_id': tokenizer.eos_token_id, 'eos_token_id': tokenizer.eos_token_id}
    reasoning_ids, transcript, closed = [], '', True
    if reasoning_budget:
        inputs = torch.tensor([tokens], device=device)
        with torch.inference_mode():
            proposed = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                       max_new_tokens=reasoning_budget, **common)
        reasoning_ids = proposed[0, len(tokens):].tolist()
        proposed_text = tokenizer.decode(reasoning_ids, skip_special_tokens=True)
        closed = '</think>' in proposed_text
        thought = proposed_text.split('</think>', 1)[0].removeprefix('<think>').strip()
        transcript = '<think>\n' + thought + '\n</think>\n\n'
    # Closing a bounded reasoning block is explicit, recorded inference control.
    # It does not repair any JSON or rewrite the source rubric.
    final_prefix = tokens + tokenizer.encode(transcript, add_special_tokens=False)
    if len(final_prefix) + max_json_tokens > model.config.max_position_embeddings:
        raise ValueError('Controlled extraction exceeds model context')
    final_input = torch.tensor([final_prefix], device=device)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(decoder, JsonSchemaParser(SCHEMA))
    with torch.inference_mode():
        result = model.generate(input_ids=final_input, attention_mask=torch.ones_like(final_input),
            max_new_tokens=max_json_tokens, prefix_allowed_tokens_fn=prefix_fn, **common)
    final_ids = result[0, len(final_prefix):].tolist()
    raw = transcript + tokenizer.decode(final_ids, skip_special_tokens=True)
    return raw, final_ids, {'json_constrained': True, 'schema_hash': digest(SCHEMA),
        'reasoning_budget_tokens': reasoning_budget, 'reasoning_generated_tokens': len(reasoning_ids),
        'reasoning_closed_by_model': closed, 'reasoning_closure_forced': not closed,
        'final_json_tokens': len(final_ids), 'max_json_tokens': max_json_tokens}
