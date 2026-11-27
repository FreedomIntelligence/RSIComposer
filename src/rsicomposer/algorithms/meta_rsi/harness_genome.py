"""Portable, declarative Harness-RSI genome and typed patch (§4.2 / Appendix C)."""
from copy import deepcopy

from .engine import digest

SLOTS = ('prompt', 'tools', 'skills', 'mcp', 'policies')
RESOURCE_SLOTS = ('tools', 'skills', 'mcp')


def _slot(slot, value):
    if slot == 'prompt':
        if type(value) is not str:
            raise ValueError('Harness prompt must be text')
        return value
    if slot in RESOURCE_SLOTS:
        if type(value) is not list or any(type(key) is not str or not key for key in value):
            raise ValueError('Harness resources must name existing domain bindings')
        if len(set(value)) != len(value):
            raise ValueError('Harness has duplicate resource bindings')
        return list(value)
    if slot == 'policies':
        if type(value) is not dict or any(type(key) is not str or not key for key in value):
            raise ValueError('Harness policies must be a named document')
        digest(value)  # Domain owns the semantics; enforce a portable JSON payload.
        return deepcopy(value)
    raise ValueError('Unknown Harness slot')


def genome(*, prompt='', tools=(), skills=(), mcp=(), policies=None):
    values = dict(prompt=prompt, tools=list(tools), skills=list(skills), mcp=list(mcp),
                  policies={} if policies is None else policies)
    return check_genome(values)


def check_genome(value):
    if type(value) is not dict or set(value) != set(SLOTS):
        raise ValueError('Invalid five-slot Harness genome')
    return {slot: _slot(slot, value[slot]) for slot in SLOTS}


def apply_harness_patch(current, patch, *, writable_slots=SLOTS, resources=None):
    """Apply exactly the requested slot replacements. No discovery or execution of resources."""
    current = check_genome(current)
    slots = set(writable_slots)
    if slots - set(SLOTS) or type(patch) is not dict or not patch or patch.keys() - slots:
        raise ValueError('Harness patch targets a non-writable slot')
    resources = {} if resources is None else resources
    values = {slot: _slot(slot, value) for slot, value in patch.items()}
    for slot in values.keys() & set(RESOURCE_SLOTS):
        if set(values[slot]) - set(resources.get(slot, ())):
            raise ValueError('Harness patch selects an unknown domain resource')
    return {**current, **values}


def experience_patch(current, requested, *, writable_slots=SLOTS, resources=None):
    """Compile a domain-authored extraction to slot changes, deduplicated against the genome."""
    before = check_genome(current)
    after = apply_harness_patch(before, requested, writable_slots=writable_slots, resources=resources)
    return {slot: after[slot] for slot in SLOTS if after[slot] != before[slot]}
