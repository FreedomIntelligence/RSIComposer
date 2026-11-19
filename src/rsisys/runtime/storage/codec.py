"""Explicit JSON codec for registered state types. Never loads Python objects by name.

This module knows no concrete state classes: the composition root registers
them (see rsisys.snapshot.codec), so runtime code can encode and decode
checkpoints without importing harnesses.
"""
from dataclasses import fields, is_dataclass
from enum import Enum

TYPES: dict[str, type] = {}
ENUMS: dict[str, type] = {}
def register_types(*classes):
    """Declare dataclasses that checkpoints may contain; every field must be present."""
    for cls in classes:
        if not is_dataclass(cls) or TYPES.get(cls.__name__, cls) is not cls:
            raise TypeError(f'State types must be uniquely named dataclasses: {cls.__name__}')
        TYPES[cls.__name__] = cls


def register_enums(*classes):
    for cls in classes:
        if not issubclass(cls, Enum) or ENUMS.get(cls.__name__, cls) is not cls:
            raise TypeError(f'State enums must be uniquely named Enum classes: {cls.__name__}')
        ENUMS[cls.__name__] = cls


def encode(value):
    if isinstance(value, Enum):
        if type(value).__name__ not in ENUMS:
            raise TypeError('Unsupported state enum')
        return {'$enum': type(value).__name__, 'value': value.value}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if is_dataclass(value) and type(value).__name__ in TYPES and TYPES[type(value).__name__] is type(value):
        return {'$type': type(value).__name__, 'fields': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {'$map': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, set, frozenset, list)):
        tag = ('$tuple' if isinstance(value, tuple) else '$frozenset' if isinstance(value, frozenset)
               else '$set' if isinstance(value, set) else '$list')
        return {tag: [encode(x) for x in value]}
    raise TypeError(f'Unsupported checkpoint state type: {type(value).__name__}')


def decode(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if not isinstance(value, dict):
        raise ValueError('Invalid checkpoint value')
    if set(value) == {'$enum', 'value'} and value['$enum'] in ENUMS:
        return ENUMS[value['$enum']](value['value'])
    if set(value) == {'$type', 'fields'} and value['$type'] in TYPES:
        cls = TYPES[value['$type']]
        expected = {f.name for f in fields(cls)}
        if not isinstance(value['fields'], dict) or set(value['fields']) != expected:
            raise ValueError('Checkpoint fields do not match the current state schema')
        return cls(**{k: decode(v) for k, v in value['fields'].items()})
    if set(value) == {'$map'}:
        return {decode(k): decode(v) for k, v in value['$map']}
    for tag, factory in (('$tuple', tuple), ('$set', set), ('$frozenset', frozenset), ('$list', list)):
        if set(value) == {tag}:
            return factory(decode(v) for v in value[tag])
    raise ValueError('Unsupported checkpoint type or encoding')
