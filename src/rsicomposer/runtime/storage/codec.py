"""Explicit JSON codec for registered state types. Never loads Python objects by name.

The bundled schema is selected by the codec, not by the caller's import order.
Plugins may explicitly register additional dataclasses and enums.
"""
from collections import ChainMap
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import cache

from .files import json_bytes

_extra_types: dict[str, type] = {}
_extra_enums: dict[str, type] = {}


@cache
def _catalogs():
    # Only the trusted bundled schema is imported, never a name from a checkpoint.
    from rsicomposer.snapshot.codec import STATE_ENUMS, STATE_TYPES
    return (ChainMap(_extra_types, {cls.__name__: cls for cls in STATE_TYPES}),
            ChainMap(_extra_enums, {cls.__name__: cls for cls in STATE_ENUMS}))


def register_types(*classes):
    """Declare dataclasses that checkpoints may contain; every field must be present."""
    types, _ = _catalogs()
    for cls in classes:
        if not is_dataclass(cls) or types.get(cls.__name__, cls) is not cls:
            raise TypeError(f'State types must be uniquely named dataclasses: {cls.__name__}')
        _extra_types[cls.__name__] = cls


def register_enums(*classes):
    _, enums = _catalogs()
    for cls in classes:
        if not issubclass(cls, Enum) or enums.get(cls.__name__, cls) is not cls:
            raise TypeError(f'State enums must be uniquely named Enum classes: {cls.__name__}')
        _extra_enums[cls.__name__] = cls


def encode(value, *, sort_maps=False):
    if isinstance(value, Enum):
        if _catalogs()[1].get(type(value).__name__) is not type(value):
            raise TypeError('Unsupported state enum')
        return {'$enum': type(value).__name__, 'value': value.value}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if is_dataclass(value) and _catalogs()[0].get(type(value).__name__) is type(value):
        return {'$type': type(value).__name__, 'fields': {
            f.name: encode(getattr(value, f.name), sort_maps=sort_maps) for f in fields(value)}}
    if isinstance(value, dict):
        pairs = [[encode(k, sort_maps=sort_maps), encode(v, sort_maps=sort_maps)] for k, v in value.items()]
        # An adapter's native state can depend on insertion order. Only a
        # metadata comparison explicitly requests order-insensitive mappings.
        if sort_maps:
            pairs.sort(key=lambda pair: json_bytes(pair[0]))
        return {'$map': pairs}
    if isinstance(value, (tuple, set, frozenset, list)):
        tag = ('$tuple' if isinstance(value, tuple) else '$frozenset' if isinstance(value, frozenset)
               else '$set' if isinstance(value, set) else '$list')
        items = [encode(x, sort_maps=sort_maps) for x in value]
        if tag in ('$set', '$frozenset'):
            items.sort(key=json_bytes)  # Unordered, including across processes/hash seeds.
        return {tag: items}
    raise TypeError(f'Unsupported checkpoint state type: {type(value).__name__}')


def same_metadata(left, right):
    """Typed codec identity for configuration or provenance, not ordered live state."""
    return json_bytes(encode(left, sort_maps=True)) == json_bytes(encode(right, sort_maps=True))


def decode(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if not isinstance(value, dict):
        raise ValueError('Invalid checkpoint value')
    if set(value) == {'$enum', 'value'} and (cls := _catalogs()[1].get(value['$enum'])) is not None:
        return cls(value['value'])
    if set(value) == {'$type', 'fields'} and (cls := _catalogs()[0].get(value['$type'])) is not None:
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
