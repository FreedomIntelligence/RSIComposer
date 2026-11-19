"""Stable references for Lab-owned benchmark and training data.

The framework standardizes identity, provenance and access role. A Lab still
interprets the records or task files behind a binding according to its domain.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
import re

from .profiles import SPLIT_ROLES


_SHA256 = re.compile(r'^[a-f0-9]{64}$')


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be nonempty text')


@dataclass(frozen=True)
class DatasetBinding:
    """A declaration for one logical dataset split or benchmark cohort."""

    name: str
    split: str
    role: str
    source: str
    schema_version: str
    content_digest: str
    row_ids: tuple[str, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for value, label in ((self.name, 'Dataset name'), (self.split, 'Dataset split'),
                             (self.source, 'Dataset source'),
                             (self.schema_version, 'Dataset schema version')):
            _text(value, label)
        if not isinstance(self.role, str) or self.role not in SPLIT_ROLES:
            raise ValueError('Dataset role must name a standard split role')
        if not isinstance(self.content_digest, str) or not _SHA256.fullmatch(self.content_digest):
            raise ValueError('Dataset content_digest must be a lowercase SHA-256 digest')
        if not isinstance(self.row_ids, tuple) or any(
                not isinstance(row_id, str) or not row_id.strip() for row_id in self.row_ids):
            raise TypeError('Dataset row_ids must be a tuple of nonempty strings')
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError('Dataset row_ids must be unique')
        if not isinstance(self.provenance, Mapping):
            raise TypeError('Dataset provenance must be a mapping')
        provenance = deepcopy(dict(self.provenance))
        try:
            provenance = json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError('Dataset provenance must be JSON-compatible') from error
        object.__setattr__(self, 'provenance', provenance)

    @property
    def identity(self):
        return {
            'name': self.name,
            'split': self.split,
            'role': self.role,
            'source': self.source,
            'schema_version': self.schema_version,
            'content_digest': self.content_digest,
            'row_ids': list(self.row_ids),
            'provenance': deepcopy(dict(self.provenance)),
        }

    def to_json(self):
        return self.identity

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError('Dataset binding must be an object')
        required = {'name', 'split', 'role', 'source', 'schema_version', 'content_digest'}
        missing = required - set(value)
        extra = set(value) - required - {'row_ids', 'provenance'}
        if missing or extra:
            raise ValueError(f'Invalid dataset binding fields, missing={sorted(missing)}, extra={sorted(extra)}')
        rows = value.get('row_ids', [])
        if not isinstance(rows, list):
            raise TypeError('Dataset binding row_ids must be an array')
        return cls(**{key: value[key] for key in required},
                   row_ids=tuple(rows),
                   provenance=value.get('provenance', {}))
