"""Explicit attempt membership, derived from the decision ledger only."""
from dataclasses import dataclass
from rsisys.loops.contracts.decision import validate_attempt_fields


def attempt_details(row):
    # Older ledger/history records stay ungrouped and retain their replay shape.
    return {key: row[key] for key in ('research_attempt_id', 'research_attempt_title') if key in row}


def known_attempts(records):
    attempts = {}
    for row in records:
        if not row.get('validation_error') and row.get('research_attempt_id') is not None:
            ident, title = row['research_attempt_id'], row.get('research_attempt_title', '')
            validate_attempt_fields(ident, title)
            if not title or (ident in attempts and attempts[ident] != title):
                raise ValueError('Research attempt must retain its original title')
            attempts[ident] = title
    return attempts


def requested_attempt(ident, title, records):
    """Validate a request without assigning a system identity or rewriting it."""
    validate_attempt_fields(ident, title)
    if ident is None:
        return title
    known = known_attempts(records)
    if ident not in known:
        raise ValueError(f'Unknown research_attempt_id: {ident}')
    if title and title != known[ident]:
        raise ValueError('Cannot rename an existing research attempt')
    return known[ident]


@dataclass(frozen=True)
class AttemptContract:
    """Host-owned admission rule; never infer model-owned research grouping."""

    required: bool = False
    operations: tuple[str, ...] = ()

    def __post_init__(self):
        required, operations = self.required, self.operations
        if type(required) is not bool:
            raise TypeError('Attempt requirement must be boolean')
        if (not isinstance(operations, tuple) or any(not isinstance(name, str) or not name.strip()
                for name in operations) or len(set(operations)) != len(operations)):
            raise ValueError('Attempt operations must be distinct nonempty names')

    def applies(self, kind, operation=None):
        return self.required and (kind == 'explore' or (kind == 'invoke' and operation in self.operations))

    def validate(self, kind, operation, ident, title):
        if self.applies(kind, operation) and ident is None and not title:
            raise ValueError('Research action requires an Attempt: provide research_attempt_title for a new '
                             'research idea or research_attempt_id to continue a known attempt')
