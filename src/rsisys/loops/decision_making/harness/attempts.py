"""Explicit attempt membership, derived from the decision ledger only."""


def validate_attempt_fields(ident, title):
    if ident is not None and (not isinstance(ident, str) or not ident.strip() or len(ident) > 200):
        raise ValueError('research_attempt_id must be a nonempty string of at most 200 characters or None')
    if not isinstance(title, str) or (title and not title.strip()) or len(title) > 500:
        raise ValueError('research_attempt_title must be a string of at most 500 characters, nonblank when supplied')


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
