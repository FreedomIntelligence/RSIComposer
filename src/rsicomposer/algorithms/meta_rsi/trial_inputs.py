"""Path-independent input attestations for matched MetaRSI trials, never raw examples."""
from uuid import UUID

from .aime import sha256_text
from .engine import digest


ROLES = ('train', 'search', 'validation')


def mounted_inputs(names, datasets, protocols):
    if set(names) != set(ROLES):
        raise ValueError('A MetaRSI trial must mount exactly train, search and validation')
    # Hash the interpreted rows too: different field mappings of the very same
    # bytes can change the question, reference answer or ID without a new file hash.
    inputs = {}
    for role in ROLES:
        rows = datasets[names[role]]
        inputs[role] = {'protocol': protocols[role], 'count': len(rows),
                        'rows_sha256': digest([[r.benchmark, r.id, r.prompt, r.answer] for r in rows])}
    return inputs


def validate_inputs(value):
    if (not isinstance(value, dict) or set(value) != set(ROLES)
            or any(not isinstance(entry, dict) or set(entry) != {'protocol', 'count', 'rows_sha256'}
                   or not sha256_text(entry['protocol']) or not sha256_text(entry['rows_sha256'])
                   or type(entry['count']) is not int or entry['count'] < 1 for entry in value.values())):
        raise ValueError('Invalid MetaRSI trial input binding')


def validate_seed_id(value):
    try:
        valid = isinstance(value, str) and UUID(value).hex == value
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Invalid MetaRSI trial admission seed call ID')
