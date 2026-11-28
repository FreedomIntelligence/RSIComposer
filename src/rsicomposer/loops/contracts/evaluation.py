"""Identity of a declared evaluation, shared by Host delivery and run-level goals."""
from dataclasses import replace
from hashlib import sha256
import json

from .lifecycle import ExecutionStatus

_INPUT_KEYS = ('split', 'task_ids', 'indices', 'dataset')
_CANDIDATE_PREFIX = 'candidate-json-sha256:'


def candidate_id(arguments, fields):
    """Identify an admitted target from its declared ordered argument paths only.

    Ordinary single string IDs stay opaque. Other JSON values and tuples are
    content addressed; a string resembling a generated ID cannot impersonate it.
    Names may differ across operations: position, not spelling, identifies parts.
    """
    if not fields:
        return None
    values = []
    for path in fields:
        value = arguments
        for key in path.split('.'):
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        values.append(value)
    if len(values) == 1 and isinstance(values[0], str) and not values[0].startswith(_CANDIDATE_PREFIX):
        return values[0]
    try:
        encoded = json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if json.loads(encoded) != values:  # Reject non-JSON keys/types rather than silently coercing them.
            return None
    except (TypeError, ValueError, RecursionError):
        return None
    return _CANDIDATE_PREFIX + sha256(encoded.encode('ascii')).hexdigest()


def check_candidate(call, result, fields):
    """Reject a successful receipt contradicting the installed candidate declaration."""
    expected = candidate_id(call.arguments, fields)
    if expected is None:
        raise ValueError(f'{call.name} requires the declared candidate identity fields: {fields}')
    if 'candidate_id' in result.data and result.data['candidate_id'] != expected:
        raise ValueError(f'{call.name} receipt candidate_id differs from its declared candidate identity')
    return expected


def canonical_refs(refs):
    """Explicit, ordered cohort references. No coercion or silent deduplication."""
    if (not isinstance(refs, (list, tuple)) or not refs
            or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
            or len(set(refs)) != len(refs)):
        return None
    return tuple(refs)


def _selector_refs(key, value):
    if value is None and key in ('task_ids', 'indices'):
        return ()  # The bundled Labs use null for the entire requested split.
    values = value if isinstance(value, (list, tuple)) else (value,)
    if key == 'indices':
        if any(type(n) is not int or n < 0 for n in values):
            return None
        values = tuple(str(n) for n in values)
    values = canonical_refs(values)
    return None if values is None else tuple(f'{key}:{item}' for item in values)


def input_refs(selectors):
    """Only known selectors can establish missing refs; input_refs on a request are a claim."""
    refs = []
    for key in _INPUT_KEYS:
        if key in selectors:
            part = _selector_refs(key, selectors[key])
            if part is None:
                return None
            refs.extend(part)
    return tuple(refs) or None


def _selectors(arguments, data):
    return {key: arguments.get(key, data.get(key)) for key in _INPUT_KEYS
            if key in arguments or key in data}


def matches_cohort(arguments, data, refs):
    """Compare native selectors when knowable; Labs may use their own canonical cohort IDs."""
    if 'input_refs' in arguments and canonical_refs(arguments['input_refs']) != refs:
        return False
    for key in _INPUT_KEYS:
        if key in arguments and key in data:
            requested, reported = _selector_refs(key, arguments[key]), _selector_refs(key, data[key])
            if requested != reported or (requested is None and arguments[key] != data[key]):
                return False
    expected = {}
    for key, value in _selectors(arguments, data).items():
        part = _selector_refs(key, value)
        if part is None and 'input_refs' not in data:
            return False  # Only the Lab can identify a cohort with a non-native selector.
        if part != ():
            expected[key] = part
    namespaces = {ref.partition(':')[0] if ':' in ref else None for ref in refs}
    for key in expected.keys() & namespaces:
        if tuple(ref for ref in refs if ref.startswith(key + ':')) != expected[key]:
            return False
    # Purely native references may not omit a selector. An opaque provider
    # reference (e.g. a dataset digest for a split) can identify a cohort itself.
    return not namespaces < expected.keys()


def tag_evaluation_identity(call, result, scorer, fields):
    """Fill missing facts on a Host-delivered declared evaluation, not in the Lab journal."""
    if result.status != ExecutionStatus.COMPLETED:
        return result
    args, data = call.arguments, result.data
    try:
        candidate = check_candidate(call, result, fields) if fields else None
    except ValueError as error:
        return replace(result, status=ExecutionStatus.FAILED, error=str(error), adopted_candidate_ids=())
    derived = {'candidate_id': candidate}
    if 'quality_metrics' in data:
        derived['protocol_ref'] = scorer
        if 'input_refs' not in data:
            refs = input_refs(_selectors(args, data))
            if refs is not None and matches_cohort(args, data, refs):
                derived['input_refs'] = refs
    additions = {key: value for key, value in derived.items()
                 if value and key not in data}
    if not additions:
        return result
    return replace(result, data={**data, **additions})
