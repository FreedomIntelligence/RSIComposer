"""Receipt predicates for task acceptance; never judge free-form scientific claims."""
import json
import re
from dataclasses import asdict

from rsicomposer.loops.contracts.decision import OperationArgumentError
from rsicomposer.runtime.storage.json_tools import result_id

CHECKS = {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {
    'type': 'object', 'properties': {
        'id': {'type': 'string', 'minLength': 1, 'maxLength': 128},
        'operation': {'type': 'string', 'minLength': 1, 'maxLength': 128},
        'result_path': {'type': 'string', 'minLength': 1, 'maxLength': 512},
        'expected': {'type': ['string', 'number', 'boolean']}},
    'required': ['id', 'operation', 'result_path', 'expected'], 'additionalProperties': False}}
EVIDENCE = {'type': 'object', 'maxProperties': 4,
            'additionalProperties': {'type': 'string', 'minLength': 1, 'maxLength': 512}}
RESOLUTIONS = ['resolved', 'abandoned', 'inconclusive', 'interrupted']


def validate_checks(checks):
    if not isinstance(checks, list) or not 1 <= len(checks) <= 4:
        raise OperationArgumentError('Supply 1..4 acceptance_checks')
    ids = set()
    for check in checks:
        if (not isinstance(check, dict) or set(check) != {'id', 'operation', 'result_path', 'expected'}
                or any(not isinstance(check[k], str) or not check[k].strip()
                       or len(check[k]) > limit for k, limit in
                       [('id', 128), ('operation', 128), ('result_path', 512)])
                or not check['result_path'].startswith('/')
                or re.search(r'~(?![01])', check['result_path'])
                or type(check['expected']) not in (str, int, float, bool)
                or len(str(check['expected'])) > 1000 or check['id'] in ids):
            raise OperationArgumentError('Acceptance checks require unique ids, operation, JSON result_path and scalar expected')
        # Reject non-finite values as well as otherwise non-JSON arguments.
        try:
            json.dumps(check['expected'], allow_nan=False)
        except ValueError as exc:
            raise OperationArgumentError('Expected must be finite JSON') from exc
        ids.add(check['id'])


def accept(task, evidence, native):
    checks = task.get('acceptance_checks', [])
    if not checks:
        raise OperationArgumentError('Resolved tasks require explicit acceptance_checks; update the task first. '
                                     'Use inconclusive/abandoned/interrupted when acceptance is incomplete.')
    if set(evidence) != {c['id'] for c in checks}:
        raise OperationArgumentError('Supply acceptance_evidence for every acceptance check')
    receipts = {}
    for check in checks:
        row = native.get(evidence[check['id']])
        result = row.details.get('result', {}) if row is not None else {}
        data = result.get('data') or {}
        if (row is None or row.kind != 'operation' or row.simulated or result.get('simulated')
                or row.details.get('operation', {}).get('name') != check['operation']
                or result.get('status') != 'completed'
                or result.get('error') or not isinstance(data, dict)
                or data.get('state', 'succeeded') != 'succeeded'):
            raise OperationArgumentError('Acceptance needs a successful native receipt, not a submission or failed job')
        value = result
        try:
            for part in check['result_path'][1:].split('/'):
                key = part.replace('~1', '/').replace('~0', '~')
                if isinstance(value, list):
                    if re.fullmatch('0|[1-9][0-9]*', key) is None:
                        raise ValueError('Invalid array index')
                    value = value[int(key)]
                else:
                    value = value[key]
        except (KeyError, IndexError, TypeError, ValueError):
            raise OperationArgumentError('Acceptance result_path is absent in the cited receipt') from None
        # bool is a subclass of int, but true is not evidence of n=1.
        if type(value) is not type(check['expected']) or value != check['expected']:
            raise OperationArgumentError('Acceptance predicate failed: ' + check['id'])
        receipts[check['id']] = {'record_id': row.record_id, 'sha256': result_id(asdict(row))}
    return receipts
