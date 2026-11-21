"""Canonical evidence encoding and unambiguous object parsing."""
from hashlib import sha256
import json


class EvidencePathError(ValueError):
    """A caller-selected JSON pointer does not address recorded evidence."""


def evidence_value(value, pointer):
    """Resolve a JSON pointer without leaking key/index errors as system faults."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        raise EvidencePathError('Evidence path must be a JSON pointer')
    for part in pointer.split('/')[1:] if pointer else ():
        escaped = part.replace('~1', '').replace('~0', '')
        if '~' in escaped:
            raise EvidencePathError('Evidence path has an invalid JSON pointer escape')
        key = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif (isinstance(value, (list, tuple)) and key.isascii() and key.isdecimal()
              and (key == '0' or not key.startswith('0')) and int(key) < len(value)):
            value = value[int(key)]
        else:
            raise EvidencePathError('Evidence path does not identify a recorded value; read the root to inspect its fields')
    return value


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def result_id(value):
    return sha256(encoded(value).encode('utf-8')).hexdigest()


def load_object(text):
    """Reject ambiguous JSON before either decision parser sees an object."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key {key!r}; keys must be unique in every object')
            result[key] = value
        return result

    payload = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError('Decision must be a JSON object')
    return payload
