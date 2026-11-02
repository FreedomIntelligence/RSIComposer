"""Canonical evidence encoding and unambiguous object parsing."""
from hashlib import sha256
import json


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
