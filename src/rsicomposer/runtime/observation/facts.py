"""Public, method-declared observations. No event feeds back into an RSI run."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
import re

TOKEN = re.compile(r'[A-Za-z0-9_.-]{1,200}\Z')
METRIC = re.compile(r'[A-Za-z0-9_./-]{1,100}\Z')
REFERENCE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}\Z')
CATEGORIES = ('decision', 'quality', 'execution')
VIEWS = ('timeline', 'lineage', 'table', 'series', 'diff')
PRIVATE = re.compile(r'password|api[_-]?key|secret|authorization|credential|token[_-]?value', re.I)
MAX_RECORD_BYTES = 128 * 1024


def observation_id(*parts):
    """Build a stable opaque identity from an operation, scope or artifact reference."""
    return sha256(json.dumps(parts, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def token(value):
    return isinstance(value, str) and TOKEN.fullmatch(value) is not None


def number(value):
    return (type(value) is int and abs(value) <= 2**53
            or type(value) is float and math.isfinite(value))


def public_links(links):
    if not isinstance(links, dict):
        return {}
    return {name: list(dict.fromkeys(ids)) for name, ids in list(links.items())[:24]
            if token(name) and not PRIVATE.search(name) and isinstance(ids, (tuple, list))
            and len(ids) <= 128 and all(token(item) for item in ids)}


@dataclass(frozen=True)
class ObservationProbe:
    """Subscribe to one named event and declare its public fields and default view.

    Field types are ``id``, ``ref``, ``number``, ``bool`` and ``metrics``. Only a probe
    explicitly declaring ``content=True`` may publish text attached to a version.
    """
    kind: str
    category: str
    fields: tuple = ()
    view: str = 'timeline'
    content: bool = False

    def __post_init__(self):
        fields = tuple(tuple(pair) for pair in self.fields)
        if (not token(self.kind) or self.category not in CATEGORIES or self.view not in VIEWS
                or type(self.content) is not bool or len(fields) > 24
                or any(len(pair) != 2 or not token(pair[0]) or PRIVATE.search(pair[0])
                       or pair[1] not in ('id', 'ref', 'number', 'bool', 'metrics') for pair in fields)
                or len(dict(fields)) != len(fields)):
            raise ValueError('Invalid observation probe declaration')
        object.__setattr__(self, 'fields', fields)

    def project(self, payload):
        fields = payload.get('fields')
        fields = fields if isinstance(fields, dict) else {}
        public = {}
        for name, kind in self.fields:
            value = fields.get(name)
            if (kind == 'number' and number(value) or kind == 'id' and token(value)
                    or kind == 'ref' and isinstance(value, str) and REFERENCE.fullmatch(value)
                    or kind == 'bool' and type(value) is bool):
                public[name] = value
            elif kind == 'metrics' and isinstance(value, dict):
                public[name] = {k: v for k, v in list(value.items())[:100]
                                if isinstance(k, str) and METRIC.fullmatch(k) and not PRIVATE.search(k) and number(v)}
        return public


def emit_observation(kind, event_id, *, scope='run', subject=None, links=None, fields=None, components=None):
    """Offer a declared fact using a stable event ID; available in a run or its workers.

    ``subject`` identifies the candidate/attempt, not necessarily its content.
    Links name a relationship and point to explicit IDs (e.g. ``derived_from``,
    ``evidence``, ``evaluates``, ``selected`` or ``applied``). Undeclared fields
    and malformed events do not reach public storage. The emitter is best effort.
    """
    from . import _operation, current_observations, observations_suppressed
    from .client import INBOX_ENV, OBSERVATION_URL_ENV, worker_observation
    if observations_suppressed():
        return False
    current = current_observations()
    if current is None and not (os.environ.get(INBOX_ENV) or os.environ.get(OBSERVATION_URL_ENV)):
        return False
    try:
        if not all(token(item) for item in (kind, event_id, scope)) or subject is not None and not token(subject):
            return False
        if fields is not None and not isinstance(fields, dict):
            return False
        payload = {'kind': kind, 'event_id': event_id, 'scope': scope, 'subject': subject,
                   'links': public_links(links), 'fields': fields or {}, 'call_id': _operation.get(),
                   'components': components}
        # A size bound also limits the amount of work a fact can put on the queue.
        wire = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
        if len(wire) > MAX_RECORD_BYTES:
            return False
        if current is not None:
            return current.dispatch(current.publisher.fact, json.loads(wire)) is not False
        return worker_observation(wire)
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError, OverflowError):
        return False
