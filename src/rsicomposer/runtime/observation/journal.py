"""Versioned, append-only public facts independent of the dashboard window."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

from .facts import MAX_RECORD_BYTES, ObservationProbe, PRIVATE, public_links, token
from rsicomposer.runtime.storage.files import atomic_write, json_bytes

SCHEMA = 1
JOURNAL = 'monitor-events.jsonl'
CORE_PROBES = (
    ObservationProbe('host.evaluated', 'quality', (('benchmark', 'id'), ('split', 'id'),
        ('model', 'id'), ('scorer', 'ref'), ('protocol', 'ref'), ('direction', 'id'), ('primary_metric', 'id'),
        ('metrics', 'metrics')), 'series'),
    ObservationProbe('host.operation', 'execution', (('operation', 'id'), ('status', 'id'),
        ('metrics', 'metrics')), 'table'),
    ObservationProbe('host.measurement', 'execution', (('step', 'number'), ('metrics', 'metrics'),
        ('series', 'id'), ('source', 'id')), 'series'),
    ObservationProbe('host.version_evaluated', 'quality', (), 'table'),
    ObservationProbe('host.version_selected', 'decision', (('step', 'number'), ('action', 'id'))),
)


class ObservationJournal:
    def __init__(self, root, run_id, probes=()):
        self.root, self.run_id = Path(root), run_id
        self.probes = {p.kind: p for p in (*CORE_PROBES, *probes)}
        self.path = self.root / JOURNAL
        self.ids = set()
        self.sequence = 0
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if self.path.exists():
            fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'r+b') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError('Monitor journal must be a regular file')
                while line := stream.readline(MAX_RECORD_BYTES + 1):
                    if len(line) > MAX_RECORD_BYTES:
                        raise ValueError('Oversized Monitor journal record')
                    if not line.endswith(b'\n'):
                        stream.truncate(stream.tell() - len(line))
                        break  # A killed writer can leave an uncommitted tail.
                    row = json.loads(line)
                    if (row['schema_version'] != SCHEMA or row['run_id'] != self.run_id
                            or row['sequence'] != self.sequence + 1):
                        raise ValueError('Monitor journal schema or identity does not match this run')
                    self.ids.add((row['kind'], row['scope'], row['event_id']))
                    self.sequence = row['sequence']
        self._loaded = True

    def append(self, payload):
        kind, event_id, scope, subject = (payload.get(key) for key in ('kind', 'event_id', 'scope', 'subject'))
        if not all(token(value) for value in (kind, event_id, scope)) or subject is not None and not token(subject):
            return None
        probe = self.probes.get(kind)
        if probe is None:
            return None
        self._load()
        identity = (kind, scope, event_id)
        if identity in self.ids:
            return None
        components = payload.get('components')
        row = {'schema_version': SCHEMA, 'run_id': self.run_id, 'sequence': self.sequence + 1,
               'event_id': event_id, 'scope': scope, 'kind': probe.kind,
               'category': probe.category, 'view': probe.view, 'subject': subject,
               'links': public_links(payload.get('links')), 'fields': probe.project(payload),
               'timestamp': datetime.now(timezone.utc).isoformat()}
        if token(payload.get('call_id')):
            row['call_id'] = payload['call_id']
        if probe.content and token(subject) and isinstance(components, dict):
            text = {k: v for k, v in list(components.items())[:8] if token(k) and not PRIVATE.search(k)
                    and isinstance(v, str) and 0 < len(v) <= 16384
                    and all(not 0xD800 <= ord(c) <= 0xDFFF for c in v)}
            if text:
                path = self.root / f'monitor-content.{subject}.json'
                data = json_bytes(text)
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                except FileNotFoundError:
                    atomic_write(path, data)
                else:
                    with os.fdopen(fd, 'rb') as stream:
                        if (not stat.S_ISREG(os.fstat(stream.fileno()).st_mode)
                                or stream.read(len(data) + 1) != data):
                            raise ValueError('Published version content differs for the same identity')
                row['content_available'] = True
        encoded = json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError('Oversized Monitor event')
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        with os.fdopen(fd, 'ab') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('Monitor journal must be a regular file')
            stream.write(encoded)
        self.ids.add(identity)
        self.sequence += 1
        return row
