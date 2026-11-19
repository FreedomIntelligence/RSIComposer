"""Checkpoint exchanges replay exactly; later records belong to an abandoned branch.

Records are written atomically and chained by SHA-256 from a base digest. Only
records named by the restored checkpoint are committed. Recovery validates that
entire prefix before discarding later files and returning to live execution.
"""
import hashlib
import json
from pathlib import Path
import re

from rsisys.runtime.storage.files import atomic_write


class ReplayMismatchError(RuntimeError):
    """A committed exchange cannot be reproduced; it is not a candidate failure."""


def _record(data):
    try:
        record = json.loads(data)
        if not isinstance(record, dict) or set(record) != {'kind', 'request', 'response'}:
            raise ValueError('Invalid exchange shape')
        return record
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError('Invalid committed replay exchange') from error


def request_digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _link(chain, data):
    return hashlib.sha256((chain + hashlib.sha256(data).hexdigest()).encode()).hexdigest()


def valid_replay_point(point):
    return (isinstance(point, dict) and set(point) == {'records', 'chain'}
            and type(point['records']) is int and point['records'] >= 0
            and isinstance(point['chain'], str) and len(point['chain']) == 64
            and all(c in '0123456789abcdef' for c in point['chain']))


class ReplayLog:
    def __init__(self, directory, base, *, point=None):
        if point is not None and not valid_replay_point(point):
            raise ValueError('Invalid replay resume point')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.base, self.chain, self.count = base, base, 0
        self._pending = []
        if point is not None:
            for index in range(point['records']):
                path = self.directory / f'{index:05d}.json'
                if not path.is_file():
                    raise ValueError(f'Replay record is missing: {path}')
                data = path.read_bytes()
                self.chain = _link(self.chain, data)
                self._pending.append(_record(data))
            if self.chain != point['chain']:
                raise ValueError(f'Replay records in {self.directory} differ from the checkpoint')
            self.count = point['records']
        # Validation above must complete before any abandoned files are removed.
        # Missing or corrupt files beyond this checkpoint are not committed data.
        for path in self.directory.iterdir():
            if re.fullmatch(r'[0-9]{5,}\.json', path.name) and int(path.stem) >= self.count:
                path.unlink()

    @classmethod
    def resume(cls, directory, base, point):
        """Validate and replay precisely the exchanges named by the checkpoint."""
        if not valid_replay_point(point):
            raise ValueError('Invalid replay resume point')
        return cls(directory, base, point=point)

    def point(self):
        if self._pending:
            raise RuntimeError('A resume point is undefined while recorded exchanges remain unreplayed')
        return {'records': self.count, 'chain': self.chain}

    @property
    def replaying(self):
        return bool(self._pending)

    def take(self, kind, request):
        """Reuse a committed checkpoint exchange, otherwise request live work."""
        if not self._pending:
            return None
        index = self.count - len(self._pending)
        record = self._pending[0]
        if record.get('kind') != kind or record.get('request') != request_digest(request):
            raise ReplayMismatchError(f'Resumed worker diverged from its recorded run at record {index} '
                                      f'({kind}); the upstream state cannot be reproduced faithfully')
        self._pending.pop(0)
        return record['response']

    def assert_tail_consumed(self):
        if self._pending or (self.directory / f'{self.count:05d}.json').exists():
            raise ReplayMismatchError('Resumed worker left committed replay exchanges unconsumed')

    def record(self, kind, request, response):
        if self._pending:
            raise RuntimeError('Cannot record a live exchange while recorded exchanges remain')
        record = {'kind': kind, 'request': request_digest(request), 'response': response}
        data = json.dumps(record, allow_nan=False).encode()
        path = self.directory / f'{self.count:05d}.json'
        if path.exists():
            raise ReplayMismatchError('Another writer created a replay exchange at this position')
        atomic_write(path, data)
        self.chain = _link(self.chain, data)
        self.count += 1
