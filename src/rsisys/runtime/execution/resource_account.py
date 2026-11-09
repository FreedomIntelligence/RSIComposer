"""Offline program reservations and measured allocation occupancy, not API calls."""
import fcntl
import json
from pathlib import Path
import time

from rsisys.runtime.storage.files import atomic_write, json_bytes


class ResourceAccount:
    def __init__(self, path, *, allocation, max_seconds=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.allocation, self.max_seconds = allocation, max_seconds
        if max_seconds is not None and (type(max_seconds) not in (int, float) or max_seconds <= 0):
            raise ValueError('Resource budget must be positive')
        self._update(lambda state: None)

    def _update(self, operation):
        with self.path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                'protocol': 'offline-resource-account/v1', 'allocation': self.allocation,
                'max_seconds': self.max_seconds, 'jobs': {}}
            if state['allocation'] != self.allocation or state['max_seconds'] != self.max_seconds:
                raise ValueError('Cannot change or reset a resource account')
            result = operation(state)
            atomic_write(self.path, json_bytes(state))
            return result

    def reserve(self, ident, timeout):
        def reserve(state):
            # Unconfirmed starts keep their complete reservation, including
            # after restart. Completed work charges measured program occupancy.
            used = sum(row.get('elapsed_seconds', row['reserved_seconds']) for row in state['jobs'].values())
            if self.max_seconds is not None and used + timeout > self.max_seconds:
                raise PermissionError('Offline resource account exhausted')
            if ident in state['jobs']:
                raise ValueError('Resource reservation identity already exists')
            state['jobs'][ident] = {'state': 'reserved', 'reserved_seconds': timeout, 'started_at': time.time()}
        self._update(reserve)

    def finish(self, ident, *, elapsed, status):
        def finish(state):
            row = state['jobs'][ident]
            if row['state'] != 'reserved':
                return
            row.update(state=status, elapsed_seconds=max(0, elapsed), ended_at=time.time())
        self._update(finish)
