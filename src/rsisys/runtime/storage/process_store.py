"""Durable JSON writes and advisory locks for owned backend process records."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write(path: Path, value):
    temporary = path.with_name(path.name + '.' + uuid4().hex)
    with temporary.open('w') as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
