"""Durable files and process identity for one detached native training job."""
import json
import os
from pathlib import Path
import time
import sys
import uuid

from rsicomposer.api import JobProgress


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def process_identity(pid):
    """Linux start ticks reject PID reuse; a zombie is no longer running."""
    if not sys.platform.startswith('linux'):
        from rsicomposer.api import process_identity as host_identity
        return host_identity(pid)
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == 'Z' else fields[19]


def read_events(directory, *, after=0):
    # One atomic file per event: readers never observe a partial JSON line.
    return tuple(json.loads(path.read_text()) for path in sorted((Path(directory) / 'events').glob('*.json'))
                 if int(path.stem) > after)


def control_checkpoint(directory, call_id, sequence, action):
    events = read_events(directory, after=sequence - 1)
    if not events or events[0]['sequence'] != sequence or events[0]['kind'] != 'checkpoint':
        raise ValueError('Control must refer to an existing native checkpoint event')
    path = Path(directory) / 'controls' / f'{sequence:012d}.json'
    value = dict(call_id=call_id, sequence=sequence, action=action)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Native checkpoint already has a different control')
    else:
        atomic_json(path, value)


def latest_progress(directory):
    events = read_events(directory)
    return JobProgress(**events[-1]['progress']) if events else None


def job_expired(descriptor, timeout):
    return time.time() - descriptor['submitted_at'] > timeout
