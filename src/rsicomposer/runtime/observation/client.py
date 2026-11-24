"""Log anything to the run's board from any process: ``rsicomposer.log({"train/loss": 0.4}, step=10)``.

Standard library only, so worker interpreters with their own dependency sets
can import it. Inside the host process the values go straight to the run's
observer. In any other process started by the run they are appended to a
per-process file in the run's inbox, which the host tails and publishes.
Outside a monitored run this is a no-op returning False, like observe().
"""
import json
import math
import os
from pathlib import Path
import time

INBOX_ENV = 'RSISYS_LOG_INBOX'
OBSERVATION_URL_ENV = 'RSISYS_OBSERVATION_URL'
TEXT_LIMIT = 2000


def split(values):
    """Finite numbers become metrics, strings become log lines; anything else is rejected."""
    if not isinstance(values, dict) or not values:
        raise ValueError('log() needs a nonempty mapping of names to numbers or text')
    metrics, notes = {}, {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError('log() names must be nonempty strings')
        if isinstance(value, str):
            notes[name] = value[:TEXT_LIMIT]
        elif type(value) in (int, float) and math.isfinite(value):
            metrics[name] = value
        elif hasattr(value, 'item') and not isinstance(value, bool):  # numpy / torch scalars
            number = value.item()
            if type(number) not in (int, float) or not math.isfinite(number):
                raise ValueError(f'log() value for {name!r} is not a finite number')
            metrics[name] = number
        else:
            raise ValueError(f'log() value for {name!r} must be a finite number or text')
    return metrics, notes


def log(values, *, step=None, series=None):
    """Record scalars (charted) and text (shown in the run log) for the current run.

    `series` names one line of a chart: log the same metric from each GPU, worker or
    seed with a different series and the board draws them together, with their mean.
    """
    from . import current_observations, observations_suppressed
    if observations_suppressed():
        return False
    metrics, notes = split(values)
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError('step must be a nonnegative integer')
    if series is not None and (not isinstance(series, str) or not 0 < len(series) <= 100):
        raise ValueError('series must be a short nonempty string')
    current = current_observations()
    if current is not None:
        current.dispatch(current.log_mixed, metrics, notes, step=step, series=series)
        return True
    url, token = os.environ.get('RSISYS_LOG_URL'), os.environ.get('RSISYS_LOG_TOKEN')
    if url and token:
        from rsicomposer.runtime.execution.remote_services import post_worker
        post_worker(url, token, {'values': {**metrics, **notes}, 'step': step, 'series': series})
        return True
    inbox = os.environ.get(INBOX_ENV)
    if not inbox:
        return False
    record = {'metrics': metrics, 'notes': notes, 'step': step, 'series': series, 'time': time.time(),
              'pid': os.getpid()}
    # One file per process: appends from different workers never interleave.
    with (Path(inbox) / f'{os.getpid()}.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    return True


def worker_observation(wire):
    url, token = os.environ.get(OBSERVATION_URL_ENV), os.environ.get('RSISYS_LOG_TOKEN')
    if url and token:
        from rsicomposer.runtime.execution.remote_services import post_worker
        post_worker(url, token, {'fact': json.loads(wire)}, timeout=.25)
        return True
    inbox = os.environ.get(INBOX_ENV)
    if not inbox:
        return False
    # A single append per fact, so parallel threads in one worker cannot mix records.
    path = Path(inbox) / f'{os.getpid()}.jsonl'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, 'ab') as stream:
        stream.write(b'{"fact":' + wire.encode('ascii') + b'}\n')
    return True
