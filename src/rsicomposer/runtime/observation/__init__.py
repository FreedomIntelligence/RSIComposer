"""Run-scoped, opt-in observation; probes never choose RSI actions."""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import os
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from typing import Protocol
import math
from pathlib import Path
from queue import Empty, Full, Queue
import re
import threading
from time import monotonic
import warnings

_active = ContextVar('rsisys_observation', default=None)
_operation = ContextVar('rsisys_observed_operation', default=None)
_suppressed = ContextVar('rsisys_observations_suppressed', default=False)
_NAME = re.compile(r'[\w./-]{1,100}\Z')


def metrics(values):
    if not isinstance(values, Mapping) or not values:
        raise ValueError('Observation requires a nonempty metric mapping')
    result = {}
    for name, value in values.items():
        if (not isinstance(name, str) or not _NAME.fullmatch(name)
                or type(value) not in (int, float)
                or isinstance(value, int) and abs(value) > 2**53 or not math.isfinite(value)):
            raise ValueError('Metrics require names and finite JSON-safe numbers')
        result[name] = value
    return result


@dataclass
class Probe:
    name: str
    callback: object
    on: str
    every: int
    seen: int = 0
    calls: int = 0
    failures: int = 0


class Observations:
    """Run-owned observer; monitored applications dispatch through a bounded queue."""
    def __init__(self, publisher, sink=None):
        self.publisher, self.sink = publisher, sink
        self.probes = {}
        self.sequence = 0
        self.dispatching = False
        # Worker processes log into an inbox that a host thread drains, so the
        # publisher is shared between threads; this lock serialises its users.
        self.lock = publisher.lock if hasattr(publisher, 'lock') else threading.RLock()
        self._offsets = {}
        self._queue = None
        self._thread = None
        self._stop = threading.Event()
        self._loss_lock = threading.Lock()
        self._dropped = 0
        self._final = None

    def degraded(self, reason, *, dropped=0):
        self.publisher.data['observation_status'] = 'degraded'
        if dropped:
            data = self.publisher.data
            data['observation_dropped'] = data.get('observation_dropped', 0) + dropped
        self.publisher.log(reason, source='observation', level='ERROR')
        self.publisher.publish()

    def dispatch(self, callback, *args, **kwargs):
        if self._queue is None or threading.current_thread() is self._thread:
            return callback(*args, **kwargs)
        try:
            if self._queue.full():
                raise Full
            self._queue.put_nowait((callback, deepcopy(args), deepcopy(kwargs), _operation.get()))
        except Exception:
            with self._loss_lock:
                self._dropped += 1
            return False
        return True

    def start_background(self, inbox):
        self._queue = Queue(maxsize=1024)

        def consume():
            token = _active.set(self)
            tick = monotonic()
            try:
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        callback, args, kwargs, call_id = self._queue.get(timeout=.2)
                    except Empty:
                        callback = None
                    try:
                        with observe_operation(call_id if callback else None):
                            if callback:
                                callback(*args, **kwargs)
                    except (Exception, SystemExit) as error:
                        self.degraded('Observation failed: ' + type(error).__name__)
                    with self._loss_lock:
                        dropped, self._dropped = self._dropped, 0
                    if dropped:
                        self.degraded('Observation queue overflow', dropped=dropped)
                    if monotonic() >= tick:
                        if inbox is not None:
                            self.drain(inbox)
                        if self.publisher._needs_publish:
                            self.publisher.publish(force=True)
                        tick = monotonic() + .5
                if inbox is not None:
                    self.drain(inbox)
                with self._loss_lock:
                    dropped, self._dropped = self._dropped, 0
                if dropped:
                    self.degraded('Observation queue overflow', dropped=dropped)
                if self._final is not None:
                    callback, args = self._final
                    callback(*args)
                self.publisher.publish(force=True)
            except Exception:
                logging.getLogger(__name__).exception('Monitor stopped processing observations')
            finally:
                _active.reset(token)

        self._thread = threading.Thread(target=consume, name='rsi-observer', daemon=True)
        self._thread.start()

    def close_background(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            # The experiment has ended; a slow custom probe cannot keep it alive.
            self.publisher.incomplete = True
            logging.getLogger(__name__).warning('Monitor queue did not drain before the run ended')

    def finish(self, callback, *args):
        if self._queue is None:
            callback(*args)
        else:
            self._final = (callback, args)

    def register(self, name, callback, *, on='feedback', every=1):
        if (not isinstance(name, str) or not _NAME.fullmatch(name) or not callable(callback)
                or not isinstance(on, str) or not _NAME.fullmatch(on)
                or type(every) is not int or every < 1):
            raise ValueError('Probe requires a name, callable, event and positive interval')
        if name in self.probes:
            raise ValueError(f'Probe already registered: {name}')
        self.probes[name] = Probe(name, callback, on, every)

    def remove(self, name):
        self.probes.pop(name, None)

    def log_mixed(self, numbers, notes, *, step=None, series=None, source='log', publish=True):
        """Scalars are charted; text goes to the run log. Used by rsicomposer.log()."""
        with self.lock:
            for name, text in notes.items():
                self.publisher.log(f'{name}: {text}', source=source)
            if numbers:
                self.log(numbers, step=step, series=series, publish=False)
            if publish:
                self.publisher.publish()

    def drain(self, inbox):
        """Publish what other processes of this run have logged since the last call."""
        changed = False
        for path in sorted(Path(inbox).glob('*.jsonl')):
            try:
                with path.open('rb') as file:
                    file.seek(self._offsets.get(path, 0))
                    while True:
                        line = file.readline(1024 * 1024)
                        if not line.endswith(b'\n'):
                            break  # A partial line is still being written.
                        self._offsets[path] = file.tell()
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict) and isinstance(row.get('fact'), dict):
                                self.publisher.fact(row['fact'], publish=False)
                                changed = True
                                continue
                            self.log_mixed(metrics(row['metrics']) if row.get('metrics') else {},
                                           {str(k): str(v) for k, v in (row.get('notes') or {}).items()},
                                           step=row.get('step'), series=row.get('series'),
                                           source=f"worker-{row.get('pid')}", publish=False)
                            changed = True
                        except (ValueError, KeyError, TypeError, AttributeError) as error:
                            with self.lock:
                                self.publisher.log('Rejected a logged record: ' + type(error).__name__,
                                                   source='observation', level='ERROR')
            except OSError:
                continue
        if changed:
            with self.lock:
                self.publisher.publish()

    def log(self, values, *, step=None, definitions=None, decision_id=None, call_id=None, series=None, publish=True):
        with self.lock:
            return self._log(values, step=step, definitions=definitions, decision_id=decision_id,
                             call_id=call_id, series=series, publish=publish)

    def _log(self, values, *, step, definitions, decision_id, call_id, series, publish):
        if series is not None and (not isinstance(series, str) or not _NAME.fullmatch(series)):
            raise ValueError('Observation series must be a short name')
        values = metrics(values)
        if step is not None and (type(step) is not int or not 0 <= step <= 2**53):
            raise ValueError('Observation step must be a nonnegative JSON-safe integer')
        if decision_id is None and call_id is None:
            call_id = _operation.get()
        self.publisher.observe_metrics(values, step=step, definitions=definitions, decision_id=decision_id,
                                       call_id=call_id,
                                       **({'series': series} if series is not None else {}))
        if self.sink is not None:
            try:
                self.sink.log(values, sequence=self.sequence, step=step)
            except Exception as error:
                self.publisher.data['observation_status'] = 'degraded'
                self.publisher.log('Observation sink failed: ' + type(error).__name__, source='observation', level='ERROR')
                warnings.warn('Observation sink failed; local observations retained', RuntimeWarning, stacklevel=2)
        self.sequence += 1
        if publish:
            self.publisher.publish()

    def trigger(self, event, payload=None):
        with self.lock:
            return self._trigger(event, payload)

    def _trigger(self, event, payload):
        if self.dispatching:
            raise RuntimeError('Recursive probe triggering is not supported')
        self.dispatching = True
        try:
            for probe in list(self.probes.values()):
                if probe.on != event:
                    continue
                probe.seen += 1
                if (probe.seen - 1) % probe.every:
                    continue
                probe.calls += 1
                try:
                    result = probe.callback(deepcopy(payload or {}))
                    if result is not None:
                        self.log({f'{probe.name}/{key}': value for key, value in metrics(result).items()},
                                 step=probe.calls - 1)
                except (Exception, SystemExit) as error:
                    probe.failures += 1
                    self.degraded(f'Probe {probe.name} failed: {type(error).__name__}')
            self.publisher.data['probes'] = [
                {'name': p.name, 'on': p.on, 'every': p.every, 'calls': p.calls, 'failures': p.failures}
                for p in self.probes.values()]
            self.publisher.publish()
        finally:
            self.dispatching = False


class RunReporter(Protocol):
    """What the host, harnesses and Labs may report about a run; implemented by the snapshot publisher."""
    def start(self, decision_id, operation, arguments=None, details=None, *, call_id=None): ...
    def record(self, event): ...
    def report_samples(self, records, stage=None): ...
    def observe_devices(self, devices): ...
    def observe_model_calls(self, attempted, limit): ...
    def read_training_log(self, path, stage=None): ...
    def version(self, version_id, *, label, round_index=None, parent_ids=None, source_decision_ids=()): ...
    def version_evaluation(self, version_id, evaluation_id): ...
    def version_selection(self, version_id, step, *, scope='run', event_id=None, action='selected'): ...


def current_observations():
    """Get this run's observer, or None outside a monitored execution."""
    return None if observations_suppressed() else _active.get()


def observations_suppressed():
    return _suppressed.get()


@contextmanager
def suppress_observations():
    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)


@contextmanager
def activate(observations, inbox=None, *, background=False):
    """Make one observer current for the enclosed run; the host calls this, never a Lab.

    With an inbox directory, child processes inherit its path and their
    rsicomposer.log() calls are drained about once a second while the run executes.
    """
    from .client import INBOX_ENV
    token = _active.set(observations)
    previous, stop, reader = os.environ.get(INBOX_ENV), threading.Event(), None
    if inbox is not None:
        try:
            Path(inbox).mkdir(parents=True, exist_ok=True)
        except OSError:
            inbox = None
            if background:
                observations.degraded('Could not open the worker observation inbox')
        else:
            os.environ[INBOX_ENV] = str(inbox)

    if background:
        observations.start_background(inbox)
    elif inbox is not None:
        def poll():
            while not stop.wait(1.0):
                observations.drain(inbox)
        reader = threading.Thread(target=poll, name='rsisys-log-inbox', daemon=True)
        reader.start()
    try:
        yield observations
    finally:
        if background:
            observations.close_background()
        if reader is not None:
            stop.set()
            reader.join(timeout=10)
            observations.drain(inbox)  # Nothing logged just before exit is lost.
        if inbox is not None:
            if previous is None:
                os.environ.pop(INBOX_ENV, None)
            else:
                os.environ[INBOX_ENV] = previous
        _active.reset(token)


class _Locked:
    """Serialise reports with the inbox thread without changing the reporter interface."""
    def __init__(self, observations):
        self._observations = observations

    def __getattr__(self, name):
        method, lock = getattr(self._observations.publisher, name), self._observations.lock

        def call(*args, **kwargs):
            with lock:
                return method(*args, **kwargs)
        return lambda *args, **kwargs: self._observations.dispatch(call, *args, **kwargs)


def _reporter() -> RunReporter | None:
    current = current_observations()
    return _Locked(current) if current is not None else None


@contextmanager
def observe_operation(call_id):
    token = _operation.set(call_id)
    try:
        yield
    finally:
        _operation.reset(token)


def report_stage_start(decision_id, operation, arguments=None, details=None, *, call_id=None):
    """An operation is about to execute. No-op outside a monitored run, like every report_* function.

    `details` may carry the decision's stated reason and its declared relations to earlier
    decisions (depends_on_decision_ids, source_decision_ids, research_attempt_id/title).
    """
    reporter = _reporter()
    if reporter is not None:
        reporter.start(decision_id, operation, arguments, details, call_id=call_id)


def report_step(event):
    """One Runner step event (decision, progress, receipt or feedback)."""
    reporter = _reporter()
    if reporter is not None:
        reporter.record(event)


def report_samples(records):
    """Per-sample evaluation results for the operation currently executing."""
    reporter = _reporter()
    if reporter is not None:
        reporter.report_samples(records, _operation.get())


def report_devices(devices):
    reporter = _reporter()
    if reporter is not None:
        reporter.observe_devices(devices)


def report_training_log(path):
    """New JSON lines of a worker training log, attributed to the operation currently executing."""
    reporter = _reporter()
    if reporter is not None:
        reporter.read_training_log(path, _operation.get())


def report_model_calls(attempted, limit):
    reporter = _reporter()
    if reporter is not None:
        reporter.observe_model_calls(attempted, limit)


def report_version(version_id, *, label, round_index=None, parent_ids=None, source_decision_ids=()):
    """Observe a candidate using a stable public ID and label. None means unreported ancestry;
    an empty tuple explicitly declares no parents. Evidence references are not parents.
    """
    reporter = _reporter()
    if reporter is not None:
        reporter.version(version_id, label=label, round_index=round_index, parent_ids=parent_ids,
                         source_decision_ids=source_decision_ids)


def report_version_evaluation(version_id, evaluation_id):
    """Associate a version with an existing evaluation (normally its operation call ID)."""
    reporter = _reporter()
    if reporter is not None:
        reporter.version_evaluation(version_id, evaluation_id)


def report_version_selection(version_id, *, step, scope='run', event_id=None, action='selected'):
    """Observe a real incumbent update or final selection. This does not adopt a version."""
    reporter = _reporter()
    if reporter is not None:
        reporter.version_selection(version_id, step, scope=scope, event_id=event_id, action=action)


def observe(values, *, step=None, definitions=None):
    """Log custom scalar observations to the run's local metric journal.

    Returns False outside a monitored run. Does not interpret or adopt metrics.
    From a worker process use rsicomposer.log(), which also accepts text.
    """
    current = current_observations()
    if current is None:
        from .client import log
        return log(values, step=step)
    current.dispatch(current.log, values, step=step, definitions=definitions)
    return True


def register_probe(name, callback, *, on='feedback', every=1):
    current = current_observations()
    if current is None:
        raise RuntimeError('Register probes inside a monitored run or a --probe plugin')
    current.dispatch(current.register, name, callback, on=on, every=every)


def trigger_probe(event, payload=None):
    current = current_observations()
    if current is None:
        return False
    current.dispatch(current.trigger, event, payload)
    return True
