"""Host lifecycle for dependency-isolated upstream research agents.

Methods supply bootstrap/selection rules and translate native callback payloads.
The host owns worker transport, metered credentials, correlation and recovery.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import Decision, OperationCall
from rsisys.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.execution.replay import request_digest, valid_replay_point
from rsisys.runtime.execution.research_bridge import ResearchBridge
from rsisys.runtime.execution.workers import worker_environment
from rsisys.runtime.execution.upstream_worker import UpstreamWorker
from rsisys.runtime.storage.files import atomic_write, json_bytes


_COMMON = {'worker_config', 'request_digest', 'pending', 'pending_origin', 'last_event', 'last_reply',
           'cached_event', 'deferred_feedback', 'job', 'started', 'worker_finished', 'closed', 'remaining_seconds',
           'origin_attribution', 'stopped', 'failure'}


class UpstreamAgentPolicy:
    def __init__(self, context, config, *, root, worker_module, worker_python=None, researcher,
                 timeout, bridge_factory=ResearchBridge, resume_validator=valid_replay_point,
                 location_fields=('root', 'directory')):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Upstream worker timeout must be finite and positive')
        if not callable(resume_validator):
            raise TypeError('Upstream resume validator must be callable')
        self.config, self.models = deepcopy(config), context.models
        self.artifacts = context.artifacts
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=bool(getattr(context, 'restoring', False)))
        self.worker_module, self.worker_python = worker_module, worker_python or sys.executable
        self.researcher, self.timeout = deepcopy(researcher), timeout
        self.bridge_factory, self.resume_validator = bridge_factory, resume_validator
        self.location_fields = tuple(location_fields)
        self.bridge = self._endpoint = None
        self.pending = self.pending_origin = None
        self.last_event = self.last_reply = self.cached_event = self.deferred_feedback = None
        self.worker_config = self.request_digest = self.job = None
        self.started = self.worker_finished = self.closed = False
        self.deadline = self.origin_attribution = None
        self.stopped = self.failure = None
        self._joining = False
        self._expected_event = None
        self._files = {}
        self._remaining_seconds = None
        self._file_cache, self._content_refs = {}, {}

    def before_upstream(self, window):
        return None

    def event_operation(self, event):
        raise NotImplementedError

    def receipt_message(self, event, data):
        raise NotImplementedError

    def finished_event(self, event):
        return Decision('finish', reason='Upstream research completed')

    def method_state(self):
        return {}

    def restore_method_state(self, state):
        if state != {}:
            raise ValueError('Unexpected upstream method state')

    def invoke(self, name, arguments, *, reason=''):
        if self.pending is not None:
            raise RuntimeError('Upstream controller already has an outstanding Lab request')
        self.pending = OperationCall(name, deepcopy(arguments))
        self.pending_origin = 'method'
        return Decision.invoke(deepcopy(self.pending), reason=reason)

    def consume(self, window):
        if self.pending is None:
            raise RuntimeError('No upstream Lab request is awaiting feedback')
        feedbacks = tuple(window.feedbacks)
        matches = [row for row in feedbacks if row.call_id == self.pending.call_id]
        if self.deferred_feedback is not None:
            matches.append(self.deferred_feedback)
        if not matches or any(row != matches[0] for row in matches[1:]):
            raise RuntimeError('Upstream agent requires exactly its matching terminal Lab receipt')
        result = matches[0]
        if result.status != ExecutionStatus.COMPLETED or result.simulated:
            raise RuntimeError('Upstream agent requires a real successful Lab receipt: ' + str(result.error))
        self.pending = self.pending_origin = self.deferred_feedback = None
        return deepcopy(result.data)

    def _semantic_request(self, config):
        return UpstreamWorker.semantic_config({key: value for key, value in config.items()
            if key not in self.location_fields})

    @staticmethod
    def _worker_config(config):
        if not isinstance(config, dict):
            raise ValueError('Upstream worker config must be a mapping')
        result = {}
        for key, value in config.items():
            if key in {'endpoint', 'endpoints', 'api_key', 'credential_env'}:
                continue
            result[key] = UpstreamAgentPolicy._worker_config(value) if isinstance(value, dict) else deepcopy(value)
        json_bytes(result)
        return result

    def start_worker(self, config):
        if self.started or self.closed:
            raise RuntimeError('Upstream worker was already started or closed')
        self.worker_config = self._semantic_request(self._worker_config(config))
        self.request_digest = request_digest(self._semantic_request(self.worker_config))
        self.started = True
        self.job = JobRef(uuid4().hex, 'upstream-agent')
        self._launch()

    def _launch(self):
        if self.bridge is not None:
            return
        if self.closed or self.worker_finished or not self.started:
            raise RuntimeError('No upstream worker is available to launch')
        original = self.root / 'worker-request.json'
        if original.exists():
            saved = json.loads(original.read_text())
            saved = {key: value for key, value in saved.items()
                     if key not in ('upstream_root', 'task_file', 'python', 'worker_python')}
            if request_digest(self._semantic_request(saved)) != self.request_digest:
                raise ValueError('Upstream worker request changed after the checkpoint')
        config = deepcopy(self.worker_config)
        for field in self.location_fields:
            config[field] = str(self.root)
        for field in ('upstream_root', 'task_file'):
            if field in self.config:
                config[field] = str(Path(self.config[field]).resolve())
        if self._semantic_request(config) != self.worker_config:
            raise ValueError('Current upstream inputs differ from their saved research identity')
        resume = self._expected_event.get('resume') if self._joining else None
        if resume is not None:
            config['resume'] = deepcopy(resume)
        if original.exists():
            attempt = next(n for n in range(1, 10_000) if not (self.root / f'worker-resume-{n}.json').exists())
            request, log = self.root / f'worker-resume-{attempt}.json', self.root / f'worker-resume-{attempt}.log'
        else:
            request, log = original, self.root / 'worker.log'
        atomic_write(request, json_bytes(config))
        if self.origin_attribution is None:
            self.origin_attribution = deepcopy(self.models.direct_attribution())
        self._endpoint = self.models.background_endpoint(self.researcher, purpose='researcher',
                                                        attribution=self.origin_attribution)
        issued = self._endpoint.environment()
        env = {key: value for key, value in worker_environment(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='').items()
               if not re.search(r'(API_KEY|ACCESS_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)', key, re.I)}
        env.update(issued)
        try:
            self.bridge = self.bridge_factory([self.worker_python, '-m', self.worker_module, str(request)],
                                              log=log, env=env)
        except BaseException:
            self._close_worker()
            raise
        remaining = self.timeout if self._remaining_seconds is None else self._remaining_seconds
        self.deadline = time.monotonic() + remaining
        self._remaining_seconds = None

    def _close_worker(self):
        try:
            if self.bridge is not None:
                self.bridge.close()
                self.bridge = None
        finally:
            if self._endpoint is not None:
                self._endpoint.revoke()
                self._endpoint = None

    def _pump(self):
        if self.closed or self.worker_finished:
            return self.cached_event is not None
        if self.cached_event is not None and not self._joining:
            return True
        if self.bridge is None:
            self._launch()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('Upstream worker timed out; inspect ' + str(self.root))
        event = self.bridge.poll(0)
        if event is None:
            return False
        if self._joining:
            if event != self._expected_event:
                raise RuntimeError('Resumed upstream worker did not reproduce the checkpointed request; only an exact rejoin is supported')
            self._joining = False
            if self.cached_event is not None:
                self.cached_event = deepcopy(event)
                return True
            if self.pending_origin == 'worker':
                self.deadline = None
                return True
            if self.last_reply is None:
                raise ValueError('Resumed upstream continuation lacks its acknowledged callback reply')
            self.bridge.send(deepcopy(self.last_reply))
            return False
        self.cached_event = deepcopy(event)
        return True

    def _wait(self):
        if self.job is None:
            raise RuntimeError('Upstream wait has no job identity')
        return Decision('wait', job=self.job, reason='Waiting for upstream research worker')

    def decide(self, window):
        if self.stopped is not None:
            return Decision('finish', reason=self.stopped)
        initial = self.before_upstream(window)
        if initial is not None:
            return initial
        if not self.started:
            raise RuntimeError('Upstream method must start its worker after bootstrap')
        if self.pending_origin == 'worker' and window.feedbacks:
            matches = [row for row in window.feedbacks if row.call_id == self.pending.call_id]
            if not matches or any(row != matches[0] for row in matches[1:]):
                raise RuntimeError('Upstream agent requires its matching terminal Lab receipt')
            self.deferred_feedback = deepcopy(matches[0])
        if self.pending_origin == 'worker':
            if self._joining and not self._pump():
                return self._wait()
            data = self.consume(window)
            message = self.receipt_message(deepcopy(self.last_event), data)
            json_bytes(message)
            self.last_reply = deepcopy(message)
            self.bridge.send(message)
            self.deadline = time.monotonic() + self.timeout
        if not self._pump():
            return self._wait()
        event, self.cached_event = self.cached_event, None
        if event is None:
            return self._wait()
        if event['kind'] == 'execute':
            if not self.resume_validator(event.get('resume')):
                raise ValueError('Upstream execute event lacks a valid resume point')
            call = self.event_operation(deepcopy(event))
            if not isinstance(call, OperationCall):
                raise TypeError('Upstream callback mapping must return an OperationCall')
            self.last_event, self.last_reply = deepcopy(event), None
            self.pending, self.pending_origin = deepcopy(call), 'worker'
            self.deadline = None
            return Decision.invoke(deepcopy(call), reason='Upstream execution callback')
        if event['kind'] != 'finished':
            raise ValueError('Unexpected upstream worker event')
        self.worker_finished = True
        self.deadline = None
        self._close_worker()
        return self.finished_event(deepcopy(event))

    def can_poll_job(self, job):
        return isinstance(job, JobRef) and job == self.job

    def poll_job(self, job):
        if not self.can_poll_job(job):
            raise ValueError('Upstream wait job identity does not match')
        if self.closed:
            return JobOutcome(job, JobStatus.CANCELLED)
        if self.failure is not None:
            return JobOutcome(job, JobStatus.FAILED, error=self.failure)
        try:
            ready = self._pump()
        except Exception as error:
            self.failure = self.stopped = f'{type(error).__name__}: {error}'
            self._close_worker()
            return JobOutcome(job, JobStatus.FAILED, error=self.failure)
        return JobOutcome(job, JobStatus.SUCCEEDED, result={'ready': True}) if ready else JobOutcome(job, JobStatus.RUNNING)

    def cancel_job(self, job):
        if not self.can_poll_job(job):
            raise ValueError('Upstream wait job identity does not match')
        self.stopped = self.stopped or 'Upstream research cancelled'
        self.close()
        return JobOutcome(job, JobStatus.CANCELLED)

    def _safe_boundary(self):
        return (not self.started or self.worker_finished or self.closed or self.pending_origin == 'worker'
                or self.cached_event is not None and self.cached_event.get('kind') in {'execute', 'finished'})

    def _bundle(self):
        files = {}
        for path in sorted(self.root.rglob('*')):
            if path.is_symlink():
                raise ValueError('Upstream run files may not contain symlinks')
            if (not path.is_file() or any(part.startswith('.') for part in path.relative_to(self.root).parts)
                    or path.name.startswith(('worker-request', 'worker-resume', 'worker.log', 'worker.stderr.log'))):
                continue
            name = path.relative_to(self.root).as_posix()
            stat = path.stat()
            fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = self._file_cache.get(name)
            if cached is not None and cached[0] == fingerprint:
                ref = cached[1]
            else:
                data = path.read_bytes()
                digest = sha256(data).hexdigest()
                ref = self._content_refs.get(digest)
                if ref is None:
                    ref = self.artifacts.put(data, kind='upstream-checkpoint')
                    self._content_refs[digest] = ref
                self._file_cache[name] = (fingerprint, ref)
            files[name] = ref
        self._file_cache = {name: entry for name, entry in self._file_cache.items() if name in files}
        return files

    def state(self):
        if self._safe_boundary():
            self._files = self._bundle()
        common = {name: deepcopy(getattr(self, name)) for name in _COMMON if name != 'remaining_seconds'}
        common['remaining_seconds'] = (max(0., self.deadline - time.monotonic()) if self.deadline is not None
                                       else self._remaining_seconds)
        return {'schema': 2, 'common': common, 'method': deepcopy(self.method_state()),
                'files': deepcopy(self._files)}

    def restore(self, state):
        if self.bridge is not None or self.started:
            raise RuntimeError('Upstream state must be restored before its worker starts')
        if not isinstance(state, dict) or set(state) != {'schema', 'common', 'method', 'files'} or state['schema'] != 2:
            raise ValueError('Unexpected upstream policy state schema')
        common = state['common']
        if not isinstance(common, dict) or set(common) != _COMMON:
            raise ValueError('Unexpected upstream policy state fields')
        if any(type(common[name]) is not bool for name in ('started', 'worker_finished', 'closed')):
            raise ValueError('Invalid upstream policy lifecycle state')
        if common['pending'] is not None and not isinstance(common['pending'], OperationCall):
            raise ValueError('Invalid upstream pending request')
        if common['pending_origin'] not in (None, 'worker', 'method') or ((common['pending'] is None) != (common['pending_origin'] is None)):
            raise ValueError('Invalid upstream pending request owner')
        if common['job'] is not None and (not isinstance(common['job'], JobRef) or common['job'].backend != 'upstream-agent' or not common['job'].job_id):
            raise ValueError('Invalid upstream wait job identity')
        remaining = common['remaining_seconds']
        if remaining is not None and (type(remaining) not in (int, float) or not math.isfinite(remaining) or remaining < 0):
            raise ValueError('Invalid upstream remaining timeout')
        for name in ('last_event', 'cached_event'):
            event = common[name]
            if event is not None and (not isinstance(event, dict) or event.get('kind') not in {'execute', 'finished'}):
                raise ValueError('Invalid upstream saved event')
            if event is not None and event['kind'] == 'execute' and not self.resume_validator(event.get('resume')):
                raise ValueError('Invalid upstream saved resume point')
        if common['deferred_feedback'] is not None and not isinstance(common['deferred_feedback'], OperationResult):
            raise ValueError('Invalid upstream deferred receipt')
        if common['pending_origin'] == 'worker' and (common['last_event'] is None
                or common['last_event']['kind'] != 'execute' or common['last_reply'] is not None):
            raise ValueError('Outstanding upstream request does not match its callback boundary')
        deferred = common['deferred_feedback']
        if deferred is not None and (common['pending_origin'] != 'worker'
                or deferred.call_id != common['pending'].call_id):
            raise ValueError('Deferred upstream receipt does not match its callback request')
        for name in ('last_reply', 'worker_config', 'origin_attribution'):
            if common[name] is not None and not isinstance(common[name], dict):
                raise ValueError('Invalid upstream saved ' + name)
        if common['started'] and (common['job'] is None or common['worker_config'] is None
                or request_digest(self._semantic_request(common['worker_config'])) != common['request_digest']
                or self._semantic_request(common['worker_config']) != common['worker_config']):
            raise ValueError('Upstream worker request identity does not match')
        attribution = common['origin_attribution']
        if common['started'] and (not isinstance(attribution, dict)
                or set(attribution) != {'component', 'decision_id', 'call_id'}
                or not isinstance(attribution['component'], str) or not attribution['component']
                or any(value is not None and (not isinstance(value, str) or not value)
                       for value in attribution.values())):
            raise ValueError('Invalid upstream worker model attribution')
        if not isinstance(state['files'], dict):
            raise ValueError('Invalid upstream file bundle')
        checked = {}
        for name, ref in state['files'].items():
            path = Path(name)
            if (not isinstance(name, str) or not name or path.is_absolute() or '..' in path.parts
                    or path.as_posix() != name or not isinstance(ref, ArtifactRef)
                    or ref.kind != 'upstream-checkpoint' or not ref.digest):
                raise ValueError('Invalid upstream bundled file path or reference')
            data = self.artifacts.read(ref)
            if sha256(data).hexdigest() != ref.digest:
                raise ValueError('Upstream bundled file digest mismatch')
            target = self.root / path
            if any(parent.is_symlink() for parent in (target, *target.parents) if parent != self.root.parent):
                raise ValueError('Upstream bundled file escapes through a symlink')
            checked[name] = (ref, data)
        for name, (ref, data) in checked.items():
            target = self.root / name
            # Retain newer local files; worker replay discards only uncommitted
            # exchanges beyond the exact checkpoint point before continuing live.
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(target, data)
            self._content_refs[ref.digest] = ref
        self._files = {name: ref for name, (ref, _) in checked.items()}
        for name in _COMMON - {'remaining_seconds'}:
            setattr(self, name, deepcopy(common[name]))
        self._remaining_seconds = remaining
        self.deadline = None
        self.restore_method_state(deepcopy(state['method']))
        cached_finished = self.cached_event is not None and self.cached_event['kind'] == 'finished'
        expected = self.cached_event if self.cached_event is not None else self.last_event
        self._joining = bool(self.started and not self.worker_finished and not self.closed and expected is not None)
        if cached_finished:
            self._joining = False
        self._expected_event = deepcopy(expected) if self._joining else None
        if self.pending_origin == 'worker' and not self._joining:
            raise ValueError('Outstanding upstream callback lacks an exact rejoin event')

    def validate_snapshot(self):
        if not self._safe_boundary():
            raise RuntimeError('Upstream portable snapshot requires a callback boundary or a finished worker')

    def close(self):
        self._close_worker()
        self.closed = True
