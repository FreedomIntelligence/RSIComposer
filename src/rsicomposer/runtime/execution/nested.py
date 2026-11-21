"""Independent child runs with durable identities and parent-owned admission.

Composition is injected through nested_manifest; runtime never imports a
Registry, Application, factory closure or arbitrary Python serialization.
"""
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time

from rsicomposer.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
from .executor import WorkerSpec
from .errors import WorkerLaunchUncertainError
from .process_identity import process_identity, process_group_alive
from .resources import ResourceRequest


class _LocalAttachment:
    """Observe the original supervisor; an absent PID never proves cleanup."""
    def __init__(self, descriptor):
        self.directory = Path(descriptor['directory'])
        self.pid, self.identity = descriptor['pid'], descriptor['identity']

    def poll(self):
        result = self.directory / 'result.json'
        if result.exists():
            return json.loads(result.read_text())['returncode']
        if process_identity(self.pid) == self.identity:
            return None
        raise RuntimeError('Nested supervisor disappeared before cleanup publication; effects remain uncertain')

    def terminate(self, grace_seconds=10):
        # A newly exec'ed supervisor has not installed its signal handlers yet.
        # Wait for its child/terminal publication before asking it to cancel.
        publication_deadline = time.monotonic() + min(5, grace_seconds)
        while not (self.directory / 'child.json').exists() and not (self.directory / 'result.json').exists():
            if process_identity(self.pid) != self.identity:
                raise RuntimeError('Nested supervisor exited before publishing child identity; cleanup is uncertain')
            if time.monotonic() >= publication_deadline:
                raise RuntimeError('Nested supervisor has not published child identity; cleanup is uncertain')
            time.sleep(.02)
        if (self.directory / 'result.json').exists():
            return
        if process_identity(self.pid) == self.identity:
            os.kill(self.pid, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline:
                if (self.directory / 'result.json').exists():
                    return
                time.sleep(.02)
        path = self.directory / 'child.json'
        if not path.exists():
            raise RuntimeError('Nested worker identity was never published; cleanup is uncertain')
        child = json.loads(path.read_text())
        if process_identity(child['pid']) == child['identity']:
            try:
                os.killpg(child['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process_group_alive(child['pid']):
            raise RuntimeError('Nested worker identity changed; refusing to signal an unrelated process group')
        deadline = time.monotonic() + 5
        while process_group_alive(child['pid']):
            if time.monotonic() >= deadline:
                raise RuntimeError('Nested worker process group has not exited')
            time.sleep(.02)
        if process_identity(self.pid) == self.identity:
            os.kill(self.pid, signal.SIGCONT)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if (self.directory / 'result.json').exists():
                    return
                time.sleep(.02)
            raise RuntimeError('Nested supervisor has not confirmed cleanup')
        # The verified child process group is gone. Child budget validation is
        # still mandatory before settlement, including any grandchild leases.


def _serialized(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return guarded


class NestedRun:
    """Reattach identity is distinct from recovering an ephemeral model gateway.

    A parent-process crash cannot transparently rebind a live child's model
    routes. Such calls fail or reach the finite child deadline; this host keeps
    the original job identity and never resubmits it to hide that interruption.
    """
    backend = 'nested-run'

    def __init__(self, context, *, root=None, role='nested'):
        self.context = context
        self._lock = threading.RLock()
        self.root = Path(root or (Path(context.data_dir) / 'nested_runs')).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.resources, self.executor, self.budget = context.resources, context.executor, context.budget
        from .resources import ResourceManager
        if isinstance(self.resources, ResourceManager):
            self.resources = self.resources.component(role)
        if any(value is None for value in (self.resources, self.executor, self.budget)):
            raise TypeError('NestedRun requires host resources, executor and budget')
        self.role = role
        self.resources.declare({role: ResourceRequest(gpus=0)})
        self._jobs, self._handles, self._routes = {}, {}, {}
        self._closed = False
        for path in self.root.glob('*/descriptor.json'):
            saved = json.loads(path.read_text())
            self._validate_descriptor(saved, path.parent.name)
            self._jobs[path.parent.name] = saved

    def _validate_descriptor(self, value, ident):
        if (not isinstance(value, dict) or set(value) != {'schema', 'job_id', 'request_digest', 'phase', 'worker', 'outcome', 'gateway_owner'}
                or type(value['schema']) is not int or value['schema'] != 2 or value['job_id'] != ident
                or not isinstance(ident, str) or len(ident) != 32
                or any(c not in '0123456789abcdef' for c in ident)
                or not isinstance(value['request_digest'], str) or len(value['request_digest']) != 64
                or any(c not in '0123456789abcdef' for c in value['request_digest'])
                or value['phase'] not in {'prepared', 'launching', 'uncertain', 'running', 'terminal'}):
            raise ValueError('Invalid nested run descriptor')
        if value['phase'] == 'terminal':
            outcome = value['outcome']
            if not isinstance(outcome, dict) or set(outcome) != {'status', 'error', 'result'}:
                raise ValueError('Invalid terminal nested outcome')
            parsed = JobOutcome(JobRef(ident, self.backend), JobStatus(outcome['status']),
                                result=outcome['result'], error=outcome['error'])
            if not parsed.terminal:
                raise ValueError('A terminal nested descriptor needs a terminal outcome')
        elif value['outcome'] is not None:
            raise ValueError('An active nested run cannot carry a terminal outcome')
        owner = value['gateway_owner']
        if owner is not None and (not isinstance(owner, dict)
                or set(owner) != {'pid', 'identity', 'hostname'}
                or type(owner['pid']) is not int or owner['pid'] < 1
                or not isinstance(owner['identity'], str) or not owner['identity']
                or not isinstance(owner['hostname'], str) or not owner['hostname']):
            raise ValueError('Invalid nested model gateway owner identity')
        worker = value['worker']
        if worker is not None:
            if not isinstance(worker, dict) or worker.get('backend') not in {'local', 'ray'}:
                raise ValueError('Invalid nested worker reference')
            if worker['backend'] == 'local':
                if (set(worker) != {'backend', 'directory', 'pid', 'identity'}
                        or type(worker['pid']) is not int or worker['pid'] < 0
                        or not isinstance(worker['directory'], str) or not Path(worker['directory']).is_absolute()
                        or (worker['identity'] is None) != (worker['pid'] == 0)):
                    raise ValueError('Invalid local nested worker identity')
            elif set(worker) != {'backend', 'reference'} or not isinstance(worker['reference'], dict):
                raise ValueError('Invalid Ray nested worker identity')
        elif value['phase'] == 'running':
            raise ValueError('A running nested descriptor requires its worker reference')

    def _save(self, job):
        atomic_write(self.root / job['job_id'] / 'descriptor.json', json_bytes(job))

    @_serialized
    def start(self, controller, labs, *, objective, budget, call_id):
        if self._closed:
            raise RuntimeError('NestedRun admission is closed')
        if not isinstance(call_id, str) or not call_id.strip() or not isinstance(objective, str) or not objective.strip():
            raise ValueError('NestedRun requires a call identity and objective')
        # Every child has a finite orphan deadline, even if its parent is unbounded.
        if not isinstance(budget, dict) or 'seconds' not in budget:
            raise ValueError('NestedRun budget must include a finite seconds limit')
        manifest = self.context.nested_manifest(controller, labs)
        request = {'schema': 1, 'manifest': manifest, 'objective': objective, 'budget': budget,
                   'parent_call_id': call_id}
        digest = sha256(json_bytes(request)).hexdigest()
        ident = sha256(call_id.encode()).hexdigest()[:32]
        if ident in self._jobs:
            if self._jobs[ident]['request_digest'] != digest:
                raise ValueError('Nested call identity was reused with different inputs')
            if self._jobs[ident]['phase'] != 'prepared':
                return JobRef(ident, self.backend)
        directory = self.root / ident
        directory.mkdir(exist_ok=True)
        path = directory / 'request.json'
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError('Nested request identity changed')
        atomic_write(path, json_bytes(request))
        job = {'schema': 2, 'job_id': ident, 'request_digest': digest,
               'phase': 'prepared', 'worker': None, 'outcome': None, 'gateway_owner': None}
        self._jobs[ident] = job
        self._save(job)
        lease = None
        child_budget = None
        cleanup = None
        try:
            lease = self.resources.acquire(self.role)
            if lease.devices:
                raise ValueError('Nested supervisor placement must request zero GPUs; child workers own GPU leases')
            child_budget = self.budget.reserve_child(ident, directory / 'budget', budget)
            delegated, ephemeral, cleanup = self.context.prepare_nested_models(
                child_budget, call_id, self.context.host_config, controller, labs)
            if any(delegated.get(group) for group in ('backends', 'routes')):
                job['gateway_owner'] = {'pid': os.getpid(), 'identity': process_identity(os.getpid()),
                                        'hostname': socket.gethostname()}
                if job['gateway_owner']['identity'] is None:
                    raise RuntimeError('Cannot establish nested model gateway process identity')
            host_config = deepcopy(self.context.host_config)
            if (self.resources.owner.config.get('executor') == 'ray'
                    and self.resources.owner.config.get('host', {}).get('mode', 'head') == 'head'):
                host_config['application']['resources'].setdefault('host', {})['address'] = self.resources.owner.cluster_address
            bootstrap = {'schema': 1, 'manifest': manifest, 'host': host_config,
                'objective': objective, 'parent_call_id': call_id,
                'budget_root': str(child_budget.root), 'budget': child_budget.limits,
                'child_dir': str(directory / 'child'), 'run_id': ident,
                'delegated_models': delegated, 'output': str(directory / 'worker_result.json'),
                'coordinator_visibility': {key: (os.environ.get(key) if
                    self.resources.owner.config.get('executor', 'local') == 'local' else None)
                    for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES')}}
            atomic_write(directory / 'bootstrap.json', json_bytes(bootstrap))
            # Only explicitly delegated tokens and a small runtime allowlist are
            # inherited. Provider keys in the parent's environment never cross.
            environment = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'LC_ALL',
                'TMPDIR', 'SYSTEMROOT', 'SSL_CERT_FILE', 'SSL_CERT_DIR') if key in os.environ}
            import rsicomposer
            environment['PYTHONPATH'] = str(Path(rsicomposer.__file__).resolve().parent.parent)
            if os.environ.get('PYTHONPATH'):
                environment['PYTHONPATH'] += os.pathsep + os.environ['PYTHONPATH']
            environment.update(ephemeral)
            command = (sys.executable, '-m', 'rsicomposer.nested_worker', str(directory / 'bootstrap.json'))
            job['phase'] = 'launching'
            self._save(job)  # Never blindly replay after this side-effect boundary.
            with (directory / 'worker.log').open('a') as log:
                handle = self.executor.launch(WorkerSpec(command, environment=environment,
                    stdout=log, timeout=child_budget.timeout(), detached=True), lease)
            self._handles[ident], self._routes[ident] = handle, cleanup
            cleanup = None
            reference = getattr(handle, 'reference', None)
            job['worker'] = ({'backend': 'ray', 'reference': reference()} if callable(reference) else
                {'backend': 'local', 'directory': str(handle.directory), 'pid': handle.pid,
                 'identity': process_identity(handle.pid)})
            if job['worker']['backend'] == 'local' and job['worker']['identity'] is None:
                raise RuntimeError('Nested supervisor identity could not be established')
            if job['worker']['backend'] == 'local':
                self._handles[ident] = _LocalAttachment(job['worker'])
            job['phase'] = 'running'
            self._save(job)
            return JobRef(ident, self.backend)
        except BaseException as error:
            submitted = (ident in self._handles or getattr(lease, '_launched_count', 0) > 0
                         or isinstance(error, WorkerLaunchUncertainError))
            if not submitted:
                if child_budget is not None:
                    self.budget.settle_child(ident, unstarted=True)
                job['phase'] = 'terminal'
                job['outcome'] = {'status': 'failed', 'error': 'Nested run admission failed: ' + str(error), 'result': None}
                self._save(job)
                if cleanup is not None:
                    self._routes[ident] = cleanup
                    try:
                        cleanup()
                    except BaseException as cleanup_error:
                        error.add_note('Credential cleanup remains pending for poll/close retry: ' + str(cleanup_error))
                    else:
                        del self._routes[ident]
            else:
                # Launch may already have succeeded; retain token and quota
                # until confirmed cancellation, never reissue the command.
                if cleanup is not None:
                    self._routes[ident] = cleanup
                job['phase'] = 'uncertain'
                self._save(job)
            raise
        finally:
            if lease is not None:
                lease.release()

    def _job(self, ref):
        if not isinstance(ref, JobRef) or ref.backend != self.backend or ref.job_id not in self._jobs:
            raise ValueError('Unknown nested run reference')
        saved = json.loads((self.root / ref.job_id / 'descriptor.json').read_text())
        self._validate_descriptor(saved, ref.job_id)
        if saved['request_digest'] != self._jobs[ref.job_id]['request_digest']:
            raise ValueError('Nested durable request identity changed')
        self._jobs[ref.job_id] = saved
        return saved

    def _handle(self, job):
        ident = job['job_id']
        if ident in self._handles:
            return self._handles[ident]
        worker = job['worker']
        if worker is None:
            worker = self._discover_worker(job)
            if worker is None:
                raise RuntimeError('Nested launch outcome is uncertain; automatic replay is forbidden')
            job['worker'], job['phase'] = worker, 'running'
            self._save(job)
        handle = (self.executor.reattach(worker['reference']) if worker['backend'] == 'ray'
                  else _LocalAttachment(worker))
        self._handles[ident] = handle
        return handle

    def _discover_worker(self, job):
        """Close the submit/publication crash window without submitting again."""
        directory = self.root / job['job_id']
        command = (sys.executable, '-m', 'rsicomposer.nested_worker', str(directory / 'bootstrap.json'))
        find = getattr(self.executor, 'find_job', None)
        if callable(find):
            reference = find(command)
            return {'backend': 'ray', 'reference': reference} if reference is not None else None
        matches = []
        for request in self.executor.root.glob('*/request.json'):
            if json.loads(request.read_text()).get('command') == list(command):
                matches.append(request.parent)
        if len(matches) > 1:
            raise RuntimeError('Multiple workers claim one nested call identity')
        if not matches:
            return None
        supervisor = matches[0]
        if (supervisor / 'result.json').exists():
            return {'backend': 'local', 'directory': str(supervisor), 'pid': 0, 'identity': None}
        publication = directory / 'worker_started.json'
        if not publication.exists() or not (supervisor / 'child.json').exists():
            return None
        started = json.loads(publication.read_text())
        child = json.loads((supervisor / 'child.json').read_text())
        if child != {'pid': started['pid'], 'identity': started['identity']}:
            raise RuntimeError('Nested worker identity publication conflicts with its supervisor')
        if (started['supervisor_identity'] is None or
                process_identity(started['supervisor_pid']) != started['supervisor_identity']):
            raise RuntimeError('Nested supervisor vanished during admission; cleanup remains uncertain')
        return {'backend': 'local', 'directory': str(supervisor), 'pid': started['supervisor_pid'],
                'identity': started['supervisor_identity']}

    def _import_evidence(self, job, value):
        exports, imported, visiting = value['artifacts'], {}, set()
        def project(item):
            if isinstance(item, dict) and set(item) == {'$nested_artifact'}:
                ident = item['$nested_artifact']
                if ident not in exports or ident in visiting:
                    raise ValueError('Invalid child evidence artifact namespace')
                if ident not in imported:
                    visiting.add(ident)
                    child = exports[ident]
                    payload = {'child_run_id': job['job_id'], 'source': {
                        'child_artifact_id': ident, 'kind': child['kind'], 'digest': child['digest']},
                        'content': project(child['content'])}
                    imported[ident] = self.context.artifacts.put(json_bytes(payload),
                        kind='nested-run-evidence-artifact')
                    visiting.remove(ident)
                return asdict(imported[ident])
            if isinstance(item, dict):
                return {key: project(part) for key, part in item.items()}
            if isinstance(item, list):
                return [project(part) for part in item]
            return item
        return project(value['evidence'])

    def _finish(self, job, *, cancelled=False, failure=None):
        ident = job['job_id']
        cleanup = self._routes.get(ident)
        if cleanup is not None:
            cleanup()
            del self._routes[ident]
        # This rejects live grandchildren even after the immediate worker exits.
        usage = self.budget.settle_child(ident, closed_tree=True)
        output = self.root / ident / 'worker_result.json'
        value = json.loads(output.read_text()) if output.exists() else None
        if failure is not None:
            outcome = {'status': 'failed', 'error': failure, 'result': None}
        elif cancelled:
            outcome = {'status': 'cancelled', 'error': None, 'result': None}
        elif value is not None and value.get('schema') == 1 and value.get('status') == 'succeeded':
            evidence = self.context.artifacts.put(json_bytes({
                'parent_call_id': value['parent_call_id'], 'child_run_id': ident,
                'events': self._import_evidence(job, value)}), kind='nested-run-evidence')
            outcome = {'status': 'succeeded', 'error': None, 'result': {
                'child_run_id': ident, 'parent_call_id': value['parent_call_id'],
                'evidence_ref': asdict(evidence), 'validation': value['validation'], 'budget_used': usage}}
        else:
            outcome = {'status': 'failed', 'error': 'Nested child failed; see its owned worker log', 'result': None}
        job['phase'], job['outcome'] = 'terminal', outcome
        self._save(job)

    @_serialized
    def reconcile(self, ref):
        """Discover the original submitted worker without issuing another launch.

        Unknown submission keeps its quota and returns UNKNOWN until the
        executor publishes an identity or a completed cleanup record.
        """
        job = self._job(ref)
        if job['phase'] in {'launching', 'uncertain'} and job['worker'] is None:
            worker = self._discover_worker(job)
            if worker is None:
                return JobOutcome(ref, JobStatus.UNKNOWN,
                    error='Nested launch outcome is uncertain; reconcile after executor publication')
            job['worker'], job['phase'] = worker, 'running'
            self._save(job)
        return self.poll(ref)

    @_serialized
    def poll(self, ref):
        job = self._job(ref)
        if job['phase'] in {'launching', 'uncertain'} and job['worker'] is None:
            worker = self._discover_worker(job)
            if worker is None:
                return JobOutcome(ref, JobStatus.UNKNOWN,
                    error='Nested launch outcome is uncertain; use reconcile after executor publication')
            job['worker'], job['phase'] = worker, 'running'
            self._save(job)
        if job['phase'] != 'terminal':
            handle = self._handle(job)
            owner = job['gateway_owner']
            if owner is not None and (owner['hostname'] != socket.gethostname()
                    or process_identity(owner['pid']) != owner['identity']):
                handle.terminate()
                self._finish(job, failure='parent_model_gateway_lost: original child cancelled; job was not replayed')
                return self.poll(ref)
            if handle.poll() is None:
                return JobOutcome(ref, JobStatus.RUNNING)
            self._finish(job)
        cleanup = self._routes.get(ref.job_id)
        if cleanup is not None:
            cleanup()
            del self._routes[ref.job_id]
        value = job['outcome']
        return JobOutcome(ref, JobStatus(value['status']), result=value['result'], error=value['error'])

    def job_result(self, ref):
        outcome = self.poll(ref)
        if not outcome.terminal:
            raise RuntimeError('Nested run has not finished')
        return outcome

    @_serialized
    def cancel(self, ref):
        job = self._job(ref)
        if job['phase'] in {'launching', 'uncertain'} and job['worker'] is None:
            reconciled = self.reconcile(ref)
            if reconciled.status == JobStatus.UNKNOWN or reconciled.terminal:
                return reconciled
            job = self._job(ref)
        if job['phase'] != 'terminal':
            self._handle(job).terminate()
            self._finish(job, cancelled=True)
        return self.poll(ref)

    @_serialized
    def shutdown(self, mode='cancel'):
        if mode not in {'drain', 'cancel'}:
            raise ValueError('Nested shutdown mode must be drain or cancel')
        self._closed = True
        for ident in tuple(self._jobs):
            ref = JobRef(ident, self.backend)
            self.cancel(ref) if mode == 'cancel' else self.poll(ref)
        return all(job['phase'] == 'terminal' for job in self._jobs.values())

    def close(self):
        self.shutdown('cancel')

    @_serialized
    def state(self):
        return {'schema': 2, 'jobs': json.loads(json.dumps(self._jobs)), 'closed': self._closed}

    @_serialized
    def restore(self, state):
        if not isinstance(state, dict) or set(state) != {'schema', 'jobs', 'closed'} or state['schema'] != 2:
            raise ValueError('Unsupported nested run state')
        if not isinstance(state['jobs'], dict) or type(state['closed']) is not bool:
            raise ValueError('Invalid nested run state')
        for ident, job in state['jobs'].items():
            self._validate_descriptor(job, ident)
            previous = self._jobs.get(ident)
            # Durable descriptors may be ahead of the parent's last checkpoint.
            if previous is not None and previous['request_digest'] != job['request_digest']:
                raise ValueError('Nested durable identity differs from restored checkpoint')
            if previous is None:
                if job['phase'] != 'terminal':
                    raise RuntimeError('Portable restore cannot relocate active nested workers')
                (self.root / ident).mkdir(exist_ok=True)
                self._jobs[ident] = job
                self._save(job)
        self._closed = state['closed']

    @_serialized
    def validate_snapshot(self):
        for ident in self._jobs:
            self.poll(JobRef(ident, self.backend))
        if any(job['phase'] != 'terminal' for job in self._jobs.values()):
            raise RuntimeError('Snapshot requires every nested run to finish or cancel')
