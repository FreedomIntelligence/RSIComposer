"""Explicit same-directory persistence for the bundled trusted local executor.

Retain workspaces and terminal process receipts. Never adopt a PID as a Popen,
signal a recovered PID, or replay a submission with an unknown outcome. An
unclean shutdown without an owner's terminal receipt requires reconciliation.
This adapter deliberately does not advertise portable snapshot support.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading

from rsisys.loops.contracts.jobs import JobStatus
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.recovery import RecoveryRequired
from rsisys.runtime.sandbox import SandboxRef, ExecRequest, ProcessRef, ProcessState


def _write(path, payload):
    atomic_write(path, json_bytes({'payload': payload,
        'sha256': hashlib.sha256(json_bytes(payload)).hexdigest()}))


def _read(path):
    raw = json.loads(path.read_text())
    if (set(raw) != {'payload', 'sha256'}
            or hashlib.sha256(json_bytes(raw['payload'])).hexdigest() != raw['sha256']):
        raise ValueError('Local process recovery digest mismatch')
    return raw['payload']


def process_identity(pid):
    """Observational ownership evidence; never used to signal a recovered PID."""
    proc = Path('/proc') / str(pid)
    if proc.exists():
        stat = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'uid': proc.stat().st_uid, 'start_ticks': stat[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    created = subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart='], text=True, timeout=5).strip()
    if not created:
        raise RuntimeError('Cannot record owned process creation identity')
    return {'pid': pid, 'uid': os.getuid(), 'created': created, 'host': socket.gethostname()}


class RetainedLocalProcessBackend(LocalProcessBackend):
    """Keep native supervision; add durable identities and terminal log receipts.

    Graceful pause cancels owned jobs before checkpointing. After a hard crash,
    only receipts written by the original supervisor can be reattached. Missing
    receipts and extra submissions are uncertain, even if a PID has disappeared.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persistence_lock = threading.RLock()
        self._registry_path = self.root / 'process-recovery.json'
        self._receipt_root = self.root / 'process-receipts'
        self._receipt_root.mkdir(exist_ok=True)
        self._refs, self._restored = {}, {}
        self._registry = (_read(self._registry_path) if self._registry_path.exists() else
            {'schema': 1, 'uid': os.getuid(), 'root': str(self.root), 'sessions': {}, 'processes': {}})
        if (self._registry.get('schema') != 1 or self._registry.get('uid') != os.getuid()
                or self._registry.get('root') != str(self.root)):
            raise ValueError('Local process recovery owner/workspace mismatch')

    def create(self, spec):
        ref = super().create(spec)
        with self._persistence_lock:
            self._registry['sessions'][ref.sandbox_id] = {
                'path': str(self.workspace(ref)), 'timeout_seconds': spec.timeout_seconds}
            _write(self._registry_path, self._registry)
        return ref

    def _process_started(self, ref, request, entry):
        with self._persistence_lock:
            self._refs[id(entry)] = ref
            # Record evidence for an operator; restore never sends a PID signal.
            self._registry['processes'][ref.process_id] = {
                'sandbox_id': ref.sandbox.sandbox_id,
                'request': {'argv': list(request.argv), 'timeout_seconds': request.timeout_seconds},
                **process_identity(entry.process.pid)}
            _write(self._registry_path, self._registry)

    def _finish(self, entry):
        super()._finish(entry)
        with self._persistence_lock:
            ref = self._refs[id(entry)]
            _write(self._receipt_root / (ref.process_id + '.json'), {
                'identity': self._registry['processes'][ref.process_id],
                'state': {**asdict(entry.state), 'status': entry.state.status.value},
                'logs': self.log_info(ref)})

    def capture_state(self):
        with self._persistence_lock:
            return deepcopy(self._registry)

    def restore_state(self, state):
        if self._sessions or self._processes or self._restored:
            raise RuntimeError('Restore requires an unattached local executor')
        if state != self._registry:
            raise RecoveryRequired('Local executor identities changed after the checkpoint; no automatic replay')
        sessions, processes = {}, {}
        for ident, row in state['sessions'].items():
            path = Path(row['path'])
            if (path.is_symlink() or not path.is_dir() or path.resolve().parent != self.root):
                raise ValueError('Retained Lab workspace is missing or changed')
            sessions[SandboxRef(ident)] = (path, row['timeout_seconds'])
        for ident, row in state['processes'].items():
            path = self._receipt_root / (ident + '.json')
            if not path.is_file():
                raise RecoveryRequired(f'Owned process {ident} (recorded PID {row["pid"]}) has no '
                    'terminal receipt; verify its original ownership and reconcile before resuming')
            receipt = _read(path)
            if receipt['identity'] != row:
                raise ValueError('Owned process receipt identity mismatch')
            status = ProcessState(**{**receipt['state'], 'status': JobStatus(receipt['state']['status'])})
            if status.status not in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                raise RecoveryRequired('A recovered process is not confirmed terminal')
            for log in receipt['logs'].values():
                for key, digest in (('path', 'sha256'), ('tail_path', 'tail_sha256')):
                    file = Path(log[key])
                    if (file.is_symlink() or not file.is_file()
                            or not file.resolve().is_relative_to(self.log_root)
                            or hashlib.sha256(file.read_bytes()).hexdigest() != log[digest]):
                        raise ValueError('Retained process log identity/digest mismatch')
            ref = ProcessRef(ident, SandboxRef(row['sandbox_id']))
            if ref.sandbox not in sessions:
                raise ValueError('Recovered process has no matching session')
            processes[ref] = (status, receipt['logs'])
        # Attach only after the entire inventory has passed. No process starts.
        self._sessions, self._restored = sessions, processes

    def poll(self, ref):
        return self._restored[ref][0] if ref in self._restored else super().poll(ref)

    def wait(self, ref):
        return self.poll(ref) if ref in self._restored else super().wait(ref)

    def cancel(self, ref):
        if ref not in self._restored:
            super().cancel(ref)

    def log_info(self, ref):
        return deepcopy(self._restored[ref][1]) if ref in self._restored else super().log_info(ref)

    def read_log(self, ref, stream, *, offset=None, max_bytes=8192):
        if ref not in self._restored:
            return super().read_log(ref, stream, offset=offset, max_bytes=max_bytes)
        if (stream not in ('stdout', 'stderr') or type(max_bytes) is not int or max_bytes < 1
                or (offset is not None and (type(offset) is not int or offset < 0))):
            raise ValueError('Invalid log stream or byte window')
        log = self.log_info(ref)[stream]
        total, stored, tail = log['observed_bytes'], log['stored_bytes'], log['tail_offset']
        offset = max(0, total - max_bytes) if offset is None else offset
        gap = None
        if offset < stored:
            with Path(log['path']).open('rb') as saved:
                saved.seek(offset)
                raw = saved.read(min(max_bytes, stored - offset))
        elif offset >= tail:
            with Path(log['tail_path']).open('rb') as saved:
                saved.seek(offset - tail)
                raw = saved.read(max_bytes)
        else:
            raw = b''
            gap = {'start': stored, 'end': tail, 'next_retained_offset': tail}
        return {'stream': stream, 'offset': offset, 'next_offset': tail if gap else offset + len(raw),
            'content': raw.decode('utf-8', errors='replace'), 'bytes': len(raw),
            'observed_bytes': total, 'gap': gap, 'eof': offset + len(raw) >= total,
            'complete': True, 'reference': log}

    def close(self, sandbox):
        if sandbox not in self._sessions:
            return
        for ref, entry in tuple(self._processes.items()):
            if ref.sandbox == sandbox:
                if not entry.done.is_set():
                    self.cancel(ref)
                    if not entry.done.wait(10):
                        raise TimeoutError('Owned process cleanup incomplete; workspace retained')
                self._finish(entry)
        # Recovery owns deletion/retention policy, not session close.
        del self._sessions[sandbox]


class LocalLabState:
    """One explicit state port for the bundled LocalProcessLab and its backend."""
    state_id = 'rsisys-retained-local-lab/v1'

    def __init__(self, lab, *, capture_extra=lambda: None, restore_extra=lambda state: None,
                 configuration=None):
        self.lab = lab
        self.capture_extra, self.restore_extra = capture_extra, restore_extra
        self.extra_configuration = configuration

    def configuration(self):
        lab = self.lab
        result = {'timeout_seconds': lab.timeout_seconds, 'max_jobs': lab.max_jobs,
            'max_parallel': lab.max_parallel, 'max_file_bytes': lab.max_file_bytes,
            'max_poll_wait_seconds': lab.max_poll_wait_seconds,
            'max_output_bytes': lab.backend.max_output_bytes, 'max_log_bytes': lab.backend.max_log_bytes,
            'extension': self.extra_configuration}
        isolation = getattr(lab.backend, 'isolation', None)
        if isolation is not None:
            result['execution_policy'] = isolation.policy.configuration()
        return result

    def capture(self):
        lab = self.lab
        return {'backend': lab.backend.capture_state(), 'session': lab.session.sandbox_id,
            'jobs': {ident: {'sandbox': ref.sandbox.sandbox_id, 'request': asdict(request)}
                     for ident, (ref, request) in lab._jobs.items()},
            'keys': {key: ident for key, (ident, _) in lab._keys.items()},
            'terminal': deepcopy(lab._terminal), 'accepting': lab._accepting,
            'extra': deepcopy(self.capture_extra())}

    def restore(self, state):
        lab = self.lab
        jobs = {ident: (ProcessRef(ident, SandboxRef(row['sandbox'])), ExecRequest(**row['request']))
                for ident, row in state['jobs'].items()}
        if (set(jobs) != set(state['backend']['processes']) or not set(state['terminal']) <= jobs.keys()
                or any(ident not in jobs for ident in state['keys'].values())):
            raise ValueError('Lab job/key inventory differs from executor identities')
        lab.backend.restore_state(state['backend'])
        session = SandboxRef(state['session'])
        lab.backend.workspace(session)
        for ident, (ref, _) in jobs.items():
            if ref.sandbox != session:
                raise ValueError('Lab job belongs to another session')
        for ident, result in state['terminal'].items():
            process = lab.backend.poll(jobs[ident][0])
            if (result.data.get('job_id') != ident or result.data.get('argv') != list(jobs[ident][1].argv)
                    or any(result.data.get(key) != getattr(process, key)
                           for key in ('stdout', 'stderr', 'exit_code'))):
                raise ValueError('Cached Lab result differs from its process receipt')
            for ref in result.artifacts:
                lab.artifacts.get(ref)
        lab.session, lab._jobs = session, jobs
        lab._keys = {key: (ident, jobs[ident][1]) for key, ident in state['keys'].items()}
        lab._terminal, lab._accepting = deepcopy(state['terminal']), state['accepting']
        self.restore_extra(deepcopy(state['extra']))
