"""Programmable local Lab using the existing owned-process backend.

This is trusted local execution, not OS isolation. Programs may edit their own
training/evaluation implementations; device/data/network isolation belongs to
the deployed executor. No framework SFT/RL/evaluation workflow is injected.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from uuid import uuid4

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.delivery import FeedbackDelivery
from rsisys.loops.contracts.execution import LabInfrastructureError, LabToolError
from rsisys.loops.contracts.jobs import JobStatus, TERMINAL_STATUSES
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.run_policy import positive
from rsisys.runtime.sandbox import SandboxSpec, ExecRequest


class LocalProcessLab:
    """Persistent code workspace plus start/poll/cancel and artifact export.

    Background submissions return invocation receipts, not blocking JobRefs.
    Host-only limits and shutdown cannot be changed by a tool argument. The Lab
    closes its own session, never other sessions/services sharing the backend.
    """
    def __init__(self, backend: LocalProcessBackend, artifacts, *, timeout_seconds=300,
                 max_jobs=1024, max_parallel=4, max_file_bytes=1024 * 1024):
        if not isinstance(backend, LocalProcessBackend):
            raise TypeError('LocalProcessLab requires an explicit LocalProcessBackend')
        positive(timeout_seconds, 'timeout_seconds')
        for name, value in (('max_jobs', max_jobs), ('max_parallel', max_parallel), ('max_file_bytes', max_file_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
            raise NotImplementedError('LocalProcessLab requires descriptor-relative no-follow opens')
        self.backend, self.artifacts = backend, artifacts
        self.timeout_seconds, self.max_jobs = timeout_seconds, max_jobs
        self.max_parallel, self.max_file_bytes = max_parallel, max_file_bytes
        self._jobs, self._keys, self._terminal = {}, {}, {}
        self._accepting, self._closed = True, False
        self.session = backend.create(SandboxSpec('local', timeout_seconds))

    @property
    def workspace(self):
        return self.backend.workspace(self.session)

    def describe(self):
        text = {'type': 'string', 'minLength': 1}
        path = {**text, 'description': 'Relative workspace file path; no absolute path, .. or symlink traversal.'}
        tools = (
            ('lab_write_file', 'Write or replace a UTF-8 code/config file. You choose the implementation, including SFT, RL and evaluation.',
             {'path': path, 'content': {'type': 'string'}}, ['path', 'content'], False),
            ('lab_read_file', 'Read a bounded UTF-8 file window; offset and next_offset count bytes.',
             {'path': path, 'offset': {'type': 'integer', 'minimum': 0},
              'max_bytes': {'type': 'integer', 'minimum': 1, 'maximum': self.max_file_bytes}}, ['path'], True),
            ('lab_export', 'Preserve a workspace file in the host artifact store before workspace cleanup.',
             {'path': path}, ['path'], True),
            ('lab_start', 'Start any installed program in the persistent workspace. A key identifies one exact submission. Submission is not completion; inspect lab_poll, then deliver terminal evidence with lab_result.',
             {'key': text, 'argv': {'type': 'array', 'items': text, 'minItems': 1},
              'timeout_seconds': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': self.timeout_seconds}},
             ['key', 'argv'], False),
            ('lab_poll', 'Observe a known owned job without delivering a research result. When terminal, use lab_result for feedback/memory. Native stdout/stderr are not automatically validated metrics or adoption.',
             {'job_id': text}, ['job_id'], True),
            ('lab_status', 'Read current owned jobs and recent terminal jobs, independently of remembered history.',
             {}, [], True),
            ('lab_read_log', 'Read durable stdout/stderr by byte offset. Omit offset for the latest tail. Returns exact retained coverage and any quota gap.',
             {'job_id': text, 'stream': {'enum': ['stdout', 'stderr']},
              'offset': {'type': 'integer', 'minimum': 0},
              'max_bytes': {'type': 'integer', 'minimum': 1, 'maximum': self.max_file_bytes}},
             ['job_id', 'stream'], True),
            ('lab_result', 'Collect terminal evidence for feedback. Repeated collection is archived but organizes memory once per result event. Set reprocess=true to explicitly organize the same result again, without rerunning the program. Inspect lab_poll first; collection does not validate evaluation or adoption.',
             {'job_id': text, 'reprocess': {'type': 'boolean'}}, ['job_id'], False),
            ('lab_cancel', 'Request cancellation of a known owned job. Poll to confirm exit; acknowledgement is not completion.',
             {'job_id': text}, ['job_id'], False),
        )
        return tuple(OperationSpec(name, description, {
            'type': 'object', 'properties': deepcopy(properties), 'required': required,
            'additionalProperties': False,
        }, read_only) for name, description, properties, required, read_only in tools)

    def execute(self, call: OperationCall):
        if self._closed:
            raise RuntimeError('Lab is closed')
        if not isinstance(call, OperationCall) or not isinstance(call.arguments, dict):
            raise TypeError('Expected OperationCall with an argument object')
        spec = next((item for item in self.describe() if item.name == call.name), None)
        if spec is None:
            raise ValueError('Unknown Lab tool')
        args = deepcopy(call.arguments)
        if set(args) - spec.input_schema['properties'].keys() or set(spec.input_schema['required']) - args.keys():
            raise LabToolError('Missing or unknown Lab arguments')
        json.dumps(args, allow_nan=False)
        if not self._accepting and call.name in ('lab_start', 'lab_write_file'):
            raise RuntimeError('Lab is shutting down; new work is closed')
        if call.name == 'lab_start':
            return self._start(args)
        if call.name == 'lab_status':
            return OperationResult('Current owned Lab state', (), False, data=self._status())
        if call.name == 'lab_read_log':
            ident = args['job_id']
            if not isinstance(ident, str) or ident not in self._jobs:
                raise LabToolError('Unknown owned Lab job')
            maximum = args.get('max_bytes', min(8192, self.max_file_bytes))
            offset = args.get('offset')
            if (args['stream'] not in ('stdout', 'stderr') or type(maximum) is not int
                    or not 1 <= maximum <= self.max_file_bytes
                    or (offset is not None and (type(offset) is not int or offset < 0))):
                raise LabToolError('Invalid log stream or byte window')
            data = self.backend.read_log(self._jobs[ident][0], args['stream'], offset=offset, max_bytes=maximum)
            return OperationResult('Owned process log window', (), False, data={'job_id': ident, **data})
        if call.name in ('lab_poll', 'lab_result', 'lab_cancel'):
            ident = args['job_id']
            if not isinstance(ident, str) or ident not in self._jobs:
                raise LabToolError('Unknown owned Lab job')
            if call.name == 'lab_result' and type(args.get('reprocess', False)) is not bool:
                raise LabToolError('reprocess must be a bool')
            if call.name == 'lab_cancel':
                self.backend.cancel(self._jobs[ident][0])
                return OperationResult('Cancellation requested; poll to confirm exit', (), False,
                                       data={'job_id': ident, 'cancellation_requested': True})
            result = self._observe(ident)
            if call.name == 'lab_result':
                if result.data['state'] not in {status.value for status in TERMINAL_STATUSES}:
                    return OperationResult('Program result is not ready; poll before collection', (), False,
                                           data=result.data)
                result = replace(result, feedback_delivery=FeedbackDelivery(
                    f'local:{self.session.sandbox_id}:{ident}:terminal', args.get('reprocess', False)))
            if result.data.get('failure_origin') == 'infrastructure':
                # Preserve the collected event identity and archived evidence
                # before escalating. Internal teardown checks exit separately.
                raise LabInfrastructureError(result)
            return result
        return self._file(call.name, args)

    def _start(self, args):
        key, argv = args['key'], args['argv']
        if not isinstance(key, str) or not key.strip():
            raise LabToolError('A nonempty submission key is required')
        if not isinstance(argv, list):
            raise LabToolError('argv must be a list')
        timeout = args.get('timeout_seconds', self.timeout_seconds)
        try:
            request = ExecRequest(tuple(argv), timeout)
        except ValueError as exc:
            raise LabToolError(str(exc)) from exc
        if timeout > self.timeout_seconds:
            raise ValueError('Requested timeout exceeds the host limit')
        if key in self._keys:
            ident, previous = self._keys[key]
            if previous != request:
                raise LabToolError('Submission key was already used for different arguments; '
                                   'use a new key for changed arguments. The original submission is unchanged')
            state = self.backend.poll(self._jobs[ident][0])
            return OperationResult('Existing submission; no program was started again', (), False,
                data={'job_id': ident, 'state': state.status.value, 'argv': list(request.argv),
                      'replayed': True})
        else:
            if len(self._jobs) >= self.max_jobs:
                raise PermissionError('Host job admission limit reached')
            active = self._status()['active_jobs']
            if len(active) >= self.max_parallel:
                # A busy slot is temporary backpressure, not a permission
                # denial. No process or key is reserved until admission succeeds.
                raise LabToolError('Host parallel execution slots are busy; use lab_poll or lab_cancel '
                                   'on an existing job, then retry after a slot is released',
                                   data={'reason': 'execution_busy', 'active_jobs': active})
            try:
                ref = self.backend.exec(self.session, request)
            except OSError as exc:
                # A missing/invalid program is repairable by decision. Missing
                # workspaces, permissions and backend failures remain host errors.
                if exc.errno in (errno.ENOENT, errno.ENOEXEC) and exc.filename == request.argv[0]:
                    raise LabToolError(f'{type(exc).__name__}: {exc}') from exc
                raise
            ident = ref.process_id
            self._jobs[ident] = (ref, request)
            self._keys[key] = (ident, request)
        # The action completed submission, not the program/experiment. Always
        # report the submission separately, even when the process exited fast.
        return OperationResult('Command submitted; program completion is not confirmed', (), False,
                               data={'job_id': ident, 'state': 'submitted', 'argv': list(request.argv),
                                     'replayed': False})

    def _status(self):
        active, terminal = [], []
        for ident, (ref, request) in self._jobs.items():
            state = self.backend.poll(ref)
            row = {'job_id': ident, 'state': state.status.value,
                   'poll': {'operation': 'lab_poll', 'job_id': ident},
                   'logs': {'operation': 'lab_read_log', 'job_id': ident}}
            (terminal if state.status in TERMINAL_STATUSES else active).append(row)
        return {'active_jobs': active, 'recent_terminal_jobs': terminal[-8:],
                'terminal_jobs_total': len(terminal), 'accepting': self._accepting}

    def runtime_context(self):
        return json.dumps(self._status(), ensure_ascii=False, separators=(',', ':'))

    def _observe(self, ident):
        if ident in self._terminal:
            return deepcopy(self._terminal[ident])
        ref, request = self._jobs[ident]
        state = self.backend.poll(ref)
        data = {'job_id': ident, 'state': state.status.value, 'argv': list(request.argv),
                'stdout': state.stdout, 'stderr': state.stderr, 'exit_code': state.exit_code,
                'failure_origin': state.failure_origin}
        data['logs'] = self.backend.log_info(ref)
        terminal = state.status in TERMINAL_STATUSES
        status = (ExecutionStatus.FAILED if state.status == JobStatus.FAILED else
                  ExecutionStatus.CANCELLED if state.status == JobStatus.CANCELLED else ExecutionStatus.COMPLETED)
        error = (state.stderr or f'Program exited with code {state.exit_code}') if status == ExecutionStatus.FAILED else None
        artifacts = ()
        if terminal:
            payload = {'process': asdict(ref), 'request': asdict(request), **data}
            artifacts = (self.artifacts.put(json.dumps(payload, ensure_ascii=False).encode(), kind='lab_process_result'),)
        result = OperationResult('Program ' + state.status.value if terminal else 'Program still running',
                                 artifacts, False, status, error, data)
        if terminal:
            self._terminal[ident] = result
        return deepcopy(result)

    @contextmanager
    def _parent(self, path, *, create=False):
        if not isinstance(path, str) or not path or '\x00' in path:
            raise LabToolError('Expected a nonempty workspace file path')
        if Path(path).is_absolute() or '..' in Path(path).parts or not Path(path).name:
            raise ValueError('Expected a relative workspace file path without ..')
        parts = Path(path).parts
        if not parts:
            raise ValueError('Expected a file path')
        fd = os.open(self.workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                try:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                except FileNotFoundError as exc:
                    raise LabToolError('Workspace directory does not exist: ' + path) from exc
                except NotADirectoryError as exc:
                    # O_NOFOLLOW may report ENOTDIR for symlinks too. Only a
                    # confirmed regular-file component is model-correctable.
                    if stat.S_ISREG(os.stat(part, dir_fd=fd, follow_symlinks=False).st_mode):
                        raise LabToolError('Workspace parent is a file, not a directory: ' + path) from exc
                    raise
                os.close(fd)
                fd = child
            yield fd, parts[-1]
        finally:
            os.close(fd)

    def _file(self, name, args):
        path = args['path']
        if name == 'lab_write_file':
            content = args['content']
            if not isinstance(content, str):
                raise LabToolError('File content must be text')
            try:
                content = content.encode('utf-8')
            except UnicodeEncodeError as exc:
                raise LabToolError('File content must be valid UTF-8 text') from exc
            if len(content) > self.max_file_bytes:
                raise ValueError('File write exceeds the host byte limit')
            with self._parent(path, create=True) as (parent, leaf):
                temp = '.lab-write-' + uuid4().hex
                try:
                    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(content)
                    try:
                        os.replace(temp, leaf, src_dir_fd=parent, dst_dir_fd=parent)
                    except IsADirectoryError as exc:
                        raise LabToolError('Workspace destination is a directory: ' + path) from exc
                finally:
                    try:
                        os.unlink(temp, dir_fd=parent)
                    except FileNotFoundError:
                        pass
            return OperationResult('Workspace file written', (), False, data={
                'path': path, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        offset, limit = args.get('offset', 0), args.get('max_bytes', min(65536, self.max_file_bytes))
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise LabToolError('Invalid file window')
        if limit > self.max_file_bytes:
            raise ValueError('File window exceeds the host byte limit')
        if name == 'lab_export':
            limit = self.max_file_bytes
        with self._parent(path) as (parent, leaf):
            try:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            except FileNotFoundError as exc:
                raise LabToolError('Workspace file does not exist: ' + path) from exc
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise LabToolError('Only regular files are readable')
                with os.fdopen(fd, 'rb', closefd=False) as stream:
                    stream.seek(offset)
                    content = stream.read(limit + 1)
            finally:
                os.close(fd)
        more, content = len(content) > limit, content[:limit]
        if name == 'lab_export':
            if more:
                raise ValueError('Artifact export exceeds the host byte limit')
            ref = self.artifacts.put(content, kind='lab_file')
            return OperationResult('Workspace file exported', (ref,), False, data={
                'path': path, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        return OperationResult('Workspace file window', (), False, data={
            'path': path, 'offset': offset, 'content': content.decode('utf-8', errors='replace'),
            'next_offset': offset + len(content) if more else None, 'truncated': more})

    def shutdown(self, mode):
        if mode not in ('drain', 'cancel'):
            raise ValueError('Unknown shutdown mode')
        self._accepting = False
        if self._closed:
            return True
        complete = True
        terminal = []
        errors = []
        for ident, (ref, _) in self._jobs.items():
            try:
                state = self.backend.poll(ref)
                exited = state.status in TERMINAL_STATUSES
            except Exception as exc:
                errors.append(exc)
                exited = False
            if not exited:
                complete = False
                if mode == 'cancel':
                    try:
                        self.backend.cancel(ref)
                    except Exception as exc:
                        errors.append(exc)
            else:
                terminal.append(ident)
        # Request every cancellation before an evidence-store failure can raise.
        for ident in terminal:
            try:
                self._observe(ident)  # retain evidence even without a final model poll
            except Exception as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup('Lab shutdown failed', errors)
        return complete

    def close(self):
        if self._closed:
            return
        try:
            self.shutdown('cancel')
        finally:
            self.backend.close(self.session)
        for ident in self._jobs:
            self._observe(ident)
        self._closed = True
