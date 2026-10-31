from pathlib import Path
import shutil

from rsisys.loops.contracts.jobs import JobStatus
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.artifacts import FileArtifactStore
from . import Transition



class LocalProcessEnvironment:
    """One command per task; persistent outputs are exported before workspace cleanup."""

    def __init__(self, backend: LocalProcessBackend, artifacts: FileArtifactStore):
        self.sandbox, self.artifacts = backend, artifacts
        self._session = None

    def reset(self, directive):
        self.close()
        self._session = self.sandbox.create(SandboxSpec('local'))
        inputs = []
        for ref in directive.input_refs:
            source = self.artifacts.resolve(ref)
            relative = f'inputs/{ref.artifact_id}'
            target = self.sandbox.file_path(self._session, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inputs.append(relative)
        return f'Task: {directive.objective}; input files: {inputs}'

    def step(self, action):
        if self._session is None:
            raise RuntimeError('Reset the environment before executing')
        if action.name not in ('echo', 'execute'):
            raise ValueError('Unsupported local action')
        argv = ('echo', action.text) if action.name == 'echo' else tuple(action.arguments['argv'])
        process = self.sandbox.exec(self._session, ExecRequest(argv))
        result = self.sandbox.wait(process)
        refs = [self.artifacts.put((result.stdout + result.stderr).encode(), kind='execution_output')]
        error = result.stderr or f'exit={result.exit_code}' if result.status != JobStatus.SUCCEEDED else None
        if result.status == JobStatus.SUCCEEDED:
            for relative in action.arguments.get('output_paths', []):
                path = self.sandbox.file_path(self._session, relative)
                if not path.is_file():
                    error = f'Expected output is missing: {relative}'
                    break
                refs.append(self.artifacts.put_file(path, kind='execution_file'))
        return Transition(result.stdout, True, False, artifacts=tuple(refs), error=error)

    def close(self):
        if self._session is not None:
            self.sandbox.close(self._session)
            self._session = None
