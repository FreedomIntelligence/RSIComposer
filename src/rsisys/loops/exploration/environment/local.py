from pathlib import Path
import shutil

from rsisys.loops.contracts.jobs import JobStatus
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.artifacts import FileArtifactStore
from . import Transition
from ..action_space import Action, ActionProposal, ActionSpec, EchoActionSpace


class CommandActionSpace(EchoActionSpace):
    def describe(self):
        return super().describe() + (ActionSpec('execute', 'Execute argv in a local workspace', {
            'type': 'object', 'properties': {'argv': {'type': 'array', 'items': {'type': 'string'}},
                                           'output_paths': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['argv'], 'additionalProperties': False,
        }),)

    def validate(self, proposal: ActionProposal, allowed_actions: tuple[str, ...]) -> Action:
        if proposal.name != 'execute':
            return super().validate(proposal, allowed_actions)
        args = proposal.arguments
        if proposal.name not in allowed_actions or not isinstance(args, dict):
            raise ValueError('execute is not allowed')
        if set(args) - {'argv', 'output_paths'} or 'argv' not in args:
            raise ValueError('execute requires argv and optional output_paths')
        argv, outputs = args['argv'], args.get('output_paths', [])
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and '\0' not in x for x in argv):
            raise ValueError('argv must be a nonempty list of strings')
        if not isinstance(outputs, list) or not all(isinstance(x, str) and x and not Path(x).is_absolute()
                                                   and '..' not in Path(x).parts for x in outputs):
            raise ValueError('output_paths must stay within the workspace')
        return Action('execute', arguments={'argv': list(argv), 'output_paths': list(outputs)})


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
