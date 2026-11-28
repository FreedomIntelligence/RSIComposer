"""Run-local bindings for agent-authored Memory; no Policy or Harness mutation."""
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationArgumentError, OperationCall
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.decision_making.action_space.capabilities import Capability
from rsicomposer.runtime.capabilities import optional_method


def memory_capabilities(memory):
    def execute(call):
        try:
            data = memory.execute(call)
        except OperationArgumentError as exc:
            return OperationResult('Memory/task arguments rejected; correct the next decision', (), False,
                                   ExecutionStatus.FAILED, str(exc))
        return OperationResult('Meta-agent memory/task operation completed', (), False, data=data)

    return tuple(Capability.from_spec(spec,
        lambda arguments, name=spec.name: memory.validate(OperationCall(name, arguments)),
        lambda arguments, name=spec.name: execute(OperationCall(name, arguments)))
        for spec in memory.specifications())


class AgentMemoryLifecycle:
    def __init__(self, configuration):
        self.configuration = configuration

    def prepare(self):
        self.configuration.prepare()

    @staticmethod
    def _history(loop):
        local = optional_method(loop.harness, 'history')
        if local is not None:
            return local
        execution = loop.host_execution
        history = optional_method(getattr(execution, 'memory', None), 'history')
        if history is None:
            raise TypeError('AgentMemory requires a read-only history provider')

        def version_history():
            # Four-port Harnesses need not keep a second archive. Never use a
            # previous version's records as the new Controller's working memory.
            version = getattr(loop, 'controller_version', None)
            records = execution.state.decision_records
            return tuple(entry for entry in history() if version is None or
                         records.get(entry.details.get('decision_id', entry.record_id), {}).get('controller_version') == version)
        return version_history

    def validate_host(self, loop):
        self._history(loop)

    def for_run(self, loop, path):
        memory = self.configuration.clone()
        memory.bind(self._history(loop), path)
        return memory

    def install(self, loop, memory, *, path=None):
        pass  # The shared assembly installs capabilities; no provider-specific side effects.
