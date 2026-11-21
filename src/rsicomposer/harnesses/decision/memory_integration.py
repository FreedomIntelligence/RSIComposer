"""Tool/context integration owned by the default harness, not by ResearchAssembly."""
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationArgumentError, OperationCall
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.runtime.memory import memory_host


class MemoryActionSpace:
    def __init__(self, base, memory):
        self.names = frozenset(spec.name for spec in memory.specifications())
        if self.names.intersection(spec.name for spec in base.describe()):
            raise ValueError('Memory tools conflict with the installed action catalog')
        self.base, self.memory = base, memory

    @property
    def permissions(self):
        return self.base.permissions

    def __getattr__(self, name):
        return getattr(self.base, name)

    def describe(self):
        return (*self.base.describe(), *self.memory.specifications())

    def validate(self, call):
        return self.memory.validate(call) if call.name in self.names else self.base.validate(call)


class MemoryTools:
    def __init__(self, base, memory):
        self.base, self.memory = base, memory
        self.names = frozenset(spec.name for spec in memory.specifications())

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def definitions(self):
        from types import SimpleNamespace
        return (*self.base.definitions, *(SimpleNamespace(spec=spec,
            validate=lambda arguments, name=spec.name: self.memory.validate(OperationCall(name, arguments)))
            for spec in self.memory.specifications()))

    def execute(self, call):
        if call.name not in self.names:
            return self.base.execute(call)
        try:
            data = self.memory.execute(call)
        except OperationArgumentError as exc:
            return OperationResult('Memory/task arguments rejected; correct the next decision', (), False,
                                   ExecutionStatus.FAILED, str(exc))
        return OperationResult('Meta-agent memory/task operation completed', (), False, data=data)


class DefaultMemoryHost:
    def __init__(self, harness):
        self.harness = harness

    def history(self):
        return self.harness.memory.history()

    def install(self, loop, memory):
        if loop.harness is not self.harness:
            raise ValueError('Memory host belongs to another loop')
        space = MemoryActionSpace(loop.action_space, memory)
        policy = getattr(loop.policy, 'implementation', loop.policy)
        parser = getattr(policy, 'parser', policy)
        if hasattr(parser, 'space'):
            parser.space = space
        loop.action_space = space
        self.harness.install_memory(memory)


class AgentMemoryLifecycle:
    """Adapter for agent-authored memory; other providers own their own lifecycle."""
    def __init__(self, configuration):
        self.configuration = configuration

    def prepare(self):
        self.configuration.prepare()

    def validate_host(self, loop):
        memory_host(loop.harness)

    def for_run(self, loop, path):
        memory = self.configuration.clone()
        memory.bind(loop.harness, path)
        return memory

    def install(self, loop, memory, *, path=None):
        memory_host(loop.harness).install(loop, memory)
