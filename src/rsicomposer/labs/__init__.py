"""Lab execution contracts and registration of reusable built-in Labs."""
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationCall
from rsicomposer.loops.contracts.execution import ExecutionLifecycle, LabExecution, LabToolError


def register_builtins(registry):
    from .coding import register_components as register_coding
    from .computer_use import register_components as register_computer_use
    from .math import register_components as register_math
    from .professional_work import register_components as register_professional_work
    from .search import register_components as register_search
    from .self_reflection import register as register_self_reflection

    register_math(registry)
    register_coding(registry)
    register_computer_use(registry)
    register_professional_work(registry)
    register_search(registry)
    register_self_reflection(registry)


__all__ = ['LabExecution', 'OperationCall', 'OperationResult', 'ExecutionLifecycle', 'LabToolError']
