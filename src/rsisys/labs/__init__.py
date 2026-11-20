"""Lab execution contracts and registration of reusable built-in Labs."""
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import ExecutionLifecycle, LabExecution, LabToolError


def register_builtins(registry):
    from .scenes.coding import register_components as register_coding
    from .scenes.math import register_components as register_math
    from .scenes.self_reflection import register as register_self_reflection

    register_math(registry)
    register_coding(registry)
    register_self_reflection(registry)


__all__ = ['LabExecution', 'OperationCall', 'OperationResult', 'ExecutionLifecycle', 'LabToolError']
