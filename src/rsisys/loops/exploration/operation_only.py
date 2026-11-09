"""Inactive exploration endpoint for assemblies executing only Lab operations."""
from rsisys.loops.contracts.coordination import ExecutionView


class OperationOnlyExploration:
    """Runner compatibility for an assembly using only Lab operation calls.

    No Actor is constructed and no domain commands are inferred. An independent
    exploration endpoint may still be supplied explicitly by the host.
    """
    def pending(self):
        return None

    def execution_view(self):
        return ExecutionView()

    def _unsupported(self, *args):
        raise ValueError('This assembly executes Lab tools through invoke, not exploration directives')

    accept = advance = inspect = cancel = acknowledge = _unsupported
