"""Adapt one public Lab catalog to the default decision harness."""
from copy import deepcopy
from functools import partial
import json

from rsisys.loops.contracts.coordination import ExecutionView, OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.execution import LabExecution, LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space.capabilities import Capability, Capabilities
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools


def _arguments(name, arguments):
    # Only the transport envelope is standardized. Domain/schema validation is
    # the Lab's responsibility, immediately before effects inside execute().
    if not isinstance(arguments, dict) or any(not isinstance(key, str) for key in arguments):
        raise TypeError('Lab arguments must be a JSON object with string keys')
    json.dumps(arguments, allow_nan=False)
    return OperationCall(name, deepcopy(arguments))


class LabExecutionTools(BoundDecisionTools):
    def __init__(self, lab: LabExecution, *, extra_definitions=(), extra_bindings=None):
        if not callable(getattr(lab, 'describe', None)) or not callable(getattr(lab, 'execute', None)):
            raise TypeError('Lab execution requires describe() and execute(call)')
        self.lab = lab
        capabilities = []
        for spec in lab.describe():
            if not isinstance(spec, OperationSpec):
                raise TypeError('Lab describe() must return OperationSpec entries')
            spec = deepcopy(spec)
            if not isinstance(spec.description, str) or not isinstance(spec.input_schema, dict):
                raise TypeError('Lab descriptions require text and a JSON schema object')
            spec.__post_init__()
            json.dumps(spec.input_schema, allow_nan=False)
            capabilities.append(Capability.from_spec(
                spec, partial(_arguments, spec.name), partial(self._invoke, spec.name)))
        catalog = Capabilities(capabilities)
        bindings = dict(extra_bindings or {})
        if set(bindings) & catalog.bindings.keys():
            raise ValueError('Lab and additional operation names must be unique')
        self.definitions = (*catalog.definitions, *extra_definitions)
        super().__init__({**catalog.bindings, **bindings})

    def _invoke(self, name, arguments):
        call = _arguments(name, arguments)
        try:
            return self.lab.execute(call)
        except LabToolError as exc:
            return OperationResult('Lab tool failed; use this feedback to choose the next action', (),
                False, ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}')


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
