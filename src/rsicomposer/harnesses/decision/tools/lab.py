"""Adapt one public Lab catalog to the default decision harness."""
import json
from copy import deepcopy
from dataclasses import replace
from functools import partial

from rsicomposer.harnesses.decision.tools import BoundDecisionTools
from rsicomposer.loops.contracts.decision import OperationCall, OperationSpec
from rsicomposer.loops.contracts.execution import LabExecution, execute_lab_call
from rsicomposer.loops.decision_making.action_space.capabilities import (
    Capabilities,
    Capability,
)
from rsicomposer.runtime.capabilities import optional_method


def _arguments(name, arguments):
    # Only the transport envelope is standardized. input_schema describes the
    # arguments to the Controller and is not enforced here: domain/schema
    # validation is the Lab's responsibility, immediately before effects inside
    # execute(), where it can reject a call with LabToolError.
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
        self._lab_names = frozenset(spec.name for spec in lab.describe())
        catalog = Capabilities(capabilities)
        bindings = dict(extra_bindings or {})
        if set(bindings) & catalog.bindings.keys():
            raise ValueError('Lab and additional operation names must be unique')
        self.definitions = (*catalog.definitions, *extra_definitions)
        super().__init__({**catalog.bindings, **bindings})

    def execute(self, call):
        if call.name in self._lab_names:
            # Keep call_id: the Lab may use it as an idempotency key.
            return self._invoke(call.name, call.arguments, call_id=call.call_id)
        return super().execute(call)

    def retry(self, call, *, after_sequence=None):
        retry = optional_method(self.lab, 'retry')
        if call.name not in self._lab_names or retry is None:
            raise ValueError('Lab does not provide an explicit infrastructure retry contract')
        return retry(deepcopy(call), after_sequence=after_sequence)

    def _invoke(self, name, arguments, *, call_id=None):
        call = _arguments(name, arguments)
        if call_id is not None:
            call = replace(call, call_id=call_id)
        return execute_lab_call(self.lab, call)

    def runtime_context(self):
        read = optional_method(self.lab, 'runtime_context')
        value = read() if read is not None else ''
        if not isinstance(value, str):
            raise TypeError('Lab runtime_context() must return text')
        return value
