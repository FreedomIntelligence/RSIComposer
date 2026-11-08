"""Route namespaced operations and own registered Lab lifecycles."""
from copy import deepcopy
from dataclasses import replace
import json
from ..loops.contracts.decision import OperationCall, OperationSpec
from ..runtime.capabilities import optional_method


class LabGroup:
    """Expose multiple Labs as alias.operation through the existing Lab port."""
    def __init__(self):
        self._labs, self._operations, self._specs = {}, {}, []
        self._owned = []

    def add(self, alias, lab):
        # Own a newly constructed provider before validating its catalog.
        close = optional_method(lab, 'close')
        if close is not None:
            self._owned.append(close)
        if not callable(getattr(lab, 'describe', None)) or not callable(getattr(lab, 'execute', None)):
            raise TypeError(f'Lab {alias} must implement describe/execute')
        if any(existing is lab for existing in self._labs.values()):
            if close is not None:
                self._owned.pop()
            raise ValueError('Lab aliases must own distinct instances')
        self._labs[alias] = lab
        for spec in lab.describe():
            if not isinstance(spec, OperationSpec) or not isinstance(spec.name, str) or not spec.name:
                raise TypeError('Lab catalogs must contain named OperationSpec entries')
            name = f'{alias}.{spec.name}'
            if name in self._operations:
                raise ValueError(f'Duplicate Lab operation: {name}')
            self._operations[name] = (lab, spec.name)
            self._specs.append(replace(spec, name=name))

    def describe(self):
        return deepcopy(tuple(self._specs))

    def execute(self, call):
        if call.name not in self._operations:
            raise ValueError(f'Unknown Lab operation: {call.name}')
        lab, name = self._operations[call.name]
        return lab.execute(OperationCall(name, deepcopy(call.arguments)))

    def runtime_context(self):
        values = {}
        for alias, lab in self._labs.items():
            read = optional_method(lab, 'runtime_context')
            if read is not None:
                value = read()
                if not isinstance(value, str):
                    raise TypeError('Lab runtime_context must return text')
                values[alias] = value
        return json.dumps(values, ensure_ascii=False) if values else ''

    def shutdown(self, mode):
        complete, errors = True, []
        for lab in self._labs.values():
            shutdown = optional_method(lab, 'shutdown')
            if shutdown is not None:
                try:
                    stopped = shutdown(mode)
                    if type(stopped) is not bool:
                        raise TypeError('Lab shutdown must confirm completion with a bool')
                    complete = stopped and complete
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup('Lab shutdown failed', errors)
        return complete

    def close(self):
        failed, errors = [], []
        for close in reversed(self._owned):
            try:
                close()
            except Exception as exc:
                failed.append(close)
                errors.append(exc)
        self._owned = list(reversed(failed))
        if errors:
            raise ExceptionGroup('Lab cleanup failed', errors)
