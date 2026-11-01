"""Opt-in experiment controls exposed through ordinary permission-checked tools.

Mount definitions on a Decision Action Space and bindings on BoundDecisionTools.
No controls are auto-installed and no permission checks are bypassed.
"""
from copy import deepcopy
from dataclasses import asdict

from rsisys.loops.contracts.backend import BackendCapabilities, validate_update_capabilities
from rsisys.loops.contracts.coordination import DirectiveUpdate, OperationResult
from rsisys.loops.decision_making.action_space import OperationCall, OperationSpec
from rsisys.loops.decision_making.action_space.capabilities import Capability, Capabilities


class ExperimentControls:
    """Explicit controls, optionally bounded by one configured backend's declaration.

    Omit backend_capabilities for heterogeneous endpoints or legacy backends;
    execution still owns per-target support and value checks. A declaration
    limits update field names, not their values, and grants no checkpoint or
    reattach operation.
    """

    def __init__(self, exploration, decision=None, *, backend_capabilities: BackendCapabilities | None = None):
        if backend_capabilities is not None:
            if not isinstance(backend_capabilities, BackendCapabilities):
                raise TypeError('Expected BackendCapabilities')
            backend_capabilities.__post_init__()
        self.exploration, self.decision = exploration, decision
        self.backend_capabilities = backend_capabilities
        names = ['inspect', 'cancel']
        if backend_capabilities is None or backend_capabilities.update_fields:
            names.append('update')
        if decision is not None:
            names.append('compare')
        self.capabilities = Capabilities(self._capability(name) for name in names)

    def describe(self):
        return tuple(self.capabilities)

    @property
    def definitions(self):
        return self.capabilities.definitions

    @property
    def bindings(self):
        return self.capabilities.bindings

    def _capability(self, kind):
        fields = {'directive_id': {'type': 'string', 'minLength': 1}}
        description = {'inspect': 'Inspect one experiment by directive_id, including its current update version.',
            'cancel': 'Request cancellation of only the named experiment; poll until terminal.',
            'update': 'Update only backend-supported live controls. Use inspected expected_version and a stable '
                      'idempotency_key; retry uncertain responses with identical arguments. Unsupported fields '
                      'or backends are rejected. This does not replace the original directive.',
            'compare': 'Rank completed candidate evaluations using a metric, requiring the same explicit parent '
                       'system or known equal full training bases, plus equal evaluation protocol, inputs and '
                       'simulation mode.'}[kind]
        if kind == 'update':
            fields.update(expected_version={'type': 'integer', 'minimum': 0},
                          idempotency_key={'type': 'string', 'minLength': 1},
                          changes={'type': 'object', 'minProperties': 1, 'additionalProperties': True})
            if self.backend_capabilities is not None:
                fields['changes'].update(
                    properties={field: {} for field in self.backend_capabilities.update_fields},
                    additionalProperties=False)
        if kind == 'compare':
            fields = {'candidate_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1},
                                        'minItems': 2, 'uniqueItems': True},
                      'metric': {'type': 'string', 'minLength': 1}, 'higher_is_better': {'type': 'boolean'}}
        name = f'experiment_{kind}'
        spec = OperationSpec(name, description, {'type': 'object', 'properties': fields,
            'required': list(fields), 'additionalProperties': False}, read_only=kind in ('inspect', 'compare'))

        def validate(arguments):
            if not isinstance(arguments, dict) or set(arguments) != set(fields):
                raise ValueError(f'{name} requires exactly {tuple(fields)}')
            if kind == 'compare':
                ids = arguments['candidate_ids']
                if (not isinstance(ids, (list, tuple)) or len(ids) < 2
                        or any(not isinstance(ident, str) or not ident for ident in ids)
                        or len(set(ids)) != len(ids) or type(arguments['higher_is_better']) is not bool
                        or not isinstance(arguments['metric'], str) or not arguments['metric']):
                    raise ValueError('Invalid comparison arguments')
            else:
                if not isinstance(arguments['directive_id'], str) or not arguments['directive_id']:
                    raise ValueError('A directive ID is required')
                if kind == 'update':
                    request = DirectiveUpdate(**arguments)
                    if self.backend_capabilities is not None:
                        validate_update_capabilities(self.backend_capabilities, tuple(request.changes))
            normalized = deepcopy(arguments)
            if kind == 'compare':
                normalized['candidate_ids'] = list(normalized['candidate_ids'])
            return OperationCall(name, normalized)

        def execute(arguments):
            arguments = validate(arguments).arguments
            if kind == 'inspect':
                progress = self.exploration.inspect(arguments['directive_id'])
                return OperationResult('Experiment progress', (), progress.directive.simulated,
                                       data=asdict(progress))
            if kind == 'cancel':
                progress = self.exploration.cancel(arguments['directive_id'])
                return OperationResult('Cancellation requested; inspect terminal status', (),
                                       progress.directive.simulated, data=asdict(progress))
            if kind == 'update':
                receipt = self.exploration.update(DirectiveUpdate(**arguments))
                progress = self.exploration.inspect(arguments['directive_id'])
                return OperationResult('Backend update confirmed', (), progress.directive.simulated,
                                       data=asdict(receipt))
            rows = self.decision.compare_candidates(arguments['candidate_ids'], metric=arguments['metric'],
                                                    higher_is_better=arguments['higher_is_better'])
            return OperationResult('Compared completed evaluations', (), any(row.simulated for row in rows),
                                   data={'ranking': [asdict(row) for row in rows], 'metric': arguments['metric']})

        return Capability.from_spec(spec, validate, execute)
