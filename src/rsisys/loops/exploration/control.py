"""Opt-in experiment controls exposed through ordinary permission-checked tools.

Mount definitions on a Decision Action Space and bindings on BoundDecisionTools.
No controls are auto-installed and no permission checks are bypassed.
"""
from copy import deepcopy
from dataclasses import asdict

from rsisys.loops.contracts.coordination import DirectiveUpdate, OperationResult
from rsisys.loops.decision_making.action_space import OperationCall, OperationSpec
from rsisys.loops.decision_making.action_space.capabilities import Capability, Capabilities


class ExperimentControls:
    def __init__(self, exploration, decision=None):
        self.exploration, self.decision = exploration, decision
        names = ['inspect', 'cancel', 'update'] + (['compare'] if decision is not None else [])
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
            'compare': 'Rank completed candidate evaluations using a metric, requiring known equal full training '
                       'bases, evaluation protocol, inputs and simulation mode.'}[kind]
        if kind == 'update':
            fields.update(expected_version={'type': 'integer', 'minimum': 0},
                          idempotency_key={'type': 'string', 'minLength': 1},
                          changes={'type': 'object', 'minProperties': 1})
        if kind == 'compare':
            fields = {'candidate_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2},
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
                    DirectiveUpdate(**arguments)
            return OperationCall(name, deepcopy(arguments))

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
