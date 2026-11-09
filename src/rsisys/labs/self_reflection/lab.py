"""Non-executing Self-Reflection Lab scaffold with discoverable operations.

The scaffold allocates no store, jobs or workers. Every valid invocation returns
an explicit failed receipt; it never reports an accepted job, candidate or gain.
The draft backend contracts are intentionally not wired into this placeholder.
"""
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus


VERSION = '0-placeholder'
_FIELDS = {
    'record_episode': ('episode_ref',),
    'build_candidate': ('baseline_version_ref', 'experience_set_ref', 'proposal_ref'),
    'start': ('candidate_ref', 'baseline_version_ref', 'task_suite_ref', 'policy_ref'),
    'poll': ('validation_id',),
    'result': ('validation_id',),
    'cancel': ('validation_id',),
}


def validate_config(config):
    if config:
        raise ValueError('Self-Reflection Lab is a placeholder; backend configuration is not implemented')


class SelfReflectionLab:
    """An optional Lab specialization, not a controller or an adoption policy."""
    def __init__(self, context, config):
        validate_config(config)

    def describe(self):
        return tuple(OperationSpec(name,
            f'PLACEHOLDER: self-reflection {name}; execution is not implemented', {
                'type': 'object',
                'properties': {key: {'type': 'string', 'minLength': 1} for key in fields},
                'required': list(fields), 'additionalProperties': False,
            }, read_only=name in {'poll', 'result'}) for name, fields in _FIELDS.items())

    def execute(self, call):
        fields = _FIELDS.get(call.name)
        if fields is None or not isinstance(call.arguments, dict) or set(call.arguments) != set(fields):
            raise LabToolError('Unknown self-reflection operation or invalid argument names')
        if any(not isinstance(value, str) or not value.strip() for value in call.arguments.values()):
            raise LabToolError('Self-reflection references must be nonempty strings')
        return OperationResult(
            'Self-Reflection Lab is not implemented', (), False,
            status=ExecutionStatus.FAILED,
            error='self_reflection_not_implemented: persistent experience, candidate construction and validation are placeholders',
            data={'implementation_status': 'placeholder', 'operation': call.name,
                  'error_code': 'self_reflection_not_implemented'},
        )
