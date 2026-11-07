"""Mechanical observation classification, independent of a Lab's tool names."""

HOUSEKEEPING = frozenset(('research_archive', 'research_read', 'research_patch',
    'research_task', 'research_task_read', 'memory_read', 'memory_search', 'memory_write', 'memory_delete',
    'reflect_report', 'system_status'))


def native_observation(row):
    if row.kind == 'feedback':
        return True
    operation = row.details.get('operation') or {}
    return (row.kind == 'operation' and bool(operation.get('name'))
            and operation['name'] not in HOUSEKEEPING
            and isinstance(row.details.get('result'), dict))


def failed_observation(row):
    if not native_observation(row):
        return False
    result = row.details.get('result', row.details.get('feedback', {}))
    return (result.get('status') in ('failed', 'cancelled', 'rejected')
            or bool(result.get('error')) or bool(row.details.get('validation_error')))


def task_observation(row):
    """Retain subsequent producer feedback without depending on tool names.

    Completion of a file read or a running-job poll is not an experiment result.
    Opaque producers can identify their feedback with FeedbackDelivery; legacy
    measurements, terminal job envelopes and failures also remain discoverable.
    Classification retains evidence, and does not verify a claimed outcome.
    """
    if not native_observation(row):
        return False
    if row.kind == 'feedback' or failed_observation(row):
        return True
    result = row.details['result']
    if result.get('feedback_delivery'):
        return True
    data = result.get('data', {})
    if not isinstance(data, dict):
        return False
    if any(data.get(key) for key in ('quality_metrics', 'training_metrics', 'execution_metrics')):
        return True
    state = data.get('state')
    if isinstance(state, dict):
        state = state.get('status')
    return bool(data.get('job_id') and state in ('succeeded', 'completed', 'failed', 'cancelled', 'rejected'))
