"""Mechanical observation classification, independent of a Lab's tool names."""

HOUSEKEEPING = frozenset(('research_archive', 'research_read', 'research_patch',
    'research_task', 'memory_read', 'memory_search', 'memory_write', 'memory_delete',
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
