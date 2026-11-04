"""Route immutable producer events without inspecting native business fields."""
from copy import deepcopy

from rsisys.runtime.json_tools import result_id


def producer_delivery(observation, jobs, *, archive_only):
    """Return persisted event identity and optional original processing job.

    Raw collection receipts retain their own IDs. The first observation reserves
    processing even if its worker later fails; unknown effects are not retried
    by another read. Explicit reprocess creates a separate, accounted attempt.
    """
    delivery = observation.feedback_delivery
    if delivery is None:
        return None, None
    result = deepcopy(observation.result)
    if observation.source == 'operation':
        result.pop('feedback_delivery', None)
    event = {'key': result_id([observation.lab_id, observation.run_id, delivery.event_id]),
             'event_id': delivery.event_id, 'result_digest': result_id(result),
             'reprocess': delivery.reprocess}
    original = None
    for row in sorted(jobs, key=lambda row: (row.get('capture_index', -1), row['job_id'])):
        previous = row.get('producer_event')
        if previous is None or previous['key'] != event['key']:
            continue
        if previous['result_digest'] != event['result_digest']:
            raise ValueError('Feedback producer event identity reused for a different result')
        if (original is None and not row.get('capture_archive_only', row.get('archive_only', False))
                and not row.get('duplicate_of')):
            original = row['job_id']
    return event, None if archive_only or delivery.reprocess else original
