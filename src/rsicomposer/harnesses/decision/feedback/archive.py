"""Raw archive events. No dataset construction."""
from copy import deepcopy
from dataclasses import asdict

from rsicomposer.harnesses.decision.feedback.knowledge import (
    trajectory_records,
    trajectory_snapshot,
)
from rsicomposer.loops.contracts.feedback_modules import ArchiveEvent
from rsicomposer.runtime.storage.files import json_bytes
from rsicomposer.runtime.storage.json_tools import EvidencePathError, result_id


def validate_archive(archive):
    if archive is None:
        return
    if all(callable(getattr(archive, name, None)) for name in ('write', 'read')):
        return
    raise TypeError('Trajectory archive requires write(event)/read(reference), or None')


def captured_archive_event(row):
    snapshot = trajectory_snapshot(row)
    return ArchiveEvent(row['job_id'], 'execution_feedback', snapshot,
        {name: deepcopy(snapshot[name]) for name in
         ('task_id', 'decision_id', 'lab_id', 'run_id', 'agent_id', 'issuing_model')})


class ArchiveBinding:
    def __init__(self, archive):
        validate_archive(archive)
        self.archive = archive

    @property
    def requires_delivery(self):
        return self.archive is not None

    def write(self, event):
        reference = self.archive.write(deepcopy(event))
        json_bytes(reference)
        return deepcopy(reference)

    def event(self, delivery):
        if delivery.get('status') != 'delivered':
            raise EvidencePathError('Archive event has not been delivered')
        value = self.archive.read(deepcopy(delivery['reference']))
        if not isinstance(value, ArchiveEvent):
            raise TypeError('Archive read must return the original ArchiveEvent')
        if result_id(asdict(value)) != delivery['event_digest']:
            raise EvidencePathError('Archive read differs from the original event')
        return deepcopy(value)

    def records(self, jobs, *, research_attempt_id=None):
        if self.archive is None:
            return []
        records = trajectory_records(jobs, research_attempt_id=research_attempt_id)
        deliveries = {row['job_id']: row.get('archive_delivery', {'status': 'pending'}) for row in jobs}
        for record in records:
            record['archive_delivery'] = deepcopy(deliveries[record['job_id']])
        return records

    def snapshot(self, row):
        return self.event(row.get('archive_delivery', {})).payload
