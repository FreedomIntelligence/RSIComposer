"""Raw archive window and journal compatibility. No dataset construction."""
from copy import deepcopy
from dataclasses import asdict

from rsisys.loops.contracts.feedback_modules import ArchiveEvent
from rsisys.runtime.files import json_bytes
from rsisys.runtime.json_tools import EvidencePathError, result_id
from .knowledge import trajectory_records, trajectory_snapshot


def validate_archive(archive):
    if archive is None:
        return
    if all(callable(getattr(archive, name, None)) for name in ('write', 'read')):
        return
    if callable(getattr(archive, 'records', None)):
        return  # Historical journal archive, isolated by ArchiveBinding.
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
        self.legacy = archive is not None and not all(
            callable(getattr(archive, name, None)) for name in ('write', 'read'))

    @property
    def requires_delivery(self):
        return self.archive is not None and not self.legacy

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
        if self.legacy:
            value = self.archive.records(deepcopy(jobs), research_attempt_id=research_attempt_id)
            if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                raise TypeError('Legacy trajectory archive must return a list of JSON objects')
            return deepcopy(value)
        records = trajectory_records(jobs, research_attempt_id=research_attempt_id)
        deliveries = {row['job_id']: row.get('archive_delivery', {'status': 'pending'}) for row in jobs}
        for record in records:
            record['archive_delivery'] = deepcopy(deliveries[record['job_id']])
        return records

    def extract(self, row):
        if not self.legacy:
            return self.event(row.get('archive_delivery', {})).payload
        return self._legacy('extract', row, 'segment extraction')

    def snapshot(self, row):
        if not self.legacy:
            return self.event(row.get('archive_delivery', {})).payload
        return self._legacy('snapshot', row, 'immutable snapshots')

    def _legacy(self, name, row, description):
        operation = getattr(self.archive, name, None)
        if not callable(operation):
            raise EvidencePathError('Configured archive does not provide ' + description)
        value = operation(deepcopy(row))
        if not isinstance(value, dict):
            raise TypeError('Legacy trajectory output must be a JSON object')
        json_bytes(value)
        return deepcopy(value)
