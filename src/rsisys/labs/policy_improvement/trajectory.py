"""Read-only bridge from the feedback archive to an improvement Lab.

References pin original capture bytes, not a changing memory/proposal view. No
dataset recipe, model inference or training is performed by this adapter.
"""
from copy import deepcopy

from rsisys.loops.contracts.improvement import TrajectoryRef
from rsisys.runtime.json_tools import result_id


class FeedbackTrajectorySource:
    def __init__(self, service, *, archive_id):
        if not isinstance(archive_id, str) or not archive_id.strip():
            raise ValueError('An explicit archive identity is required')
        self.service, self.archive_id = service, archive_id

    def reference(self, job_id):
        value = self.service.trajectory_snapshot(job_id)
        return TrajectoryRef(self.archive_id, value['trajectory_id'], result_id(value))

    def read(self, ref):
        if not isinstance(ref, TrajectoryRef):
            raise TypeError('Expected TrajectoryRef')
        ref.__post_init__()
        if ref.archive_id != self.archive_id:
            raise ValueError('Trajectory belongs to a different archive')
        prefix = 'trajectory:'
        if not ref.trajectory_id.startswith(prefix):
            raise ValueError('Unknown trajectory identity')
        value = self.service.trajectory_snapshot(ref.trajectory_id[len(prefix):])
        if value['trajectory_id'] != ref.trajectory_id or result_id(value) != ref.digest:
            raise ValueError('Archived trajectory identity or digest differs')
        return deepcopy(value)
