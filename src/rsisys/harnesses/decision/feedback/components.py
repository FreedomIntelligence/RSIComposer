"""Optional archive storing detached native events in durable delivery references."""
from copy import deepcopy
from dataclasses import asdict

from rsisys.loops.contracts.feedback_modules import ArchiveEvent


class JournalTrajectoryArchive:
    """The feedback journal persists each returned event reference atomically."""
    def write(self, event):
        return asdict(event)

    def read(self, reference):
        return ArchiveEvent(**deepcopy(reference))
