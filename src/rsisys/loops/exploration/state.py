from rsisys.loops.contracts.jobs import JobRef

from dataclasses import dataclass, field

from rsisys.loops.contracts.feedback import Feedback
from .rl import RLState
from rsisys.experience.records import ExperienceRecord
from rsisys.loops.contracts.directive import Directive


@dataclass
class PendingEpisode:
    directive: Directive
    record: ExperienceRecord
    feedback: Feedback
    count_completion: bool


@dataclass
class ExplorationState:
    completed: int = 0
    pending_episode: PendingEpisode | None = None
    active_directive_id: str | None = None
    last_feedback: Feedback | None = None
    recorded_training: set[JobRef] = field(default_factory=set)
    applied_training: set[JobRef] = field(default_factory=set)
    training_record_ids: dict[JobRef, str] = field(default_factory=dict)
    recorded_evaluation: set[JobRef] = field(default_factory=set)
    evaluation_record_ids: dict[JobRef, str] = field(default_factory=dict)
    rl: RLState | None = None
    measured_input_tokens: int = 0
    measured_output_tokens: int = 0
    calls_with_missing_usage: int = 0
