"""State types of the bundled harnesses and contracts, registered with the runtime codec.

Importing this module is what makes built-in checkpoints decodable; the
application composition root imports it. Custom components register their
own dataclasses with rsisys.runtime.storage.codec.register_types.
"""
from rsisys.harnesses.decision.learner import DecisionTrainingRequest
from rsisys.harnesses.decision.state import (
    DecisionContext,
)
from rsisys.harnesses.memory import HistoryEntry, Memory
from rsisys.harnesses.decision.state import ControllerHarnessState
from rsisys.host.execution import HostExecutionState
from rsisys.host.ledger import _PendingOperationRecord
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.coordination import (
    LoopStepResult,
    OperationResult,
)
from rsisys.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsisys.loops.contracts.controller_version import ControllerVersion, EvidenceScope
from rsisys.loops.contracts.delivery import FeedbackDelivery
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus, OperationProgress, JobControl
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef, TokenUsage
from rsisys.loops.contracts.profiles import MetricSpec, SplitRole, Invariant, LabProfile
from rsisys.loops.contracts.requests import EvaluationSpec, PromptSpec, RLSpec, SFTSpec
from rsisys.loops.contracts.training import ResumeMode, TrainingOrigin, TrainingResult
from rsisys.runtime.evidence.records import ExperienceRecord
from rsisys.runtime.orchestration.run_policy import ExperimentOutcome, RunDecision
from rsisys.runtime.orchestration.runner import RunnerStep
from rsisys.runtime.storage.codec import decode, encode, register_enums, register_types

__all__ = ['decode', 'encode']

register_types(
    ControllerVersion, EvidenceScope,
    MetricSpec, SplitRole, Invariant, LabProfile,
    OperationProgress, JobControl, Decision, RunnerStep, RunDecision, ExperimentOutcome, JobOutcome,
    ExperienceRecord, Memory, HistoryEntry, DecisionContext, OperationSpec,
    _PendingOperationRecord, DecisionTrainingRequest,
    ArtifactRef, Candidate, Feedback, FeedbackDelivery, JobRef, ModelRef, TokenUsage,
    LoopStepResult, JobProgress,
    TrainingResult, TrainingOrigin, OperationCall, OperationResult, HostExecutionState, ControllerHarnessState,
    SFTSpec, RLSpec, EvaluationSpec, PromptSpec,
)
register_enums(JobStatus, ExecutionStatus, ResumeMode)
