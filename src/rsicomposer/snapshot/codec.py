"""The bundled checkpoint schema. Importing it does not mutate a type registry."""
from rsicomposer.harnesses.decision.learner import DecisionTrainingRequest
from rsicomposer.harnesses.decision.state import (
    DecisionContext,
)
from rsicomposer.harnesses.memory import HistoryEntry, Memory
from rsicomposer.harnesses.decision.state import ControllerHarnessState
from rsicomposer.host.execution import HostExecutionState
from rsicomposer.host.ledger import _PendingOperationRecord
from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.loops.contracts.candidate import Candidate
from rsicomposer.loops.contracts.coordination import (
    LoopStepResult,
    OperationResult,
)
from rsicomposer.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsicomposer.loops.contracts.controller_version import ControllerVersion, EvidenceScope
from rsicomposer.loops.contracts.delivery import FeedbackDelivery
from rsicomposer.loops.contracts.feedback import Feedback
from rsicomposer.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus, OperationProgress, JobControl
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.model_service import ModelRef, TokenUsage
from rsicomposer.loops.contracts.profiles import MetricSpec, SplitRole, Invariant, LabProfile
from rsicomposer.loops.contracts.requests import EvaluationSpec, PromptSpec, RLSpec, SFTSpec
from rsicomposer.loops.contracts.training import ResumeMode, TrainingOrigin, TrainingResult
from rsicomposer.runtime.evidence.records import ExperienceRecord
from rsicomposer.runtime.orchestration.run_policy import EvaluationOutcome, ExperimentOutcome, RunDecision
from rsicomposer.runtime.orchestration.runner import RunnerStep
from rsicomposer.runtime.storage.codec import decode, encode

__all__ = ['decode', 'encode']

STATE_TYPES = (
    ControllerVersion, EvidenceScope,
    MetricSpec, SplitRole, Invariant, LabProfile,
    OperationProgress, JobControl, Decision, RunnerStep, RunDecision, ExperimentOutcome, EvaluationOutcome, JobOutcome,
    ExperienceRecord, Memory, HistoryEntry, DecisionContext, OperationSpec,
    _PendingOperationRecord, DecisionTrainingRequest,
    ArtifactRef, Candidate, Feedback, FeedbackDelivery, JobRef, ModelRef, TokenUsage,
    LoopStepResult, JobProgress,
    TrainingResult, TrainingOrigin, OperationCall, OperationResult, HostExecutionState, ControllerHarnessState,
    SFTSpec, RLSpec, EvaluationSpec, PromptSpec,
)
STATE_ENUMS = (JobStatus, ExecutionStatus, ResumeMode)
