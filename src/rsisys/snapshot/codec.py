"""Explicit JSON codec for known local state types. Never loads Python objects by name."""

from dataclasses import fields, is_dataclass
from enum import Enum

from rsisys.harnesses.actor.commands import CommandState
from rsisys.harnesses.actor.evaluation import EvaluationRequest, EvaluationResult
from rsisys.harnesses.actor.learner import ActorTrainingRequest
from rsisys.harnesses.actor.rl import RLPlan, RLState
from rsisys.harnesses.actor.state import ExplorationState, PendingEpisode
from rsisys.harnesses.decision.learner import DecisionTrainingRequest
from rsisys.harnesses.decision.state import (
    CandidateEvaluation,
    DecisionContext,
    DecisionState,
    DirectiveProvenance,
    _PendingOperationRecord,
    _PendingReviewRecord,
    _ReviewSession,
)
from rsisys.harnesses.memory import HistoryEntry, Memory
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.coordination import (
    DirectiveUpdate,
    ExecutionView,
    ExplorationProgress,
    LoopStepResult,
    OperationResult,
    ReviewDecision,
    UpdateReceipt,
)
from rsisys.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsisys.loops.contracts.delivery import FeedbackDelivery
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef, TokenUsage
from rsisys.loops.contracts.requests import EvaluationSpec, PromptSpec, RLSpec, SFTSpec
from rsisys.loops.contracts.training import ResumeMode, TrainingOrigin, TrainingResult
from rsisys.runtime.evidence.records import ExperienceRecord
from rsisys.runtime.orchestration.run_policy import ExperimentOutcome, RunDecision
from rsisys.runtime.orchestration.runner import RunnerStep

TYPES = {cls.__name__: cls for cls in (
    Decision, RunnerStep, RunDecision, ExperimentOutcome, JobOutcome, DirectiveUpdate, UpdateReceipt,
    ExperienceRecord, Memory, HistoryEntry, DecisionContext, OperationSpec,
    _PendingOperationRecord, _PendingReviewRecord, _ReviewSession, DecisionTrainingRequest,
    CommandState, PendingEpisode, EvaluationRequest, EvaluationResult,
    ArtifactRef, Candidate, SystemVersion, Directive, Feedback, FeedbackDelivery, JobRef, ModelRef, TokenUsage,
    ExecutionView, ExplorationProgress, ReviewDecision, LoopStepResult, JobProgress,
    DirectiveProvenance, CandidateEvaluation,
    TrainingResult, TrainingOrigin, OperationCall, OperationResult, DecisionState, ExplorationState,
    RLPlan, RLState, ActorTrainingRequest, SFTSpec, RLSpec, EvaluationSpec, PromptSpec,
)}
ENUMS = {cls.__name__: cls for cls in (JobStatus, ExecutionStatus, ResumeMode)}

# Only these additive fields may be absent in older checkpoints. Keep
# rejecting unknown fields and missing pre-existing fields on every state type.
LINEAGE_DEFAULTS = {
    'SystemVersion': {'models'}, 'Candidate': {'model_bases'},
    'Decision': {'decision_id', 'depends_on_decision_ids', 'source_decision_ids',
                 'research_attempt_id', 'research_attempt_title', 'feedback_proposal_ids'},
    'DecisionContext': {'decision_graph', 'candidate_origins'},
    'DecisionState': {'decision_records', 'directive_decisions', 'candidate_origins',
                      'pending_operation_decision_id', 'pending_decision', 'decision_results', 'operation_receipts'},
    '_PendingOperationRecord': {'decision_id'},
    'OperationResult': {'adopted_candidate_ids', 'feedback_delivery'}, 'Feedback': {'adopted_candidate_ids'},
    'LoopStepResult': {'decision_id'},
}


def encode(value):
    if isinstance(value, Enum):
        if type(value).__name__ not in ENUMS:
            raise TypeError('Unsupported state enum')
        return {'$enum': type(value).__name__, 'value': value.value}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if is_dataclass(value) and type(value).__name__ in TYPES and TYPES[type(value).__name__] is type(value):
        return {'$type': type(value).__name__, 'fields': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {'$map': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, set, list)):
        tag = '$tuple' if isinstance(value, tuple) else '$set' if isinstance(value, set) else '$list'
        return {tag: [encode(x) for x in value]}
    raise TypeError(f'Unsupported checkpoint state type: {type(value).__name__}')


def decode(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if not isinstance(value, dict):
        raise ValueError('Invalid checkpoint value')
    if set(value) == {'$enum', 'value'} and value['$enum'] in ENUMS:
        return ENUMS[value['$enum']](value['value'])
    if set(value) == {'$type', 'fields'} and value['$type'] in TYPES:
        cls = TYPES[value['$type']]
        expected = {f.name for f in fields(cls)}
        if (not isinstance(value['fields'], dict) or set(value['fields']) - expected
                or expected - set(value['fields']) - LINEAGE_DEFAULTS.get(cls.__name__, set())):
            raise ValueError('Checkpoint fields do not match the current state schema')
        return cls(**{k: decode(v) for k, v in value['fields'].items()})
    if set(value) == {'$map'}:
        return {decode(k): decode(v) for k, v in value['$map']}
    for tag, factory in (('$tuple', tuple), ('$set', set), ('$list', list)):
        if set(value) == {tag}:
            return factory(decode(v) for v in value[tag])
    raise ValueError('Unsupported checkpoint type or encoding')
