"""Explicit JSON codec for known local state types. Never loads Python objects by name."""

from dataclasses import fields, is_dataclass
from enum import Enum

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.coordination import ExecutionView, ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef, TokenUsage
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.contracts.requests import SFTSpec, RLSpec, EvaluationSpec, PromptSpec
from rsisys.loops.decision_making.action_space import OperationCall, OperationResult
from rsisys.loops.decision_making.harness.state import (
    DecisionState, DirectiveProvenance, CandidateEvaluation,
)
from rsisys.loops.exploration.harness.state import ExplorationState
from rsisys.loops.exploration.harness.rl import RLPlan, RLState
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest

TYPES = {cls.__name__: cls for cls in (
    ArtifactRef, Candidate, Directive, Feedback, JobRef, ModelRef, TokenUsage,
    ExecutionView, ExplorationProgress, ReviewDecision,
    DirectiveProvenance, CandidateEvaluation,
    TrainingResult, OperationCall, OperationResult, DecisionState, ExplorationState,
    RLPlan, RLState, ActorTrainingRequest, SFTSpec, RLSpec, EvaluationSpec, PromptSpec,
)}
ENUMS = {cls.__name__: cls for cls in (JobStatus, ExecutionStatus)}


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
        if set(value['fields']) != {f.name for f in fields(cls)}:
            raise ValueError('Checkpoint fields do not match the current state schema')
        return cls(**{k: decode(v) for k, v in value['fields'].items()})
    if set(value) == {'$map'}:
        return {decode(k): decode(v) for k, v in value['$map']}
    for tag, factory in (('$tuple', tuple), ('$set', set), ('$list', list)):
        if set(value) == {tag}:
            return factory(decode(v) for v in value[tag])
    raise ValueError('Unsupported checkpoint type or encoding')
