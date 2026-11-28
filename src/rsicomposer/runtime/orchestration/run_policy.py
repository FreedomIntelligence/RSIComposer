"""Replaceable run-level control; research strategy stays with the decision policy."""
from dataclasses import dataclass
import math
from typing import Callable, Literal, Protocol

from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.evaluation import candidate_id, canonical_refs, matches_cohort
from rsicomposer.loops.contracts.metrics import has_scored_metrics


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class ExperimentOutcome:
    receipt: OperationResult


@dataclass(frozen=True)
class EvaluationOutcome(ExperimentOutcome):
    """A delivered receipt tied to a Host-classified evaluation call."""
    operation: str
    candidate_id: str
    protocol_ref: str
    input_refs: tuple[str, ...]

    @property
    def measured(self):
        return has_scored_metrics(self.receipt.status, self.receipt.data)

    @classmethod
    def from_call(cls, call, receipt, *, candidate_fields=()):
        data = receipt.data
        candidate = data.get('candidate_id')
        protocol = data.get('protocol_ref')
        inputs = canonical_refs(data.get('input_refs'))
        args = call.arguments
        if (not isinstance(candidate, str) or not candidate.strip()
                or not isinstance(protocol, str) or not protocol.strip() or inputs is None):
            return None
        if (candidate_id(args, candidate_fields) != candidate or
                ('protocol_ref' in args and args['protocol_ref'] != protocol)):
            return None
        if not matches_cohort(args, data, inputs):
            return None
        return cls(receipt, call.name, candidate, protocol, inputs)


@dataclass(frozen=True)
class RunContext:
    objective: str
    elapsed_seconds: float
    research_round: int
    researcher_finished: bool
    outcomes: tuple[ExperimentOutcome, ...] = ()


@dataclass(frozen=True)
class RunDecision:
    action: Literal['continue', 'restart', 'stop'] = 'continue'
    reason: str = ''
    shutdown: Literal['drain', 'cancel'] = 'drain'
    restart_delay_seconds: float = 1.0
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if self.action not in ('continue', 'restart', 'stop'):
            raise ValueError('Unknown run action')
        if self.shutdown not in ('drain', 'cancel'):
            raise ValueError('Unknown shutdown mode')
        positive(self.restart_delay_seconds, 'restart_delay_seconds')
        if not isinstance(self.reason, str):
            raise TypeError('Run decision reason must be a string')
        if not isinstance(self.evidence_refs, tuple) or any(not isinstance(ref, str) for ref in self.evidence_refs):
            raise TypeError('evidence_refs must be a tuple of strings')
        if self.action == 'stop' and not self.reason:
            raise ValueError('Stopping requires a reason')


class RunPolicy(Protocol):
    """Bounded, side-effect-free checks; a tick may evaluate more than once."""
    def evaluate(self, context: RunContext) -> RunDecision: ...


@dataclass(frozen=True)
class ResearcherRunPolicy:
    """Default: a researcher finish requests run shutdown, without claiming success."""
    shutdown: Literal['drain', 'cancel'] = 'drain'

    def __post_init__(self):
        RunDecision(shutdown=self.shutdown)

    def evaluate(self, context):
        return RunDecision('stop', 'researcher_finished', self.shutdown) if context.researcher_finished else RunDecision()


@dataclass(frozen=True)
class TimeRunPolicy:
    seconds: float
    restart_delay_seconds: float = 1.0
    shutdown: Literal['drain', 'cancel'] = 'cancel'

    def __post_init__(self):
        positive(self.seconds, 'seconds')
        RunDecision(shutdown=self.shutdown, restart_delay_seconds=self.restart_delay_seconds)

    def evaluate(self, context):
        if context.elapsed_seconds >= self.seconds:
            return RunDecision('stop', 'budget_exhausted', self.shutdown)
        return RunDecision('restart' if context.researcher_finished else 'continue',
                           'time_remaining', restart_delay_seconds=self.restart_delay_seconds)


@dataclass(frozen=True)
class GoalRunPolicy:
    goal: Callable[[RunContext], bool]
    restart_delay_seconds: float = 1.0
    max_seconds: float | None = None
    shutdown: Literal['drain', 'cancel'] = 'drain'

    def __post_init__(self):
        if not callable(self.goal):
            raise TypeError('goal must be callable')
        if self.max_seconds is not None:
            positive(self.max_seconds, 'max_seconds')
        RunDecision(shutdown=self.shutdown, restart_delay_seconds=self.restart_delay_seconds)

    def evaluate(self, context):
        # A hard budget wins if the goal and deadline become visible together.
        if self.max_seconds is not None and context.elapsed_seconds >= self.max_seconds:
            return RunDecision('stop', 'budget_exhausted', 'cancel')
        achieved = self.goal(context)
        if type(achieved) is not bool:
            raise TypeError('goal must return bool')
        if achieved:
            return RunDecision('stop', 'goal_reached', self.shutdown)
        return RunDecision('restart' if context.researcher_finished else 'continue',
                           'goal_not_reached', restart_delay_seconds=self.restart_delay_seconds)


@dataclass(frozen=True)
class MetricGoal:
    """A completed evaluation threshold, not automatic candidate application.

    Only outcomes tied to a declared evaluation operation count. A result must
    carry a candidate, versioned protocol reference and exact input references.
    Prefer distinct protocol references per validation dataset, or set input_refs
    to select the precise inputs for this goal.
    """
    metric: str
    threshold: float
    protocol_ref: str
    higher_is_better: bool = True
    allow_simulated: bool = False
    input_refs: tuple[str, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.metric, str) or not self.metric or not isinstance(self.protocol_ref, str) or not self.protocol_ref:
            raise ValueError('metric and protocol_ref must be nonempty strings')
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold):
            raise ValueError('threshold must be finite')
        if type(self.higher_is_better) is not bool or type(self.allow_simulated) is not bool:
            raise TypeError('MetricGoal flags must be boolean')
        if self.input_refs is not None:
            refs = canonical_refs(self.input_refs)
            if refs is None:
                raise ValueError('MetricGoal input_refs must be distinct nonempty references')
            object.__setattr__(self, 'input_refs', refs)

    def __call__(self, context):
        seen = set()
        for outcome in reversed(context.outcomes):
            if not isinstance(outcome, EvaluationOutcome) or outcome.protocol_ref != self.protocol_ref:
                continue
            if self.input_refs is not None and outcome.input_refs != self.input_refs:
                continue
            key = (outcome.operation, outcome.candidate_id, outcome.input_refs)
            if key in seen:
                continue
            seen.add(key)
            receipt = outcome.receipt
            if not outcome.measured or (receipt.simulated and not self.allow_simulated):
                continue
            value = receipt.data.get('quality_metrics', {}).get(self.metric)
            if type(value) not in (int, float) or not math.isfinite(value):
                continue
            if (value >= self.threshold if self.higher_is_better else value <= self.threshold):
                return True
        return False
