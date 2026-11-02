from dataclasses import dataclass, field

from rsisys.experience.records import ExperienceRecord
from rsisys.loops._memory import Memory, HistoryEntry
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import ExecutionView, ReviewDecision, LoopStepResult
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef

from ..action_space import OperationCall, OperationResult, OperationSpec
from .decision import Decision
from .attempts import known_attempts, validate_attempt_fields


@dataclass(frozen=True)
class DirectiveProvenance:
    """Bindings observed at dispatch/accept; not inferred from a later training result."""
    directive_id: str
    decision_model: ModelRef | None = None
    execution_model: ModelRef | None = None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Latest receipt per candidate, including failures and different protocols.

    This is a bounded-per-candidate summary, not a complete evaluation archive;
    full older receipts remain in history and Experience.
    """
    candidate_id: str
    directive_id: str
    protocol_ref: str
    input_refs: tuple[ArtifactRef, ...]
    status: ExecutionStatus
    quality_metrics: dict[str, float]
    simulated: bool
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionContext:
    objective: str
    experience: tuple[ExperienceRecord, ...]
    operations: tuple[OperationSpec, ...]
    operation_results: tuple[OperationResult, ...]
    memories: tuple[Memory, ...] = ()
    last_feedback: Feedback | None = None
    history: tuple[HistoryEntry, ...] = ()
    directive_modes: tuple[str, ...] = ("execute",)
    candidates: tuple[Candidate, ...] = ()
    execution: ExecutionView = field(default_factory=ExecutionView)
    decision_model: ModelRef | None = None
    directive_provenance: tuple[DirectiveProvenance, ...] = ()
    candidate_evaluations: tuple[CandidateEvaluation, ...] = ()
    review_actions: tuple[str, ...] = ('continue', 'cancel')
    outstanding: tuple[Directive, ...] = ()
    executions: dict[str, ExecutionView] = field(default_factory=dict)
    run_control: dict[str, object] = field(default_factory=dict)
    context_summary: dict[str, object] = field(default_factory=dict)
    research_agenda: dict[str, object] = field(default_factory=dict)
    decision_graph: tuple[dict, ...] = ()
    candidate_origins: dict[str, str] = field(default_factory=dict)
    research_state: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingOperationRecord:
    record_id: str
    call: OperationCall | None
    result: OperationResult
    elapsed_seconds: float | None
    job: JobRef | None
    resume_result: OperationResult | None = None
    task_id: str | None = None
    event: str | None = None
    decision_id: str | None = None


@dataclass(frozen=True)
class _PendingReviewRecord:
    record_id: str
    objective: str
    directive_id: str
    decision: ReviewDecision
    details: dict[str, object]


@dataclass
class _ReviewSession:
    objective: str
    directive_id: str
    context: DecisionContext
    tool_limit: int
    queries_used: int = 0
    query_refs: list[str] = field(default_factory=list)
    simulated: bool = False
    pending_record: _PendingOperationRecord | None = None


@dataclass
class DecisionState:
    issued: int = 0
    completed: int = 0
    pending_directive_id: str | None = None
    pending_directive: Directive | None = None
    feedback_record_id: str | None = None
    last_directive: Directive | None = None
    issued_directive_ids: set[str] = field(default_factory=set)
    last_feedback: Feedback | None = None
    operation_results: list[OperationResult] = field(default_factory=list)
    pending_job: JobRef | None = None
    pending_operation: OperationCall | None = None
    pending_simulated: bool = False
    pending_operation_record: _PendingOperationRecord | None = None
    finished: bool = False
    recorded_training: set[JobRef] = field(default_factory=set)
    applied_training: set[JobRef] = field(default_factory=set)
    training_record_ids: dict[JobRef, str] = field(default_factory=dict)
    measured_input_tokens: int = 0
    measured_output_tokens: int = 0
    calls_with_missing_usage: int = 0
    candidates: dict[str, Candidate] = field(default_factory=dict)
    execution: ExecutionView = field(default_factory=ExecutionView)
    directive_provenance: dict[str, DirectiveProvenance] = field(default_factory=dict)
    candidate_evaluations: dict[str, CandidateEvaluation] = field(default_factory=dict)
    pending_review_record: _PendingReviewRecord | None = None
    review_session: _ReviewSession | None = None
    outstanding: dict[str, Directive] = field(default_factory=dict)
    feedback_record_ids: dict[str, str] = field(default_factory=dict)
    delivered_feedback: dict[str, tuple[Directive, Feedback]] = field(default_factory=dict)
    executions: dict[str, ExecutionView] = field(default_factory=dict)
    candidate_bases: dict[str, ModelRef] = field(default_factory=dict)
    review_sessions: dict[str, _ReviewSession] = field(default_factory=dict)
    review_records: dict[str, _PendingReviewRecord] = field(default_factory=dict)
    run_control: dict[str, object] = field(default_factory=dict)
    research_cycles: dict[int, dict[str, object]] = field(default_factory=dict)
    decision_records: dict[str, dict] = field(default_factory=dict)
    directive_decisions: dict[str, str] = field(default_factory=dict)
    candidate_origins: dict[str, str] = field(default_factory=dict)
    pending_operation_decision_id: str | None = None
    pending_decision: Decision | None = None
    decision_results: dict[str, LoopStepResult] = field(default_factory=dict)
    operation_receipts: dict[tuple[JobRef, str | None], OperationResult] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.decision_records, dict):
            raise ValueError('Decision records must be indexed by ID')
        attempt_origins = set()
        for ident, record in self.decision_records.items():
            if (not isinstance(ident, str) or not ident.strip() or not isinstance(record, dict)
                    or record.get('decision_id') != ident
                    or record.get('status') not in ('pending', 'executing', 'completed', 'failed', 'cancelled', 'partial', 'blocked')):
                raise ValueError('Invalid decision ledger record')
            attempt_id, title = record.get('research_attempt_id'), record.get('research_attempt_title', '')
            validate_attempt_fields(attempt_id, title)
            if (attempt_id is None) != (title == '') or (record.get('validation_error') and attempt_id is not None):
                raise ValueError('Resolved research attempt membership requires a validated ID/title pair')
            if attempt_id is not None:
                proposal = record.get('proposal', {})
                requested_id, requested_title = proposal.get('research_attempt_id'), proposal.get('research_attempt_title', '')
                if ((requested_id is not None and requested_id != attempt_id)
                        or (requested_title and requested_title != title)):
                    raise ValueError('Research attempt membership differs from its original proposal')
                if requested_id is None:
                    if requested_title != title or attempt_id in attempt_origins:
                        raise ValueError('Research attempt requires one explicit title-only origin')
                    attempt_origins.add(attempt_id)
            for name in ('depends_on_decision_ids', 'source_decision_ids', 'adopted_candidate_ids', 'adopted_decision_ids',
                         'unresolved_adopted_candidate_ids'):
                refs = record.get(name, ())
                if (not isinstance(refs, tuple) or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                        or len(set(refs)) != len(refs)):
                    raise ValueError(f'Invalid ledger {name}')
            if record.get('adopted_candidate_ids') and record['status'] != 'completed':
                raise ValueError('Only completed decisions may confirm adoption')
        if set(known_attempts(self.decision_records.values())) != attempt_origins:
            raise ValueError('Research attempt must have a validated ledger origin')
        for mapping in (self.directive_decisions, self.candidate_origins):
            if not isinstance(mapping, dict) or any(not isinstance(k, str) or not k.strip()
                    or not isinstance(v, str) or v not in self.decision_records for k, v in mapping.items()):
                raise ValueError('Lineage indexes must reference known decisions')
        if any(self.decision_records[v]['status'] != 'completed' for v in self.candidate_origins.values()):
            raise ValueError('Candidate origins require completed producer decisions')
        for record in self.decision_records.values():
            if not record.get('validation_error'):
                for name in ('depends_on_decision_ids', 'source_decision_ids'):
                    if any(ref == record['decision_id'] or ref not in self.decision_records
                           for ref in record.get(name, ())):
                        raise ValueError('Confirmed lineage requires known non-self references')
                if any(self.decision_records[ref]['status'] != 'completed'
                       for ref in record.get('depends_on_decision_ids', ())):
                    raise ValueError('Confirmed dependencies require completed decisions')
            adopted = record.get('adopted_candidate_ids', ())
            unresolved = record.get('unresolved_adopted_candidate_ids', ())
            sources = tuple(dict.fromkeys(self.candidate_origins.get(ref) for ref in adopted if ref not in unresolved))
            if (not set(unresolved) <= set(adopted) or None in sources or record['decision_id'] in sources
                    or sources != record.get('adopted_decision_ids', ())):
                raise ValueError('Adoption must match known producer origins')
            edges = record.get('confirmed_edges', ())
            if not isinstance(edges, tuple) or any(not isinstance(edge, dict)
                    or set(edge) != {'from', 'to', 'kind', 'confirmed'} or edge['confirmed'] is not True
                    or edge['to'] != record['decision_id'] or edge['from'] not in self.decision_records
                    or edge['from'] == edge['to'] or edge['kind'] not in ('depends_on', 'source', 'adopted') for edge in edges):
                raise ValueError('Invalid confirmed decision edges')
            expected = tuple({'from': ref, 'to': record['decision_id'], 'kind': kind, 'confirmed': True}
                for kind, refs in (('depends_on', record.get('depends_on_decision_ids', ())),
                                   ('source', record.get('source_decision_ids', ())), ('adopted', sources))
                for ref in refs) if not record.get('validation_error') else ()
            if edges != expected:
                raise ValueError('Confirmed edges differ from validated lineage')
        if self.pending_operation_decision_id is not None and self.pending_operation_decision_id not in self.decision_records:
            raise ValueError('Pending operation must reference a known decision')
        if self.pending_decision is not None and (not isinstance(self.pending_decision, Decision)
                or self.pending_decision.decision_id not in self.decision_records):
            raise ValueError('Pending decision must retain its registered identity')
        if not isinstance(self.decision_results, dict) or any(k not in self.decision_records
                or not isinstance(v, LoopStepResult) or v.decision_id != k for k, v in self.decision_results.items()):
            raise ValueError('Decision results must reference their ledger identity')
        if not isinstance(self.operation_receipts, dict) or any(not isinstance(k, tuple) or len(k) != 2
                or not isinstance(k[0], JobRef) or not isinstance(v, OperationResult)
                or (k[1] is not None and k[1] not in self.decision_records) for k, v in self.operation_receipts.items()):
            raise ValueError('Invalid operation completion receipts')
