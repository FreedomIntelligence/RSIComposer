"""Default-ledger adapter for the independent feedback learning service.

Default and DSH harnesses retain their own policies, scheduling and compaction.
Other harnesses supply FeedbackAdapter rather than inheriting this implementation.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.json_tools import EvidencePathError

from ..feedback import FeedbackAdapter, FeedbackEvidence, FeedbackLearningService
from ..feedback import WorkingMemoryPolicy
from ..action_space import OperationCall, OperationResult, OperationSpec
from .context import result_id
from .engine import DefaultDecisionHarness
from .feedback_learning import FeedbackLearning, SerializedLearningBackend
from .research_state import research_working_view


# Compatibility for callers that used the old class's memory conversion helper.
# Installation no longer constructs a specialized harness subclass.
FeedbackLearningHarness = FeedbackLearningService


class DefaultFeedbackAdapter:
    """Translate the built-in execution ledger; the learning service is agnostic."""
    def __init__(self, harness, model=None, working_memory_policy=None):
        if model is not None and model != harness.policy.model:
            raise ValueError('Feedback must use the active meta-agent model')
        self.harness = harness
        self.working_memory_policy = (WorkingMemoryPolicy() if working_memory_policy is None
                                      else working_memory_policy)
        if not isinstance(self.working_memory_policy, WorkingMemoryPolicy):
            raise TypeError('working_memory_policy must be a WorkingMemoryPolicy')
        self.service = None

    def __getattr__(self, name):
        return getattr(self.harness, name)

    def model(self):
        # Resolve for each new capture; an install-time argument must not pin
        # this role to old weights after the meta-agent changes model version.
        return self.harness.policy.model

    def evidence(self, directive, feedback):
        return FeedbackEvidence(self.memory.history(), tuple(self.state.candidate_evaluations.values()),
            self._learning_scope(directive.directive_id), self._grouping_scope(directive.directive_id))

    def publish(self, memory):
        self.memory.remember(memory)

    def decision_ids(self):
        return tuple(self.state.decision_records)

    def proposal_decisions(self):
        return tuple({key: deepcopy(row[key]) for key in
            ('decision_id', 'status', 'kind', 'operation', 'directive_id', 'error', 'feedback_proposal_ids')
            if key in row} for row in self.state.decision_records.values() if row.get('feedback_proposal_ids'))

    def install(self, service):
        self.service = service
        self.harness.add_feedback_listener(service.receive_feedback)
        self.harness.add_context_processor(self.project_context)
        self.harness.add_cleanup(service.close)
        self.harness.add_receipt_listener(self._receive_operation)
        self.harness.feedback_proposal_resolver = service.resolve_harness_proposals
        # Read-only compatibility/navigation used by monitor and existing callers.
        self.harness.feedback_learning = service.learning
        self.harness.feedback_service = service
        self.harness.attempt_interpretations = service.attempts
        self.harness.last_working_view = None

    def _receive_operation(self, receipt):
        # Denied operations are absent from the authorized catalog, so their
        # read_only flag may be unknown. Our own navigation must never create
        # feedback about reading feedback, including denied reads.
        if not receipt.read_only and receipt.request.get('name') != READ_SPEC.name:
            return self.service.observe(receipt, evidence=self.receipt_evidence(receipt))

    def receipt_evidence(self, receipt):
        return FeedbackEvidence(self.memory.history(), tuple(self.state.candidate_evaluations.values()),
            self._learning_scope(None, decision_id=receipt.decision_id),
            self._grouping_scope(None, decision_id=receipt.decision_id))

    def _grouping_scope(self, directive_id, *, decision_id=None):
        """Bounded evidence window, independent of canonical execution membership."""
        records = self.state.decision_records
        ident = decision_id or self.state.directive_decisions.get(directive_id)
        selected = list(records)[-24:]
        if ident in records and ident not in selected:
            selected.append(ident)
        # Keep producer and declared dependency links, including older decisions.
        for key in list(selected):
            record = records[key]
            sources = list(record.get('source_decision_ids', ())) + list(record.get('depends_on_decision_ids', ()))
            proposal = record['proposal']
            directive = proposal.get('directive') or {}
            payload = directive.get('payload') or {}
            candidates = [payload.get('candidate_id')] if isinstance(payload, dict) else []
            if isinstance(payload, dict) and isinstance(payload.get('candidate'), dict):
                candidates.append(payload['candidate'].get('candidate_id'))
            for ref in directive.get('input_refs', ()):
                candidates.append(ref.get('artifact_id'))
            for candidate in candidates:
                origin = self.state.candidate_origins.get(candidate)
                if isinstance(origin, str):
                    sources.append(origin)
            for source in sources:
                if source in records and source not in selected and len(selected) < 32:
                    selected.append(source)
        graph = {r['decision_id']: r for r in self._decision_graph()}
        decisions = []
        for key in selected:
            record = records[key]
            proposal = record['proposal']
            action = {k: deepcopy(proposal[k]) for k in ('kind', 'operation', 'simulated') if k in proposal}
            if proposal.get('directive'):
                action['directive'] = {k: deepcopy(proposal['directive'][k])
                    for k in ('mode', 'strategy', 'payload', 'input_refs')}
            decisions.append({**graph[key], 'kind': record['kind'], 'reason': record.get('reason', ''),
                'proposal': action, 'error': record.get('error'),
                'directive_id': record.get('directive_id'),
                'decision_model': deepcopy(record.get('decision_model'))})
        catalog = self.attempt_interpretations()['items']
        attempts = [r for r in catalog if set(r['member_decision_ids']).intersection(selected) or r in catalog[-8:]]
        directives = {records[key].get('directive_id') for key in selected} - {None}
        observations = [asdict(h) for h in self.memory.history()
                        if h.task_id in directives or h.record_id in selected
                        or h.details.get('decision_id') in selected]
        return {'decisions': decisions, 'attempts': attempts, 'observations': observations[-16:],
                'total_observations': len(observations),
                'candidate_origins': {key: value for key, value in self.state.candidate_origins.items()
                                      if value in selected},
                'scope': 'bounded_observed_decisions', 'total_decisions': len(records),
                'truncated': len(decisions) < len(records), 'receipt_decision_id': ident}

    def attempt_interpretations(self):
        return self.service.attempts()

    def _learning_scope(self, directive_id, *, decision_id=None):
        """Snapshot canonical membership; never infer groups from a proposal/title."""
        ident = decision_id or self.state.directive_decisions.get(directive_id)
        origin = self.state.decision_records.get(ident, {})
        attempt = origin.get('research_attempt_id')
        members = []
        for item in self._decision_graph():
            if (item['decision_id'] != ident and
                    (attempt is None or item.get('research_attempt_id') != attempt)):
                continue
            record = self.state.decision_records[item['decision_id']]
            members.append({**item, 'directive_id': record.get('directive_id'),
                            'kind': record['kind'], 'reason': record.get('reason', '')})
        scope = dict(decision_id=ident, research_attempt_id=attempt,
                     research_attempt_title=origin.get('research_attempt_title', ''),
                     member_decisions=members, scope='observed_members_at_feedback',
                     attempt_completion='not_reported')
        scope['feedback_proposal_ids'] = list(origin.get('feedback_proposal_ids', ()))
        provenance = self.state.directive_provenance.get(directive_id)
        scope['issuing_decision_model'] = (asdict(provenance.decision_model)
            if provenance is not None and provenance.decision_model is not None else origin.get('decision_model'))
        scope['learning_decision_model'] = asdict(self.model())
        return {**scope, 'snapshot_digest': result_id(scope)}

    def project_context(self, context):
        self.service.refresh()
        context = replace(context, memories=self.memory.recall(limit=self.memory_limit))
        jobs = self.service.learning.ordered_jobs()
        navigation = {'read_operation': 'feedback_learning_read',
            'authority': 'unverified_proposals', 'total': len(jobs),
            'recent': [{'job_id': row['job_id'], 'status': row['status'],
                        'research_attempt_id': row['input'].get('research_context', {}).get('research_attempt_id'),
                        'kinds': [p['kind'] for p in row['proposals']],
                        'products': deepcopy(row.get('products', {}))} for row in jobs[-6:]]}
        proposals = self.service.proposals()
        navigation['harness_proposals'] = {'total': len(proposals),
            'read_path': {'operation': 'feedback_learning_read', 'view': 'proposals', 'target': 'harness'},
            'construction_owner': 'main_meta_agent_decision',
            'recent': [{'proposal_id': p['proposal_id'], 'subject': p['subject'],
                'content_preview': p['proposal']['content'][:200], 'read_path': p['read_path'],
                'decision_links': p['decision_links'][-4:], 'applied': False} for p in proposals[-6:]]}
        context = replace(context, research_agenda={**context.research_agenda, 'feedback_learning': navigation})
        active = {r['decision_id'] for r in context.decision_graph
                  if r['status'] not in ('completed', 'failed', 'cancelled')}
        knowledge = self.service.working_memory(active_decision_ids=active,
                                                policy=self.working_memory_policy)
        view = research_working_view(context, jobs, self.attempt_interpretations()['items'], knowledge=knowledge)
        self.harness.last_working_view = deepcopy({'research_state': view.research_state,
            'projection': view.research_agenda['working_projection'],
            'scope': 'last assembled view before token-budget compaction',
            'history_records_before': len(context.history), 'history_records_after': len(view.history),
            'memories_before': len(context.memories), 'memories_after': len(view.memories)})
        return view


READ_SPEC = OperationSpec('feedback_learning_read',
    'Read feedback jobs or view=attempts/proposals/trajectories. Trajectories archive captured public '
    'decisions, execution and feedback; they are not training-ready datasets. '
    'Context/memory organization is active via a replaceable provider; read /context_memory. '
    'Trajectory archival defaults on but may be disabled/replaced; check trajectory.enabled before /trajectory. '
    'Read /trajectory_segment to extract captured public material without generating training targets. '
    'Harness proposals can be referenced by feedback_proposal_ids in a later ordinary decision; '
    'feedback does not construct or activate them. '
    'Proposal targets are context_memory, harness, weights '
    '(legacy system changes stay separate). Attempt views return latest revisions; proposals stay unverified. '
    'These are unverified proposals, never applied changes or training-ready examples. '
    'Omit job_id to list jobs; use next_offset for the next page. '
    'With job_id and evidence_path (JSON pointer, empty for root), read original evidence '
    'in JSON-text pages; offset counts characters and each limit unit is 128 characters. '
    '/proposals and /attempt_revisions address interpretation output, not captured evidence. '
    '/feedback_request and /feedback_response read the recorded feedback model exchange when available. '
    'Knowledge views also return JSON-text pages; pass sha256 as expected_digest on subsequent pages.',
    {'type': 'object', 'properties': {
        'view': {'enum': ['jobs', 'attempts', 'proposals', 'trajectories']},
        'target': {'enum': ['context_memory', 'harness', 'weights', 'system']},
        'expected_digest': {'type': 'string', 'minLength': 64, 'maxLength': 64},
        'job_id': {'type': 'string', 'minLength': 1},
        'evidence_path': {'type': 'string', 'maxLength': 1000,
                          'description': 'JSON pointer into original evidence; requires job_id.'},
        'research_attempt_id': {'type': 'string', 'minLength': 1, 'maxLength': 200,
                                'description': 'Filter job listings by a recorded attempt ID.'},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
     'additionalProperties': False}, read_only=True)


def _read_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) - {'job_id', 'research_attempt_id', 'offset', 'limit',
                                                          'evidence_path', 'view', 'target', 'expected_digest'}:
        raise ValueError('Invalid feedback_learning_read arguments')
    view = arguments.get('view', 'jobs')
    if view not in ('jobs', 'attempts', 'proposals', 'trajectories'):
        raise ValueError('Invalid feedback view')
    if view != 'jobs' and ('job_id' in arguments or 'evidence_path' in arguments):
        raise ValueError('Knowledge view cannot be combined with job evidence')
    if 'target' in arguments and (view != 'proposals' or arguments['target'] not in
                                ('context_memory', 'harness', 'weights', 'system')):
        raise ValueError('target requires a proposals view')
    if 'expected_digest' in arguments and (not isinstance(arguments['expected_digest'], str)
            or len(arguments['expected_digest']) != 64):
        raise ValueError('Invalid expected_digest')
    if 'expected_digest' in arguments and view == 'jobs' and 'evidence_path' not in arguments:
        raise ValueError('expected_digest requires an evidence or knowledge page')
    if 'job_id' in arguments and (not isinstance(arguments['job_id'], str) or not arguments['job_id'].strip()):
        raise ValueError('job_id must be a nonempty string')
    if 'evidence_path' in arguments:
        pointer = arguments['evidence_path']
        if ('job_id' not in arguments or not isinstance(pointer, str) or len(pointer) > 1000
                or (pointer and not pointer.startswith('/'))):
            raise ValueError('evidence_path requires job_id and a bounded JSON pointer')
    if 'research_attempt_id' in arguments:
        ident = arguments['research_attempt_id']
        if not isinstance(ident, str) or not ident.strip() or len(ident) > 200 or 'job_id' in arguments:
            raise ValueError('research_attempt_id must be a bounded ID for a job listing')
    if type(arguments.get('offset', 0)) is not int or arguments.get('offset', 0) < 0:
        raise ValueError('offset must be nonnegative')
    if type(arguments.get('limit', 8)) is not int or not 1 <= arguments.get('limit', 8) <= 16:
        raise ValueError('limit must be in 1..16')
    return deepcopy(arguments)


class LearningActionSpace:
    def __init__(self, base):
        if any(spec.name == READ_SPEC.name for spec in base.describe()):
            raise ValueError('feedback_learning_read is already installed')
        self.base = base

    @property
    def permissions(self):
        # Authorization intentionally ignores attributes fabricated by __getattr__.
        return self.base.permissions

    def __getattr__(self, name):
        return getattr(self.base, name)

    def describe(self):
        return (*self.base.describe(), deepcopy(READ_SPEC))

    def validate(self, call):
        if call.name == READ_SPEC.name:
            return OperationCall(call.name, _read_arguments(call.arguments))
        return self.base.validate(call)


class LearningTools:
    def __init__(self, base, learning):
        self.base, self.learning = base, learning

    def __getattr__(self, name):
        return getattr(self.base, name)

    def execute(self, call):
        if call.name != READ_SPEC.name:
            return self.base.execute(call)
        try:
            data = self.learning.read(**_read_arguments(call.arguments))
        except EvidencePathError as exc:
            return OperationResult('Feedback read rejected', (), False,
                ExecutionStatus.FAILED, str(exc), data={
                    'view': call.arguments.get('view', 'jobs'),
                    'job_id': call.arguments.get('job_id'),
                    'evidence_path': call.arguments.get('evidence_path'),
                    'read_operation': READ_SPEC.name})
        return OperationResult('Feedback learning proposals; validation still required', (),
                               bool(data.get('simulated', False)), data=data)


def feedback_learning_decision_factory(*, decision_factory=None, backend=None, model=None,
                                       adapter_factory=None, **learning_options):
    """Compose feedback with any decision factory and explicit adapter/model backend."""
    def build(context, exploration):
        from rsisys.assembly import build_default_decision
        loop = (decision_factory or build_default_decision)(context, exploration)
        path = context.data_dir / 'feedback_learning.json' if context.data_dir is not None else None
        adapter = adapter_factory(loop.harness) if adapter_factory else None
        enable_feedback_learning(loop, path=path, backend=backend, model=model,
                                 adapter=adapter, **learning_options)
        context.add_cleanup(loop.feedback_service.close)
        return loop
    return build


def enable_feedback_learning(base_loop, *, path=None, backend=None, model=None, adapter=None,
                             working_memory_policy=None, **options):
    """Attach once before execution, retaining harness and compactor identities.

    An explicit backend may differ from the decision backend. It must enforce its
    host's shared admission/budget when that deployment has a global call limit.
    Custom harnesses supply a FeedbackAdapter with lifecycle/context integration.
    """
    from .policy import ModelDecisionPolicy
    base = base_loop.harness
    if getattr(base_loop, 'feedback_service', None) is not None:
        raise ValueError('Feedback learning is already installed')
    if adapter is None:
        if not isinstance(base, DefaultDecisionHarness):
            raise TypeError('Custom decision harness requires an explicit FeedbackAdapter')
        from .state import DecisionState
        pristine = DecisionState()
        pristine.operation_results = list(base.state.operation_results)
        if base.state != pristine:
            raise ValueError('Install feedback learning before the first decision')
        adapter = DefaultFeedbackAdapter(base, model=model, working_memory_policy=working_memory_policy)
    elif working_memory_policy is not None:
        raise ValueError('Custom adapters select their own working memory policy')
    for method in ('model', 'evidence', 'publish', 'decision_ids', 'install'):
        if not callable(getattr(adapter, method, None)):
            raise TypeError('FeedbackAdapter requires ' + method)
    policy = getattr(base, 'policy', None)
    if backend is None or (isinstance(policy, ModelDecisionPolicy) and backend is policy.backend):
        # Shared stateless backend is a convenience of this adapter, not a service requirement.
        if not isinstance(policy, ModelDecisionPolicy):
            raise ValueError('Session/custom policies require an explicit feedback backend')
        shared = (policy.backend if isinstance(policy.backend, SerializedLearningBackend)
                  else SerializedLearningBackend(policy.backend))
        backend = shared
        policy.backend = shared
    backend = backend if isinstance(backend, SerializedLearningBackend) else SerializedLearningBackend(backend)
    learning = FeedbackLearning(backend, path=path, **options)
    service = FeedbackLearningService(learning, adapter)
    if isinstance(adapter, DefaultFeedbackAdapter):
        space = LearningActionSpace(base_loop.action_space)
        base.tools = LearningTools(base.tools, learning)
        parser = getattr(base.policy, 'parser', base.policy)
        if hasattr(parser, 'space'):
            parser.space = space
        base_loop.action_space = space
    adapter.install(service)
    base_loop.feedback_service = service
    return base
