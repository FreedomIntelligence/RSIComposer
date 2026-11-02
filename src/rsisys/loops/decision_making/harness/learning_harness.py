"""Opt-in learning from completed experiments with receipt-backed memory views."""
from copy import deepcopy
from dataclasses import asdict, replace

from rsisys.loops._memory import Memory
from ..action_space import OperationCall, OperationResult, OperationSpec
from .context import encoded, result_id
from .engine import DefaultDecisionHarness
from .feedback_learning import FeedbackLearning, SerializedLearningBackend
from .learning_policy import MemoryPrefixDecisionPolicy


class FeedbackLearningHarness(DefaultDecisionHarness):
    def __init__(self, *args, feedback_learning: FeedbackLearning, **kwargs):
        super().__init__(*args, **kwargs)
        if (not isinstance(self.policy, MemoryPrefixDecisionPolicy)
                or self.policy.backend is not feedback_learning.backend):
            raise ValueError('Decision and learning must share the serialized backend and memory-prefix policy')
        self.feedback_learning = feedback_learning

    def receive_feedback(self, directive, feedback):
        # Existing validation, idempotence, ledger and delivery acknowledgement
        # semantics remain authoritative. Learning is never the terminal receipt.
        super().receive_feedback(directive, feedback)
        self.feedback_learning.capture(directive, feedback, self.policy.model,
            self.memory.history(),
            evaluations=tuple(self.state.candidate_evaluations.values()),
            research_context=self._learning_scope(directive.directive_id))
        self.feedback_learning.poll()

    def _learning_scope(self, directive_id):
        """Snapshot canonical membership; never infer groups from a proposal/title."""
        ident = self.state.directive_decisions.get(directive_id)
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
        provenance = self.state.directive_provenance.get(directive_id)
        scope['issuing_decision_model'] = (asdict(provenance.decision_model)
            if provenance is not None and provenance.decision_model is not None else None)
        scope['learning_decision_model'] = asdict(self.policy.model)
        return {**scope, 'snapshot_digest': result_id(scope)}

    @staticmethod
    def _memory(row):
        proposals = [p for p in row['proposals'] if p['kind'] == 'memory']
        if not proposals:
            return None
        receipt = row['input']['receipt']
        feedback, directive = receipt['feedback'], receipt['directive']
        # These fields are copied from receipts, not rewritten by the model.
        facts = {key: feedback[key] for key in ('directive_id', 'status', 'quality_metrics',
            'training_metrics', 'execution_metrics', 'error', 'evidence_refs', 'adopted_candidate_ids')}
        facts.update(mode=directive['mode'], candidate_ids=[c['candidate_id'] for c in feedback['candidates']])
        return Memory('feedback-learning:' + row['job_id'], 'episodic',
            encoded({'receipt_facts': facts, 'interpretations': proposals, 'learned_by': row['model'],
                     'research_context': row['input'].get('research_context', {}),
                     'simulated': row['simulated'], 'authority': 'unverified_interpretations',
                     'read_operation': 'feedback_learning_read', 'job_id': row['job_id']}),
            tuple(row['input']['source_refs']),
            applicability=encoded({'objective': directive['objective'], 'experiment': directive['directive_id'],
                                   'scope': 'This experiment and its original model/protocol; not current global state'}),
            status='unverified')

    def _publish(self):
        for ident, row in list(self.feedback_learning.state['jobs'].items()):
            if row['status'] != 'completed' or row.get('published'):
                continue
            memory = self._memory(row)
            if memory is not None:
                self.memory.remember(memory)
            state = deepcopy(self.feedback_learning.state)
            state['jobs'][ident].update(published=True,
                memory_digest=result_id(asdict(memory)) if memory is not None else None)
            # If persistence fails, retry publishing the same deterministic ID.
            self.feedback_learning._commit(state)

    def _context(self, objective, action_space=None):
        self.feedback_learning.poll()
        self._publish()
        context = super()._context(objective, action_space)
        # Input coverage proves what the model saw, not what its prose retained.
        # Unverified memory must not suppress contradictory/uncited receipts. The
        # normal budget compactor owns archival, with source navigation and the
        # latest feedback/history protection; learning alone never deletes a view.
        jobs = list(self.feedback_learning.state['jobs'].values())
        navigation = {'read_operation': 'feedback_learning_read',
            'authority': 'unverified_proposals', 'total': len(jobs),
            'recent': [{'job_id': row['job_id'], 'status': row['status'],
                        'research_attempt_id': row['input'].get('research_context', {}).get('research_attempt_id'),
                        'kinds': [p['kind'] for p in row['proposals']]} for row in jobs[-6:]]}
        return replace(context, research_agenda={**context.research_agenda, 'feedback_learning': navigation})

    def close(self, *, timeout=30):
        self.feedback_learning.close(timeout=timeout)
        self._publish()


READ_SPEC = OperationSpec('feedback_learning_read',
    'Read feedback-learning job status or its proposed memory/training material/harness/system changes. '
    'These are unverified proposals, never applied changes or training-ready examples. '
    'Omit job_id to list jobs; use next_offset for the next page.',
    {'type': 'object', 'properties': {
        'job_id': {'type': 'string', 'minLength': 1},
        'research_attempt_id': {'type': 'string', 'minLength': 1, 'maxLength': 200,
                                'description': 'Filter job listings by a recorded attempt ID.'},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
     'additionalProperties': False}, read_only=True)


def _read_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) - {'job_id', 'research_attempt_id', 'offset', 'limit'}:
        raise ValueError('Invalid feedback_learning_read arguments')
    if 'job_id' in arguments and (not isinstance(arguments['job_id'], str) or not arguments['job_id'].strip()):
        raise ValueError('job_id must be a nonempty string')
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
        data = self.learning.read(**_read_arguments(call.arguments))
        return OperationResult('Feedback learning proposals; validation still required', (),
                               bool(data.get('simulated', False)), data=data)


def feedback_learning_decision_factory(**learning_options):
    """Return an Application decision_factory; no current run is reconfigured.

    The ordinary assembly supplies tools, permissions and memories. Full snapshot
    and durable Runner restoration still require an explicit extension contract.
    The optional learning journal alone persists its own completed/pending jobs.
    """
    def build(context, exploration):
        from rsisys.assembly import build_default_decision
        return _build(context, exploration, build_default_decision, learning_options)
    return build


def _build(context, exploration, default_factory, options):
    base_loop = default_factory(context, exploration)
    path = context.data_dir / 'feedback_learning.json' if context.data_dir is not None else None
    enable_feedback_learning(base_loop, path=path, **options)
    context.add_cleanup(base_loop.harness.close)
    return base_loop


def enable_feedback_learning(base_loop, *, path=None, **options):
    """Install before running; keep domain policy parsing, budgets and identities.

    This is an assembly operation, not a live-run reconfiguration API. Custom
    harnesses and nonempty decision state require their own integration contract.
    """
    from .policy import ModelDecisionPolicy
    from .state import DecisionState
    base = base_loop.harness
    if type(base) is not DefaultDecisionHarness:
        raise TypeError('Feedback learning requires an unstarted default harness')
    pristine = DecisionState()
    pristine.operation_results = list(base.state.operation_results)
    if base.state != pristine:
        raise ValueError('Install feedback learning before the first decision')
    if not isinstance(base.policy, MemoryPrefixDecisionPolicy) and type(base.policy) is not ModelDecisionPolicy:
        raise TypeError('Custom policies must implement MemoryPrefixDecisionPolicy')
    space = LearningActionSpace(base_loop.action_space)
    backend = SerializedLearningBackend(base.policy.backend)
    policy = base.policy
    if not isinstance(policy, MemoryPrefixDecisionPolicy):
        policy = MemoryPrefixDecisionPolicy(backend, base.policy.model,
            directives=base.policy.directives, allowed_actions=base.policy.allowed_actions,
            max_validation_attempts=base.policy.max_validation_attempts,
            structured_output=base.policy.structured_output, context_compactor=base.context_compactor,
            project_context=base.policy.project_context, attempt_contract=base.policy.attempt_contract)
    learning = FeedbackLearning(backend, path=path, **options)
    policy.backend = backend
    if hasattr(policy, 'space'):
        policy.space = space
    harness = FeedbackLearningHarness(policy, LearningTools(base.tools, learning), base.experience, base.learner,
        feedback_learning=learning, experience_limit=base.experience_limit, memory=base.memory,
        memory_limit=base.memory_limit, model_activator=base.model_activator, review_tool_limit=base.review_tool_limit,
        max_outstanding=base.max_outstanding, feedback_selector=base.feedback_selector,
        context_compactor=base.context_compactor, research_agenda=base.research_agenda,
        attempt_contract=base.attempt_contract)
    harness.state = base.state
    base_loop.harness, base_loop.action_space = harness, space
    return harness
