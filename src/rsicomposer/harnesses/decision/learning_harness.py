"""Explicit host-evidence adapter for the independent feedback service.

Controller harnesses own context projection; the host owns execution facts.
Four-port Harnesses receive the public Memory projection without extra hooks.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace

from rsicomposer.harnesses.decision.context import result_id
from rsicomposer.harnesses.decision.feedback import (
    FeedbackEvidence,
    FeedbackLearningService,
)
from rsicomposer.harnesses.decision.feedback.learning import FeedbackLearning
from rsicomposer.harnesses.decision.feedback.model_gateway import SerializedLearningBackend
from rsicomposer.harnesses.decision.research_state import research_working_view
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import (
    OperationArgumentError,
    OperationCall,
    OperationSpec,
)
from rsicomposer.loops.contracts.feedback_context import ContextQuery
from rsicomposer.loops.decision_making.action_space.capabilities import Capability
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.storage.json_tools import EvidencePathError

class HostFeedbackAdapter:
    """Translate the built-in execution ledger; the learning service is agnostic."""
    def __init__(self, harness, execution, *, current_model, controller_version,
                 model=None, model_provider=None):
        if not callable(current_model) or not callable(controller_version):
            raise TypeError('Feedback requires explicit endpoint model and version callbacks')
        if model_provider is not None and not callable(model_provider):
            raise TypeError('model_provider must be callable')
        self.harness, self.execution = harness, execution
        self._current_model = current_model
        self._controller_version = controller_version
        self._model_provider = model_provider
        current = self.model()
        if model is not None and model != current:
            raise ValueError('Feedback must use the active meta-agent model')
        self.service = None

    def model(self):
        # Resolve for each new capture; an install-time argument must not pin
        # this role to old weights after the meta-agent changes model version.
        active = self._current_model()
        if not isinstance(active, ModelRef):
            raise ValueError('Feedback requires an explicit host meta-agent model binding or model_provider')
        if self._model_provider is not None and self._model_provider() != active:
            raise ValueError('Feedback must use the active meta-agent model')
        return active


    def decision_ids(self):
        return tuple(self.execution.state.decision_records)

    def proposal_decisions(self):
        return tuple({key: deepcopy(row[key]) for key in
            ('decision_id', 'status', 'kind', 'operations', 'error', 'feedback_proposal_ids')
            if key in row} for row in self.execution.state.decision_records.values() if row.get('feedback_proposal_ids'))

    def install(self, service):
        self.service = service
        bind = optional_method(self.harness, 'bind_feedback')
        self._projects_in_harness = bind is not None
        self._delivered = optional_method(self.harness, 'receipt_context_delivered')
        with ExitStack() as cleanup:
            cleanup.callback(self.execution.add_receipt_listener(self._receive_operation, owner=self.harness))
            if bind is not None:
                cleanup.callback(bind(service))
            return cleanup.pop_all().close

    def project_window(self, window):
        if self._projects_in_harness:
            return window
        return self.service.memory_window(window)

    def _receive_operation(self, receipt):
        # Denied operations are absent from the authorized catalog, so their
        # read_only flag may be unknown. Our own navigation must never create
        # feedback about reading feedback, including denied reads.
        if not receipt.read_only and receipt.request.get('name') != READ_SPEC.name:
            ident = self.service.observe(receipt, evidence=self.receipt_evidence(receipt))
            if self._delivered is not None and self._delivered(receipt.task_id):
                self.service.learning.acknowledge_context_delivery(ident)
            return ident
        # Raw queries belong to the trajectory; reading feedback must not
        # recursively invoke a model to interpret that read.
        if self.service.learning.trajectory_archive is not None:
            origin = self.execution.state.decision_records.get(receipt.decision_id, {})
            # Archive this action, not another copy of its entire Attempt or
            # grouping catalog. Queries never become background memory jobs.
            return self.service.archive(receipt, evidence=FeedbackEvidence(self.execution.memory.history(),
                research_context={'decision_id': receipt.decision_id,
                    'research_attempt_id': origin.get('research_attempt_id'),
                    'feedback_proposal_ids': list(origin.get('feedback_proposal_ids', ())),
                    'scope': 'single_decision_receipt_archive'}))

    def receipt_evidence(self, receipt):
        return FeedbackEvidence(self.execution.memory.history(),
            research_context=self._learning_scope(decision_id=receipt.decision_id),
            grouping_context=self._grouping_scope(decision_id=receipt.decision_id))

    def _grouping_scope(self, *, decision_id=None):
        """Bounded evidence window, independent of canonical execution membership."""
        records = self.execution.state.decision_records
        ident = decision_id
        selected = list(records)[-24:]
        if ident in records and ident not in selected:
            selected.append(ident)
        # Keep producer and declared dependency links, including older decisions.
        for key in list(selected):
            record = records[key]
            sources = list(record.get('source_decision_ids', ())) + list(record.get('depends_on_decision_ids', ()))
            for source in sources:
                if source in records and source not in selected and len(selected) < 32:
                    selected.append(source)
        graph = {r['decision_id']: r for r in self.execution.decision_graph()}
        decisions = []
        for key in selected:
            record = records[key]
            proposal = record['proposal']
            action = {k: deepcopy(proposal[k]) for k in
                      ('kind', 'operations', 'job', 'control', 'simulated') if k in proposal}
            decisions.append({**graph[key], 'kind': record['kind'], 'reason': record.get('reason', ''),
                'proposal': action, 'error': record.get('error'),
                'decision_model': deepcopy(record.get('decision_model'))})
        catalog = self.attempt_interpretations()['items']
        attempts = [r for r in catalog if set(r['member_decision_ids']).intersection(selected) or r in catalog[-8:]]
        call_ids = {call['call_id'] for key in selected
                    for call in records[key]['proposal'].get('operations', ())}
        observations = [asdict(h) for h in self.execution.memory.history()
                        if h.task_id in call_ids or h.record_id in selected
                        or h.details.get('decision_id') in selected]
        return {'decisions': decisions, 'attempts': attempts, 'observations': observations[-16:],
                'total_observations': len(observations),
                'scope': 'bounded_observed_decisions', 'total_decisions': len(records),
                'truncated': len(decisions) < len(records), 'receipt_decision_id': ident}

    def attempt_interpretations(self):
        return self.service.attempts()

    def _learning_scope(self, *, decision_id=None):
        """Snapshot canonical membership; never infer groups from a proposal/title."""
        ident = decision_id
        origin = self.execution.state.decision_records.get(ident, {})
        attempt = origin.get('research_attempt_id')
        members = []
        for item in self.execution.decision_graph():
            if (item['decision_id'] != ident and
                    (attempt is None or item.get('research_attempt_id') != attempt)):
                continue
            record = self.execution.state.decision_records[item['decision_id']]
            members.append({**item, 'call_ids': [call['call_id'] for call in record['proposal'].get('operations', ())],
                            'kind': record['kind'], 'reason': record.get('reason', '')})
        scope = dict(decision_id=ident, research_attempt_id=attempt,
                     research_attempt_title=origin.get('research_attempt_title', ''),
                     member_decisions=members, scope='observed_members_at_feedback',
                     attempt_completion='not_reported')
        scope['feedback_proposal_ids'] = list(origin.get('feedback_proposal_ids', ()))
        scope['issuing_decision_model'] = origin.get('decision_model')
        scope['learning_decision_model'] = asdict(self.model())
        scope['controller_version'] = self._controller_version()
        return {**scope, 'snapshot_digest': result_id(scope)}


class ResearchFeedbackProjection:
    """The optional rich view owned by the native research Harness."""
    def __init__(self, harness, service):
        self.harness, self.service = harness, service

    def __call__(self, context):
        self.service.refresh()
        context = replace(context, memories=self.harness.memory.recall(limit=self.harness.memory_limit))
        jobs = self.service.learning.ordered_jobs()
        navigation = {'read_operation': 'feedback_learning_read',
            'authority': 'unverified_proposals', 'total': len(jobs),
            'recent': [{'job_id': row['job_id'], 'status': row['status'],
                        'research_attempt_id': row['input'].get('research_context', {}).get('research_attempt_id'),
                        'kinds': [p['kind'] for p in row['proposals']],
                        'context_delivery': deepcopy(row.get('context_delivery')),
                        'archive_delivery': {key: deepcopy(value) for key, value in
                            row.get('archive_delivery', {}).items() if key in ('status', 'error')},
                        'publication_delivery': deepcopy(row.get('publication_delivery')),
                        'products': deepcopy(row.get('products', {}))} for row in jobs[-6:]]}
        navigation['scheduler_error'] = self.service.learning.state.get('scheduler_error')
        navigation['context_error'] = deepcopy(self.service.learning.state.get('context_error'))
        outputs = self.service.outputs()
        navigation['product_outputs'] = {'total': len(outputs),
            'recent': [{key: deepcopy(row[key]) for key in
                       ('event_id', 'product', 'authority', 'applied', 'read_path')} for row in outputs[-6:]]}
        proposals = self.service.proposals()
        navigation['proposals'] = {'total': len(proposals),
            'read_path': {'operation': 'feedback_learning_read', 'view': 'proposals'},
            'decision_owner': 'main_meta_agent_decision', 'dataset_construction_owner': 'lab',
            'recent': [{'proposal_id': p['proposal_id'], 'kind': p['kind'], 'target': p['target'], 'subject': p['subject'],
                'content_preview': p['proposal']['content'][:200], 'read_path': p['read_path'],
                'decision_links': p['decision_links'][-4:], 'applied': False} for p in proposals[-6:]]}
        context = replace(context, research_agenda={**context.research_agenda, 'feedback_learning': navigation})
        memory_context = self.service.context(ContextQuery(objective=context.objective))
        navigation['context_error'] = deepcopy(self.service.learning.state.get('context_error'))
        # Memory organization is opaque text. Host evidence/dependency projection
        # stays separate, so providers cannot overwrite live protection metadata.
        view = research_working_view(context, jobs)
        view = replace(view, memory_context=memory_context)
        self.harness.last_working_view = deepcopy({'research_state': view.research_state,
            'memory_context': memory_context,
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
    'Read /trajectory_snapshot for an immutable raw capture. '
    'Dataset construction, filtering, labeling and policy training belong to an explicitly configured Lab. '
    'Published task proposals can be referenced by feedback_proposal_ids in a later ordinary decision; '
    'feedback does not construct datasets, train or activate changes. '
    'Attempt views return latest revisions; proposals stay unverified. '
    'These are unverified proposals, never applied changes or training-ready examples. '
    'Omit job_id to list jobs; use next_offset for the next page. '
    'With job_id and evidence_path (JSON pointer, empty for root), read original evidence '
    'in JSON-text pages; offset counts characters and each limit unit is 128 characters. '
    '/proposals and /attempt_revisions address interpretation output, not captured evidence. '
    '/outputs and /output read native product payloads; product fields are not host state or execution authority. '
    '/feedback_request and /feedback_response read the recorded feedback model exchange when available. '
    'Knowledge views also return JSON-text pages; pass sha256 as expected_digest on subsequent pages.',
    {'type': 'object', 'properties': {
        'view': {'enum': ['jobs', 'attempts', 'proposals', 'trajectories']},
        'expected_digest': {'type': 'string', 'minLength': 64, 'maxLength': 64},
        'job_id': {'type': 'string', 'minLength': 1},
        'evidence_path': {'type': 'string', 'maxLength': 1000,
                          'description': 'JSON pointer into original evidence; requires job_id.'},
        'research_attempt_id': {'type': 'string', 'minLength': 1, 'maxLength': 200,
                                'description': 'Filter job listings by a recorded attempt ID.'},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
     'additionalProperties': False}, read_only=True)


def _read_arguments(arguments, targets):
    if not isinstance(arguments, dict) or set(arguments) - {'job_id', 'research_attempt_id', 'offset', 'limit',
                                                          'evidence_path', 'view', 'target', 'expected_digest'}:
        raise OperationArgumentError('Invalid feedback_learning_read arguments')
    view = arguments.get('view', 'jobs')
    if view not in ('jobs', 'attempts', 'proposals', 'trajectories'):
        raise OperationArgumentError('Invalid feedback view')
    if view != 'jobs' and ('job_id' in arguments or 'evidence_path' in arguments):
        raise OperationArgumentError('Knowledge view cannot be combined with job evidence')
    if 'target' in arguments and (view != 'proposals' or arguments['target'] not in targets):
        raise OperationArgumentError('target requires a proposals view and an installed proposal target')
    if 'expected_digest' in arguments and (not isinstance(arguments['expected_digest'], str)
            or len(arguments['expected_digest']) != 64):
        raise OperationArgumentError('Invalid expected_digest')
    if 'expected_digest' in arguments and view == 'jobs' and 'evidence_path' not in arguments:
        raise OperationArgumentError('expected_digest requires an evidence or knowledge page')
    if 'job_id' in arguments and (not isinstance(arguments['job_id'], str) or not arguments['job_id'].strip()):
        raise OperationArgumentError('job_id must be a nonempty string')
    if 'evidence_path' in arguments:
        pointer = arguments['evidence_path']
        if ('job_id' not in arguments or not isinstance(pointer, str) or len(pointer) > 1000
                or (pointer and not pointer.startswith('/'))):
            raise OperationArgumentError('evidence_path requires job_id and a bounded JSON pointer')
    if 'research_attempt_id' in arguments:
        ident = arguments['research_attempt_id']
        if not isinstance(ident, str) or not ident.strip() or len(ident) > 200 or 'job_id' in arguments:
            raise OperationArgumentError('research_attempt_id must be a bounded ID for a job listing')
    if type(arguments.get('offset', 0)) is not int or arguments.get('offset', 0) < 0:
        raise OperationArgumentError('offset must be nonnegative')
    if type(arguments.get('limit', 8)) is not int or not 1 <= arguments.get('limit', 8) <= 16:
        raise OperationArgumentError('limit must be in 1..16')
    return deepcopy(arguments)


def feedback_read_capability(learning):
    targets = sorted({rule.target for rule in learning.proposal_targets.values()})
    schema = deepcopy(READ_SPEC.input_schema)
    if targets:
        schema['properties']['target'] = {'enum': targets, 'description': 'Filter by an installed target; requires view=proposals.'}
    spec = replace(READ_SPEC, input_schema=schema)

    def execute(arguments):
        try:
            data = learning.read(**arguments)
        except EvidencePathError as exc:
            return OperationResult('Feedback read rejected', (), False, ExecutionStatus.FAILED, str(exc),
                data={'view': arguments.get('view', 'jobs'), 'job_id': arguments.get('job_id'),
                      'evidence_path': arguments.get('evidence_path'), 'read_operation': READ_SPEC.name})
        return OperationResult('Feedback learning proposals; validation still required', (),
                               bool(data.get('simulated', False)), data=data)

    return Capability.from_spec(spec, lambda args: OperationCall(spec.name, _read_arguments(args, targets)), execute)


def enable_feedback_learning(base_loop, **options):
    """Standalone integration: an explicit Harness cleanup port or caller owns the service."""
    service = build_feedback_service(base_loop, **options)
    register = optional_method(base_loop.harness, 'add_cleanup')
    rollback = None
    try:
        rollback = install_feedback_service(base_loop, service)
        if register is not None:
            register(service.close)
    except BaseException as error:
        if rollback is not None:
            rollback()
        close_failed_install(service, error, register=register)
        raise
    return base_loop.harness


def install_feedback_service(loop, service):
    contribute = optional_method(service.memory, 'capabilities')
    tools = (feedback_read_capability(service.learning), *(contribute() if contribute else ()))
    rollback_tools = loop.install_capabilities(tools)
    try:
        service.install_adapter()
    except BaseException:
        rollback_tools()
        raise
    loop.integration = service

    def rollback():
        loop.integration = None
        rollback_tools()
    return rollback


def build_feedback_service(base_loop, *, path=None, backend=None, model=None, adapter=None, model_provider=None,
                             **options):
    """Build this version's feedback service without installing listeners or tools.

    An explicit backend may differ from the decision backend. It must enforce its
    host's shared admission/budget when that deployment has a global call limit.
    Four-port Harnesses use the public projection; a custom adapter may replace it.
    """
    from rsicomposer.harnesses.decision.feedback.learning import bind_feedback_processor
    base = base_loop.harness
    execution = getattr(base_loop, 'host_execution', None)
    if base_loop.integration is not None:
        raise ValueError('Feedback learning is already installed')
    if adapter is None:
        if execution is None:
            raise TypeError('Feedback requires a host execution evidence adapter')
        adapter = HostFeedbackAdapter(base, execution, current_model=base_loop.current_model,
                                         controller_version=lambda: getattr(base_loop, 'controller_version', None),
                                         model=model, model_provider=model_provider)
    elif model_provider is not None:
        raise ValueError('Custom feedback adapters supply their own model provider')
    for method in ('model', 'install'):
        if not callable(getattr(adapter, method, None)):
            raise TypeError('FeedbackAdapter requires ' + method)
    if not isinstance(adapter.model(), ModelRef):
        raise TypeError('Feedback adapter model() must return the current ModelRef')
    policy = getattr(base_loop, 'policy', None)
    processor = options.get('processor')
    requires_model = processor is None or bind_feedback_processor(processor).max_calls > 0
    share = optional_method(policy, 'share_backend')

    def serialized(value):
        return value if isinstance(value, SerializedLearningBackend) else SerializedLearningBackend(value)

    if backend is None and requires_model:
        if share is None:
            raise ValueError('Session/custom policies require an explicit feedback backend')
        backend = share(serialized)
    elif backend is not None and share is not None and backend is getattr(policy, 'backend', None):
        backend = share(serialized)
    if requires_model and not callable(getattr(backend, 'generate', None)):
        raise ValueError('Feedback inference requires a model backend')
    backend = serialized(backend) if backend is not None else None
    learning = FeedbackLearning(backend, path=path, **options)
    return FeedbackLearningService(learning, adapter)


def close_failed_install(service, error, *, register):
    try:
        service.close()
    except BaseException as cleanup:
        error.add_note('Feedback installation cleanup failed: ' + str(cleanup))
        if register is not None:
            register(service.close)
