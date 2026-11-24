"""Controller construction, version adoption, recovery and training lifecycle."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from rsicomposer.component_context import ControllerContext, EndpointContext
from rsicomposer.harnesses.decision.learner import DecisionTrainingRequest
from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.loops.contracts.coordination import LoopStepResult, OperationResult
from rsicomposer.loops.contracts.decision import OperationArgumentError, OperationCall, OperationSpec
from rsicomposer.loops.contracts.jobs import JobStatus
from rsicomposer.loops.contracts.metrics import InvalidMetrics, validated_metrics
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.loops.contracts.training import ModelActivationPending, TrainingResult
from rsicomposer.loops.decision_making.action_space.capabilities import Capabilities, Capability
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.evidence.records import ExperienceRecord
from rsicomposer.runtime.models.model_session import validate_model_route
from rsicomposer.runtime.orchestration.jobs import JobBook
from rsicomposer.runtime.state.controller_versions import save_controller_version
from rsicomposer.runtime.storage.files import json_bytes


class BuiltController:
    def __init__(self, endpoint, context):
        self.endpoint, self.context = endpoint, context

    @property
    def policy(self):
        return _policy(self.endpoint)

    def capture(self):
        from rsicomposer.method_state import PolicyDecisionState
        return PolicyDecisionState(self.endpoint).capture()

    def restore(self, state):
        from rsicomposer.method_state import PolicyDecisionState
        PolicyDecisionState(self.endpoint).restore(deepcopy(state))
        current_model = getattr(self.policy, 'model', None) or self.endpoint.model_binding
        if current_model is not None:
            self.bind_model(current_model)

    def bind_model(self, model):
        validate_model_route(self.context, model)
        policy = self.policy
        current = getattr(policy, 'model', None)
        if isinstance(current, ModelRef) and current != model:
            update = optional_method(policy, 'update_model')
            if update is None:
                raise TypeError('Controller policy cannot bind a different decision model')
        self.context.roles = {**self.context.roles, 'decision_maker': model}
        self.endpoint.bind_model(model)

    def validate_snapshot(self):
        from rsicomposer.method_state import PolicyDecisionState
        PolicyDecisionState(self.endpoint).validate_snapshot()
        self.capture()

    def close(self):
        self.context.close()


class ControllerEndpoint:
    """A stable host port whose active Controller is replaced at checkpoints."""
    def __init__(self, host):
        self.host = host

    def __getattr__(self, name):
        return getattr(self.host.active.endpoint, name)

    def checkpoint_boundary(self):
        return self.host.advance()

    def recovery_bindings(self):
        # The host explicitly owns and captures all replaceable Controller parts.
        return (self.host, self.host.context.host_execution)


class ControllerHostState:
    state_id = 'rsicomposer-controller-host/v5'

    def __init__(self, host):
        self.host = host

    def capture(self):
        return self.host.capture()

    def restore(self, state):
        self.host.restore(state)

    def validate_snapshot(self):
        self.host.validate_snapshot()


def _context(owner, version, *, initial, endpoint):
    root = owner.data_dir
    if not initial and root is not None:
        # IDs are data, never filesystem paths.
        root = Path(root) / 'controller_versions' / sha256(version.version_id.encode()).hexdigest() / 'workspace'
        root.mkdir(parents=True, exist_ok=True)
    roles = dict(owner.roles)
    if version.model is not None:
        validate_model_route(owner, version.model)
        roles['decision_maker'] = version.model
    local = replace(owner, data_dir=root, roles=roles,
                    restoring=owner.restoring or not initial)
    local._artifact_access, local._model_access = owner._artifact_access, owner._model_access
    local._controller_version = version.version_id
    local._evidence_owner = owner
    if hasattr(owner, '_worker_services'):
        local._worker_services = owner._worker_services
    scope = EndpointContext if endpoint else ControllerContext
    return local, scope(local, owner)


def build_controller(owner, definition, options, version, *, initial=False, memory=None):
    """Rebuild a catalog-selected policy, harness and Memory with local ownership."""
    from rsicomposer.assembly import build_default_decision
    from rsicomposer.harnesses.defaults import build_harness_decision
    local, scoped = _context(owner, version, initial=initial, endpoint=definition.endpoint)
    try:
        if definition.endpoint:
            endpoint = definition.policy(scoped, owner.lab_execution, deepcopy(options))
        else:
            policy = definition.policy(scoped, deepcopy(options))
            close = optional_method(policy, 'close')
            if close is not None:
                local.add_owned_cleanup(close)
            if not callable(policy) and optional_method(policy, 'decide') is None:
                raise TypeError('Controller factory must return a policy or explicit endpoint')
            custom = definition.harness(scoped, deepcopy(options)) if definition.harness is not None else None
            if custom is not None:
                endpoint = build_harness_decision(scoped, owner.lab_execution, policy=policy, harness=custom)
            else:
                endpoint = build_default_decision(scoped, owner.lab_execution, policy=policy)
        endpoint.controller_version = local._controller_version
        endpoint.model_scope = scoped.models.scope
        close = optional_method(endpoint.harness, 'close')
        if close is not None:
            local.add_owned_cleanup(close)
        if endpoint.operation_router.lab is not owner.lab_execution:
            raise ValueError('Controller endpoint must retain the assembled Lab routing')
        required_policy = _policy(endpoint)
        local.controller_report = optional_method(required_policy, 'report')
        owner.lab_execution.validate_controller(required_policy)
        if definition.endpoint and endpoint is not required_policy:
            owner.lab_execution.validate_controller(endpoint)
        required = optional_method(required_policy, 'required_operations')
        if required is not None:
            names = required()
            if (not isinstance(names, tuple) or any(not isinstance(name, str) or not name for name in names)
                    or len(set(names)) != len(names)):
                raise TypeError('required_operations must return distinct operation names')
            available = {spec.name for spec in owner.lab_execution.describe()}
            if set(names) - available:
                raise ValueError('Controller requires missing Lab operations: ' + ', '.join(sorted(set(names) - available)))
        if memory is None and definition.memory is not None:
            memory = definition.memory(deepcopy(options))
        if memory is not None:
            memory.install(endpoint, owner=local,
                path=local.data_dir / 'feedback_learning.json' if local.data_dir is not None else None)
        return BuiltController(endpoint, local)
    except BaseException as error:
        try:
            local.close()
        except BaseException as cleanup:
            error.add_note('Controller build cleanup failed: ' + str(cleanup))
        raise


def initial_controller_version(selection, model):
    from rsicomposer.loops.contracts.controller_version import ControllerVersion
    # Config/model determine the initial version without a random restore ID.
    from rsicomposer.runtime.storage.codec import encode
    identity = sha256(json_bytes(encode({'selection': selection, 'model': model}))).hexdigest()
    return ControllerVersion('initial-' + identity, deepcopy(selection), model=model)


def _policy(endpoint):
    from rsicomposer.harnesses.decision.public_policy import ResearchPolicyAdapter
    policy = endpoint.policy
    return policy.implementation if isinstance(policy, ResearchPolicyAdapter) else policy


def _version_ref(value):
    if isinstance(value, dict):
        try:
            value = ArtifactRef(**value)
        except (TypeError, ValueError) as exc:
            raise OperationArgumentError('version must be a ControllerVersion ArtifactRef') from exc
    if not isinstance(value, ArtifactRef) or value.kind != 'controller-version':
        raise OperationArgumentError('version must be a ControllerVersion ArtifactRef')
    value.__post_init__()
    return value


def controller_capabilities(handler):
    """Construct the host catalog before its initial Controller is built."""
    def validate(arguments):
        if not isinstance(arguments, dict) or set(arguments) != {'version'}:
            raise OperationArgumentError('host.adopt_controller requires exactly version')
        return OperationCall('host.adopt_controller', {'version': asdict(_version_ref(arguments['version']))})
    spec = OperationSpec('host.adopt_controller',
        'Request adoption of a registered ControllerVersion at the next host checkpoint boundary.',
        {'type': 'object', 'properties': {'version': {'type': 'object'}},
         'required': ['version'], 'additionalProperties': False})
    return Capabilities((Capability.from_spec(spec, validate, handler),))


class ControllerHost:
    """Own complete Controller instances while a stable endpoint delegates to active.

    build_controller(version) constructs a trusted BuiltController exposing
    endpoint, context, capture/restore, validate_snapshot and close. Factories and
    runnable code are selected by the enclosing assembly's explicit catalog.
    """
    def __init__(self, context, initial_instance, initial_version, build_controller, *, activator=None, learner=None):
        self.context, self.active = context, initial_instance
        self.current_version = deepcopy(initial_version)
        self.build_controller, self.activator = build_controller, activator
        self.pending = None
        self.archives = []
        self._retired = []
        self._adopted = {}
        self._version_deliveries = {}
        self._pending_adoption_event = None
        self.prepared_models = {}
        if learner is None and context.with_learning:
            from rsicomposer.harnesses.decision.learner.backends.mock import MockDecisionLearner
            learner = MockDecisionLearner(context.artifacts)
        self.learning = ControllerLearning(self, learner)
        self.capabilities = controller_capabilities(self.request)
        self._label(self.active)

    @property
    def version_id(self):
        return self.current_version.version_id

    def _label(self, instance, *, delivery=None):
        self.context.host_execution.activate(instance.endpoint, self.version_id, delivery=delivery)

    def _validate_execution(self, endpoint):
        active = self.active.endpoint
        if endpoint.host_execution is not active.host_execution:
            raise ValueError('Controller replacement must retain the run execution owner')
        if endpoint.operation_router.lab is not active.operation_router.lab:
            raise ValueError('Controller replacement must retain the run Lab routing')

    def _delivery(self):
        state = self.context.host_execution.state
        return {'feedback': state.decision_feedback_cursor, 'progress': state.progress_cursor}

    def _validate(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {'version'}:
            raise OperationArgumentError('host.adopt_controller requires exactly version')
        ref = _version_ref(arguments['version'])
        return OperationCall('host.adopt_controller', {'version': asdict(ref)})

    def request(self, arguments):
        return self.request_version(_version_ref(self._validate(arguments).arguments['version']))

    def request_version(self, ref, *, training_result=None):
        from rsicomposer.runtime.state.controller_versions import load_controller_version
        ref = _version_ref(ref)
        version = load_controller_version(self.context.artifacts, ref)
        registered_training = self.learning.training_result(ref)
        if training_result is None:
            training_result = registered_training
        elif registered_training is not None and registered_training != training_result:
            raise ValueError('Controller training result differs from its registered candidate')
        if version.version_id == self.version_id:
            if version != self.current_version:
                raise ValueError('Controller version ID is already bound to different content')
            return OperationResult('Controller version already active', (ref,), False,
                                   data={'controller_version': self.version_id, 'status': 'active'})
        if self.pending is not None:
            if self.pending['ref'] != ref:
                raise RuntimeError('A Controller adoption is already pending')
        else:
            if version.version_id in self._adopted:
                raise ValueError('Controller version IDs cannot be reused')
            calls = tuple(row['operation'] for row in self.active.endpoint.in_flight_operations()
                          if row['operation'].name != 'host.adopt_controller')
            planner = optional_method(_policy(self.active.endpoint), 'plan_swap')
            proposed = planner(deepcopy(calls)) if planner is not None else {}
            from collections.abc import Mapping
            if (not isinstance(proposed, Mapping) or set(proposed) - {call.call_id for call in calls}
                    or any(mode not in ('drain', 'cancel', 'handover') for mode in proposed.values())):
                raise ValueError('SwapPlanner must map pending call IDs to drain, cancel or handover')
            self.pending = {'ref': deepcopy(ref), 'version': deepcopy(version),
                'plan': {call.call_id: proposed.get(call.call_id, 'drain') for call in calls},
                'training_result': deepcopy(training_result), 'adoption_id': str(uuid4()),
                'delivery': deepcopy(self._version_deliveries.get(version.version_id, self._delivery()))}
        return OperationResult('Controller adoption queued for checkpoint boundary', (ref,),
            bool(training_result and training_result.simulated),
            data={'controller_version': version.version_id, 'status': 'pending'})

    def advance(self):
        """Drain one receipt or atomically adopt; never sample either policy here."""
        self._close_retired()
        self._publish_adoption()
        if self.pending is None:
            return None
        old = self.active
        endpoint = old.endpoint
        if endpoint.pending_records():
            endpoint.flush_pending_records()
            return LoopStepResult('progressed', reason='Controller swap records flushed', decision_steps=0)
        if endpoint.pending_research_job() is not None:
            endpoint.drain_research()
            return LoopStepResult('waiting' if endpoint.pending_research_job() is not None else 'progressed',
                job=endpoint.pending_research_job(), reason='Draining Controller research before replacement', decision_steps=0)
        tasks = endpoint.in_flight_operations()
        # Requests admitted with the adopt call but absent from the planner's
        # snapshot still drain; a planner cannot silently abandon future work.
        plan = {row['operation'].call_id: self.pending['plan'].get(row['operation'].call_id, 'drain') for row in tasks}
        lab = endpoint.operation_router.lab
        names = {spec.name for spec in lab.describe()} if lab is not None else set()
        blocking = {call.call_id: ('drain' if mode == 'handover' else mode)
                    for call in (row['operation'] for row in tasks)
                    if (mode := plan[call.call_id]) != 'handover' or call.name not in names}
        if blocking:
            return endpoint.drain_operation_plan(blocking)
        version = self.pending['version']
        training = self.pending['training_result']
        if training is not None:
            from rsicomposer.runtime.models.activation import prepare_model_activation
            self._prepared_model_target(training.model, training)
            try:
                ready = prepare_model_activation(training, self.activator,
                    idempotency_key='controller-version:' + version.version_id)
            except ModelActivationPending:
                return LoopStepResult('waiting', reason='Controller model activation pending', decision_steps=0)
            if ready != version.model:
                raise ValueError('Prepared Controller model differs from the candidate version')
            self._grant_prepared_model(ready, training)
            self.prepared_models[ready] = {'result': deepcopy(training),
                                          'activation_key': 'controller-version:' + version.version_id}
        candidate = self.build_controller(deepcopy(version))
        try:
            self._validate_execution(candidate.endpoint)
            if candidate.endpoint is endpoint or candidate.endpoint.harness is endpoint.harness:
                raise TypeError('Controller construction must return an independently owned endpoint and harness')
            integration = getattr(candidate.endpoint, 'integration', None)
            previous = getattr(old.endpoint, 'integration', None)
            if integration is not None and previous is not None and integration.memory is not None and integration.memory is previous.memory:
                raise TypeError('Controller construction must create an independently owned Memory')
            if version.policy_state is not None:
                restore = optional_method(_policy(candidate.endpoint), 'restore')
                if restore is None or restore(deepcopy(version.policy_state)) is not None:
                    raise TypeError('Candidate Controller must restore its policy state explicitly')
            self._restore_memory(candidate, version.memory_state)
            if version.model is not None:
                candidate.bind_model(version.model)
                if candidate.endpoint.current_model() != version.model:
                    raise ValueError('Candidate Controller did not bind its prepared model')
            actual_model = getattr(_policy(candidate.endpoint), 'model', None) or candidate.endpoint.model_binding
            if actual_model is not None:
                candidate.bind_model(actual_model)
            archive = {'version': deepcopy(self.current_version), 'state': deepcopy(old.capture())}
        except BaseException as error:
            try:
                candidate.close()
            except BaseException as cleanup:
                self._retired.append(candidate)
                error.add_note('Rejected Controller cleanup failed: ' + str(cleanup))
            raise
        self.archives.append(archive)
        self._retired.append(old)
        self.current_version = deepcopy(version)
        self.active = candidate
        self._adopted[version.version_id] = deepcopy(self.pending['ref'])
        delivery = self.pending['delivery']
        self._pending_adoption_event = {'adoption_id': self.pending['adoption_id'],
                                        'controller_version': self.version_id}
        self.pending = None
        self._label(candidate, delivery=delivery)
        self._publish_adoption()
        # Old cleanup can fail, but adoption is already represented in state;
        # retain failed instances for an explicit retry without reverting it.
        self._close_retired()
        return LoopStepResult('progressed', reason='Controller adopted: ' + self.version_id, decision_steps=0)

    def _publish_adoption(self):
        payload = self._pending_adoption_event
        if payload is None:
            return
        for record in self.context.experience.retrieve(kinds=('controller_event',)):
            event = record.details.get('event')
            if (isinstance(event, dict) and event.get('kind') == 'controller_adopted'
                    and isinstance(event.get('output'), dict)
                    and event['output'].get('adoption_id') == payload['adoption_id']):
                if event['output'] != payload:
                    raise ValueError('Controller adoption ID is bound to another event')
                break
        else:
            self.context.record('controller_adopted', deepcopy(payload))
        self._pending_adoption_event = None

    def _restore_memory(self, instance, state):
        if state is None:
            return
        integration = getattr(instance.endpoint, 'integration', None)
        if integration is None:
            raise TypeError('Candidate Controller requires an explicit Memory integration')
        history = self._approved_memory_history(state) if (isinstance(state, dict)
            and state.get('format') == 'rsicomposer-agent-memory-version/v1') else ()
        if integration.seed_version(deepcopy(state), history=history) is not None:
            raise TypeError('Candidate Memory seed_version must return None')

    def approve_memory_history(self, version):
        """Approve parent evidence before an isolated child receives a version."""
        state = version.memory_state
        if not isinstance(state, dict) or state.get('format') != 'rsicomposer-agent-memory-version/v1':
            return ()
        return self._approved_memory_history(state)

    def _approved_memory_history(self, state):
        from rsicomposer.harnesses.memory import HistoryEntry
        if (not isinstance(state, dict) or set(state) != {'format', 'memory', 'history'}
                or state['format'] != 'rsicomposer-agent-memory-version/v1'
                or not isinstance(state['history'], (tuple, list))
                or any(not isinstance(row, HistoryEntry) for row in state['history'])
                or len({row.record_id for row in state['history']}) != len(state['history'])):
            raise ValueError('Invalid AgentMemory Controller version evidence envelope')
        trusted = {}
        for row in getattr(self.context, '_inherited_controller_history', ()):
            if not isinstance(row, HistoryEntry):
                raise ValueError('Invalid host-approved inherited Controller history')
            trusted[row.record_id] = row
        # The run owns canonical observations across all Controller versions.
        # Candidates cannot make receipt-shaped JSON into trusted evidence.
        for row in self.context.host_execution.memory.history():
            trusted[row.record_id] = row
        if any(trusted.get(row.record_id) != row for row in state['history']):
            raise ValueError('Controller Memory version cites missing or altered host evidence')
        return deepcopy(tuple(state['history']))

    def version(self, *, version_id=None, model=None):
        """Capture a candidate of the current configuration, including policy and Memory."""
        policy = _policy(self.active.endpoint)
        capture = optional_method(policy, 'state')
        integration = getattr(self.active.endpoint, 'integration', None)
        if capture is None:
            raise TypeError('Controller versions require explicit policy state')
        version = replace(self.current_version, version_id=version_id or str(uuid4()),
            model=model if model is not None else self.current_version.model,
            policy_state=deepcopy(capture()),
            memory_state=deepcopy(integration.export_version()) if integration is not None else None)
        delivery = self._delivery()
        prior = self._version_deliveries.get(version.version_id)
        if prior is not None and prior != delivery:
            raise ValueError('A captured Controller version cannot move its delivery boundary')
        self._version_deliveries[version.version_id] = delivery
        return version

    def capture(self):
        return {'schema': 5, 'current_version': deepcopy(self.current_version), 'pending': deepcopy(self.pending),
                'archives': deepcopy(self.archives), 'adopted': deepcopy(self._adopted),
                'active': deepcopy(self.active.capture()), 'learning': self.learning.capture(),
                'prepared_models': deepcopy(self.prepared_models),
                'version_deliveries': deepcopy(self._version_deliveries),
                'pending_adoption_event': deepcopy(self._pending_adoption_event)}

    def _grant_prepared_model(self, model, training):
        """A readiness receipt authorizes this exact ref, preserving serving aliases."""
        target = self._prepared_model_target(model, training, prepared=True)
        authorize = optional_method(target, 'authorize_prepared_model')
        if authorize is not None:
            authorize(model)

    def _prepared_model_target(self, model, training, *, prepared=False):
        """Reject an unusable serving route before starting deployment effects."""
        session = getattr(self.context, 'model_session', None)
        backend = (session.backends.get(model.backend_id) if session is not None else
                   getattr(self.context, 'backends', {}).get(model.backend_id))
        from rsicomposer.backends.serving.chat_completions import RemoteModelBackend
        from rsicomposer.runtime.models.accounting import AccountedModelBackend
        transport = backend.transport if isinstance(backend, AccountedModelBackend) else backend
        if (not training.simulated and isinstance(transport, RemoteModelBackend)
                and training.base.model_id == model.model_id):
            from collections.abc import Mapping
            bindings = getattr(transport, 'models', None)
            if optional_method(transport, 'bind') is None or not isinstance(bindings, Mapping):
                raise ValueError('Revision-only weight adoption requires a revision-aware serving route; '
                                 'RemoteModelBackend must use a distinct prepared model alias')
            if prepared and model not in bindings:
                raise ValueError('Prepared Controller model has no registered serving alias')
        return session if session is not None else backend

    def restore(self, state):
        if (not isinstance(state, dict) or set(state) != {'schema', 'current_version', 'pending', 'archives',
                'adopted', 'active', 'learning', 'prepared_models', 'version_deliveries', 'pending_adoption_event'}
                or type(state['schema']) is not int or state['schema'] != 5):
            raise ValueError('Invalid Controller host state')
        from rsicomposer.loops.contracts.controller_version import ControllerVersion
        from rsicomposer.loops.contracts.training import TrainingResult
        from rsicomposer.runtime.state.controller_versions import load_controller_version
        version = state['current_version']
        if not isinstance(version, ControllerVersion):
            raise ValueError('Invalid active Controller version')
        pending = state['pending']
        if pending is not None:
            if (not isinstance(pending, dict) or set(pending) != {'ref', 'version', 'plan', 'training_result', 'adoption_id', 'delivery'}
                    or not isinstance(pending['version'], ControllerVersion)
                    or not isinstance(pending['adoption_id'], str) or not pending['adoption_id']
                    or not isinstance(pending['plan'], dict)
                    or any(not isinstance(key, str) or not key or value not in ('drain', 'cancel', 'handover')
                           for key, value in pending['plan'].items())
                    or pending['training_result'] is not None and not isinstance(pending['training_result'], TrainingResult)):
                raise ValueError('Invalid pending Controller adoption')
            if load_controller_version(self.context.artifacts, _version_ref(pending['ref'])) != pending['version']:
                raise ValueError('Pending Controller version differs from its artifact')
        deliveries = state['version_deliveries']
        execution = self.context.host_execution.state
        def valid_delivery(delivery):
            return (isinstance(delivery, dict) and set(delivery) == {'feedback', 'progress'}
                    and type(delivery['feedback']) is int
                    and 0 <= delivery['feedback'] <= len(execution.operation_results)
                    and type(delivery['progress']) is int
                    and 0 <= delivery['progress'] <= len(execution.operation_progress))
        if (not isinstance(deliveries, dict) or any(not isinstance(key, str) or not key or not valid_delivery(value)
                for key, value in deliveries.items()) or pending is not None and not valid_delivery(pending['delivery'])):
            raise ValueError('Invalid captured Controller delivery boundaries')
        event = state['pending_adoption_event']
        if event is not None and (not isinstance(event, dict) or set(event) != {'adoption_id', 'controller_version'}
                or any(not isinstance(value, str) or not value for value in event.values())
                or event['controller_version'] != version.version_id):
            raise ValueError('Invalid pending Controller adoption event')
        if (not isinstance(state['archives'], list)
                or any(not isinstance(entry, dict) or set(entry) != {'version', 'state'}
                       or not isinstance(entry['version'], ControllerVersion) or not isinstance(entry['state'], dict)
                       for entry in state['archives'])
                or not isinstance(state['adopted'], dict)
                or any(not isinstance(key, str) or not key or not isinstance(ref, ArtifactRef)
                       for key, ref in state['adopted'].items())):
            raise ValueError('Invalid Controller version archive')
        from rsicomposer.loops.contracts.model_service import ModelRef
        prepared = state['prepared_models']
        if (not isinstance(prepared, dict) or any(not isinstance(model, ModelRef)
                or not isinstance(entry, dict) or set(entry) != {'result', 'activation_key'}
                or not isinstance(entry['result'], TrainingResult) or entry['result'].model != model
                or not isinstance(entry['activation_key'], str) or not entry['activation_key']
                for model, entry in prepared.items())):
            raise ValueError('Invalid prepared Controller model inventory')
        from rsicomposer.runtime.models.activation import prepare_model_activation
        for model, entry in prepared.items():
            self._prepared_model_target(model, entry['result'])
            ready = prepare_model_activation(entry['result'], self.activator,
                                             idempotency_key=entry['activation_key'])
            if ready != model:
                raise ValueError('Recovered Controller deployment differs from its prepared model')
            self._grant_prepared_model(ready, entry['result'])
        self.prepared_models = deepcopy(prepared)
        candidate = self.active if version == self.current_version else self.build_controller(deepcopy(version))
        try:
            self._validate_execution(candidate.endpoint)
            candidate.restore(deepcopy(state['active']))
        except BaseException as error:
            if candidate is not self.active:
                try:
                    candidate.close()
                except BaseException as cleanup:
                    self._retired.append(candidate)
                    error.add_note('Recovered Controller cleanup failed: ' + str(cleanup))
            raise
        if candidate is not self.active:
            self._retired.append(self.active)
        self.active, self.current_version = candidate, deepcopy(version)
        self.pending, self.archives, self._adopted = deepcopy(state['pending']), deepcopy(state['archives']), deepcopy(state['adopted'])
        self._version_deliveries = deepcopy(deliveries)
        self._pending_adoption_event = deepcopy(event)
        self.learning.restore(state['learning'])
        self.context.host_execution.activate(candidate.endpoint, self.version_id, reset_delivery=False)
        self._close_retired()

    def validate_snapshot(self):
        self.learning.validate_snapshot()
        self.active.validate_snapshot()
        if self._retired:
            raise RuntimeError('Controller cleanup must finish before a portable snapshot')

    def _close_retired(self):
        failed, errors = [], []
        for instance in self._retired:
            try:
                instance.close()
            except BaseException as error:
                failed.append(instance)
                errors.append(error)
        self._retired = failed
        if errors:
            raise BaseExceptionGroup('Retired Controller cleanup failed', errors)

    def close(self):
        self.learning.close()
        self.context.host_execution.close()
        self._close_retired()
        self.active.close()


class ControllerLearning:
    def __init__(self, host, learner):
        self.host = host
        self.training = JobBook(learner)
        self.recorded, self.record_ids, self.candidates = set(), {}, {}
        self._closed = False

    def train(self, *, recipe_ref, experience_refs=(), data_refs=(), method='sft', idempotency_key):
        """Build a request against the active decision model for the training Lab."""
        request = DecisionTrainingRequest(self.host.active.endpoint.current_model(),
            tuple(experience_refs), recipe_ref, tuple(data_refs), method)
        return self.submit(request, idempotency_key=idempotency_key)

    def submit(self, request, *, idempotency_key):
        if self._closed:
            raise RuntimeError('Controller learning is closing; new training jobs are not admitted')
        if not isinstance(request, DecisionTrainingRequest):
            raise TypeError('Expected DecisionTrainingRequest')
        available = {row.record_id for row in self.host.context.experience.retrieve()}
        if any(ref not in available for ref in request.experience_refs):
            raise ValueError('Training references must identify visible experience records')
        return self.training.submit(request, idempotency_key=idempotency_key)

    def _normalize(self, outcome):
        result = outcome.result
        if outcome.status != JobStatus.SUCCEEDED or not isinstance(result, TrainingResult):
            return outcome
        request = self.training.request(outcome.job)
        if result.base != request.base or 'decision_maker' not in result.candidate.affected_roles:
            return outcome
        try:
            validated_metrics(result.metrics, 'training metrics')
        except InvalidMetrics:
            return replace(outcome, status=JobStatus.FAILED, result=None,
                simulated=outcome.simulated or result.simulated,
                error='Invalid training metrics: expected nonempty names and finite numeric values')
        return outcome

    def _record(self, outcome):
        request = self.training.request(outcome.job)
        result = outcome.result
        if outcome.status == JobStatus.SUCCEEDED:
            if not isinstance(result, TrainingResult) or result.base != request.base:
                raise ValueError('Training result does not match the submitted base')
            if 'decision_maker' not in result.candidate.affected_roles:
                raise ValueError('Training result does not target the Controller role')
        if outcome.terminal and outcome.job not in self.recorded:
            self.host.context.experience.record(ExperienceRecord(
                self.record_ids.setdefault(outcome.job, str(uuid4())), 'training_run', 'decision_maker', None,
                'Controller learning ' + outcome.status.value,
                outcome.simulated or bool(result and result.simulated),
                source_refs=request.experience_refs + tuple(ref.artifact_id for ref in request.data_refs)
                    + (tuple(ref.artifact_id for ref in result.candidate.changes.values()) if result else ()),
                details={'request': asdict(request), 'outcome': asdict(outcome),
                         'controller_version': self.host.version_id}))
            self.recorded.add(outcome.job)
        return outcome

    def poll(self, job):
        return self._record(self.training.poll(job, normalize=self._normalize))

    def cancel(self, job):
        return self._record(self.training.cancel(job, normalize=self._normalize))

    def candidate(self, job):
        """Publish a learned version; only an explicit host operation adopts it."""
        if job in self.candidates:
            return deepcopy(self.candidates[job])
        if self._closed:
            raise RuntimeError('Controller learning is closing; new candidates are not admitted')
        outcome = self.poll(job)
        if outcome.status != JobStatus.SUCCEEDED:
            raise RuntimeError('Training has no usable result: ' + outcome.status.value)
        result = outcome.result
        if self.host.active.endpoint.current_model() != result.base:
            raise ValueError('Training base is stale relative to the active Controller')
        if self.training.busy:
            raise RuntimeError('Cannot publish a model while another training job is active')
        version = self.host.version(version_id='trained-' + job.job_id, model=result.model)
        ref = save_controller_version(self.host.context.artifacts, version)
        self.candidates[job] = deepcopy(ref)
        return ref

    def training_result(self, ref):
        """Resolve only candidates published by this host's training service."""
        for job, candidate_ref in self.candidates.items():
            if candidate_ref == ref:
                return deepcopy(self.poll(job).result)
        return None

    def validate_snapshot(self):
        if self.training.busy:
            raise RuntimeError('Portable snapshots require Controller training jobs to finish')

    def close(self):
        """Cancel only jobs owned by this host, confirming every terminal outcome."""
        self._closed = True
        pending, errors = [], []
        for job in tuple(self.training._handles):
            try:
                # JobHandle emits cancellation once. A later close polls the
                # same request, retaining resources until completion is real.
                if not self.cancel(job).terminal:
                    pending.append(job)
            except BaseException as error:
                errors.append(error)
        if pending:
            errors.append(RuntimeError('Controller training cancellation is not yet confirmed: '
                                       + ', '.join(job.job_id for job in pending)))
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup('Controller learning cleanup failed', errors)

    def capture(self):
        from rsicomposer.runtime.state.job_books import _capture_book
        return {'schema': 2, 'training': deepcopy(_capture_book(self.training)), 'recorded': deepcopy(self.recorded),
                'record_ids': deepcopy(self.record_ids), 'candidates': deepcopy(self.candidates)}

    def restore(self, state):
        from rsicomposer.runtime.state.job_books import _restore_book
        if (not isinstance(state, dict) or set(state) != {'schema', 'training', 'recorded', 'record_ids', 'candidates'}
                or type(state['schema']) is not int or state['schema'] != 2):
            raise ValueError('Invalid Controller learning state')
        _restore_book(self.training, deepcopy(state['training']))
        self.recorded, self.record_ids, self.candidates = deepcopy(state['recorded']), deepcopy(state['record_ids']), deepcopy(state['candidates'])
