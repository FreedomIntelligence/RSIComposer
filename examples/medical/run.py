"""Independent assembly of the two default harnesses and Runner.

No legacy controller, dynamic imports, implicit model/data selection, or remote
side effects at import/build time. Deployment and inference adapters are explicit.
"""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import argparse
import json
import math
import time
from pathlib import Path

from rsisys.assembly import AssemblyContext, AssemblyRunner, ModuleAssembly, assemble_runner, module_assembly
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.requests import PromptSpec
from rsisys.loops.contracts.training import TrainingOrigin, ResumeMode, ModelActivationPending
from rsisys.loops.decision_making.action_space import (
    ModularDecisionActionSpace, OperationSpec, OperationCall, OperationResult,
)
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinitions, DirectiveDefinition, builtin_directives, training_directive, apply_candidate_directive,
)
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.commands import CommandHandler
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.sandbox import MockSandboxBackend
from rsisys.runtime.systems import SystemRegistry, SystemActivationRejected
from .evaluation import MedicalEvaluator, MedicalEvaluationError
from .task import encoded, prepare, samples, validate_training


class _ReadArtifact:
    spec = OperationSpec('read_artifact', 'Read visible train, prompt or development evidence bytes',
        {'type': 'object', 'properties': {'artifact_id': {'type': 'string'}},
         'required': ['artifact_id'], 'additionalProperties': False}, read_only=True)

    def validate(self, arguments):
        if set(arguments) != {'artifact_id'} or not isinstance(arguments['artifact_id'], str):
            raise ValueError('read_artifact requires artifact_id')
        return OperationCall('read_artifact', arguments)


class _PrepareData:
    spec = OperationSpec('prepare_data', 'Store example-format JSONL for an explicit training request',
        {'type': 'object', 'properties': {'text': {'type': 'string'}},
         'required': ['text'], 'additionalProperties': False})

    def validate(self, arguments):
        if set(arguments) != {'text'} or not isinstance(arguments['text'], str):
            raise ValueError('prepare_data requires text')
        samples(arguments['text'].encode())
        return OperationCall('prepare_data', arguments)


class _ModelBinding:
    def __init__(self, model):
        self.model = model

    def update_model(self, model):
        self.model = model


class _PromptCandidate(CommandHandler):
    def __init__(self, artifacts, systems, candidates):
        self.artifacts, self.systems, self.candidates = artifacts, systems, candidates

    def prepare(self, harness, state):
        if not isinstance(state.directive.payload, PromptSpec) or state.directive.input_refs:
            raise ValueError('Prompt candidate requires instructions and no data inputs')
        state.data['parent'] = self.systems.current

    def advance(self, harness, state, action_space):
        if state.cancel_requested:
            return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED, 'Prompt proposal cancelled', False)
        if state.request is None:
            parent = state.data['parent']
            ref = self.artifacts.put(state.directive.payload.instructions.encode(), kind='medical_prompt')
            state.request = Candidate('prompt-' + state.directive.directive_id, parent.model.revision,
                {'prompt': ref}, ('actor',), parent.version)
        self.systems.register(state.request)
        self.candidates[state.request.candidate_id] = state.request
        return Feedback(state.directive.directive_id, ExecutionStatus.COMPLETED,
            'Prompt proposed; evaluate before application', False,
            artifact_refs=tuple(state.request.changes.values()), candidates=(state.request,))


class _MedicalTraining(TrainingCommandHandler):
    def __init__(self, task, artifacts, systems, candidates, excluded, initializer, bind_inputs):
        self.task, self.artifacts, self.systems = task, artifacts, systems
        self.candidates, self.excluded = candidates, excluded
        self.initializer, self.bind_inputs = initializer, bind_inputs

    def prepare(self, harness, state):
        if state.directive.payload.apply_result:
            raise ValueError('Medical candidates must be evaluated before application')
        development = self.task.check()
        for ref in state.directive.input_refs:
            validate_training(samples(self.artifacts.get(ref)), (*development, *self.excluded))
        if not state.directive.input_refs:
            raise ValueError('Training requires explicit example-format JSONL artifacts')
        parent = self.systems.current
        origin = TrainingOrigin(parent.version, self.initializer)
        if state.directive.payload.origin not in (None, origin):
            raise ValueError('Medical training origin is host-owned: current parent, fixed initializer, restart only')
        if state.directive.payload.base not in (None, self.initializer):
            raise ValueError('Medical training must use the fixed training initializer')
        original = state.directive
        state.directive = replace(state.directive, payload=replace(state.directive.payload,
            base=origin.initial_model, origin=origin))
        # Preserve the public directive identity; only the prepared backend request
        # receives the explicit default origin.
        try:
            super().prepare(harness, state)
        finally:
            state.directive = original
        prepared = state.request
        bound = self.bind_inputs(prepared, self.artifacts)
        if replace(bound, recipe_ref=prepared.recipe_ref) != prepared:
            raise ValueError('Native input binding may change recipe_ref only; provenance must survive')
        state.request = bound

    def advance(self, harness, state, action_space):
        super().advance(harness, state, action_space)
        if state.feedback is not None and state.feedback.status == ExecutionStatus.COMPLETED:
            result = harness.candidate_result(state.feedback.candidates[0].candidate_id)
            candidate = result.candidate
            if candidate.parent_system != state.request.origin.parent_system:
                raise ValueError('Training candidate parent differs from the prepared origin')
            try:
                self.systems.register_training(result, origin=state.request.origin)
            except (ValueError, KeyError) as exc:
                return replace(state.feedback, status=ExecutionStatus.FAILED,
                    summary='Training output could not be registered as a system candidate', candidates=(),
                    error=f'{type(exc).__name__}: {exc}')
            self.candidates[candidate.candidate_id] = candidate


class MedicalApplicationRejected(ValueError):
    """Known application precondition failure; report it to the researcher."""


class _ApplyMedical(CommandHandler):
    def __init__(self, systems, evaluator, candidates, artifacts):
        self.systems, self.evaluator, self.candidates, self.artifacts = systems, evaluator, candidates, artifacts

    def prepare(self, harness, state):
        if state.directive.input_refs:
            raise ValueError('Medical application does not accept data inputs')
        ident = state.directive.payload.candidate_id
        state.request = self.candidates[ident]

    def _reject(self, state, reason):
        if 'rejection' not in state.data:
            ident = state.request.candidate_id
            result = self.evaluator.measurement(ident)
            ref = self.artifacts.put(encoded({'kind': 'application_rejection',
                'candidate_id': ident, 'incumbent': self.systems.current.version,
                'protocol_ref': self.evaluator.task.protocol_ref,
                'accepted': False, 'reason': reason,
                'source_refs': list(result.evidence_refs) if result else []}),
                kind='medical_interpretation')
            self.evaluator.evidence[ref.artifact_id] = ref
            state.data['rejection'] = Feedback(state.directive.directive_id, ExecutionStatus.FAILED,
                'Candidate application rejected by fixed medical acceptance protocol',
                result.simulated if result else state.directive.simulated,
                evidence_refs=(ref.artifact_id,), candidates=(state.request,), error=reason)
        return state.data['rejection']

    def advance(self, harness, state, action_space):
        if state.cancel_requested and not state.data.get('activation_started'):
            return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED, 'Application cancelled', False)
        if 'applied' not in state.data:
            if not state.data.get('activation_started'):
                ident = state.request.candidate_id
                try:
                    if state.request.parent_system != self.systems.current.version:
                        raise MedicalApplicationRejected('Candidate parent is stale; rebase and evaluate explicitly')
                    if not self.evaluator.completed_evaluation(ident):
                        raise MedicalApplicationRejected('Candidate has no successful completed development evaluation')
                    interpretation = self.evaluator.interpret(ident)
                    if not interpretation['accepted']:
                        reasons = []
                        if interpretation.get('selection_rule') == 'pareto':
                            reasons.append('requires at least one accuracy improvement and no accuracy regression')
                        elif interpretation.get('selection_rule') == 'primary_guardrail' and not interpretation['primary_improved']:
                            reasons.append('primary score did not strictly improve over the current incumbent')
                        if interpretation.get('selection_rule') == 'primary_guardrail' and interpretation['guardrails_enabled'] and not interpretation['guardrail_passed']:
                            reasons.append('fixed-anchor guardrail failed')
                        raise MedicalApplicationRejected('; '.join(reasons) or 'Rejected by the configured acceptance policy')
                except (MedicalApplicationRejected, MedicalEvaluationError) as exc:
                    return self._reject(state, str(exc))
                # Resolve local binding inputs before the externally visible
                # activation. Failed reads must leave the incumbent untouched.
                target = self.systems.get(ident)
                state.data['instructions'] = self.artifacts.get(target.components['prompt']).decode('utf-8')
                state.data['interpretation'] = interpretation
                # Activation may take effect before readiness/response is confirmed.
                # Cancellation after this point must reconcile this exact intent.
                state.data['activation_started'] = True
            state.phase = 'applying'
            try:
                state.data['applied'] = self.systems.activate(state.request.candidate_id,
                    idempotency_key=state.idempotency_key)
            except ModelActivationPending:
                return None
            except SystemActivationRejected as exc:
                return self._reject(state, str(exc))

        system = state.data['applied']
        # Binding changes only after the injected activator confirms readiness.
        harness.policy.update_model(system.model)
        harness.state.instructions = state.data['instructions']
        if 'evidence' not in state.data:
            state.data['evidence'] = self.artifacts.put(encoded({**state.data['interpretation'],
                'activation_confirmed': True, 'cancel_requested': state.cancel_requested}),
                kind='medical_interpretation').artifact_id
        return Feedback(state.directive.directive_id, ExecutionStatus.COMPLETED,
            ('Cancellation arrived after activation started; confirmed candidate applied' if state.cancel_requested
             else 'Candidate accepted by fixed development protocol and applied'),
            self.evaluator.measurement(system.version).simulated,
            evidence_refs=(state.data['evidence'],), candidates=(state.request,),
            adopted_candidate_ids=(state.request.candidate_id,))


@dataclass(frozen=True)
class RunSettings:
    objective: str
    max_runner_steps: int
    poll_interval_seconds: float = 1.0

    def __post_init__(self):
        if not self.objective.strip() or type(self.max_runner_steps) is not int or self.max_runner_steps < 1:
            raise ValueError('Explicit objective and positive runner step budget required')
        if (type(self.poll_interval_seconds) not in (int, float) or not math.isfinite(self.poll_interval_seconds)
                or self.poll_interval_seconds < 0):
            raise ValueError('poll_interval_seconds must be finite and nonnegative')


@dataclass(frozen=True)
class MedicalRunResult:
    status: str
    steps: tuple
    pending: object
    message: str


@dataclass
class MedicalAssembly:
    task: object
    runner: AssemblyRunner
    evaluator: MedicalEvaluator
    systems: SystemRegistry
    artifacts: object
    experience_store: object
    candidates: dict

    def close(self):
        self.runner.close()

    def __enter__(self):
        self.runner.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f'Medical assembly cleanup also failed: {cleanup_error}')

    def run(self, settings: RunSettings):
        if self.runner.closed:
            raise RuntimeError('Medical assembly is closed')
        try:
            anchor = self.evaluator.establish_anchor()  # Explicit run starts actual inference.
        except (MedicalEvaluationError, ModelActivationPending) as exc:
            return MedicalRunResult('needs_input', (), self.runner.exploration.pending(), str(exc))
        objective = (settings.objective + '\nFixed medical task: ' + json.dumps(asdict(self.task))
            + '\nEvaluation protocol: ' + self.task.protocol_ref
            + '\nFixed initial anchor: ' + json.dumps(asdict(anchor))
            + '\nPropose prompt or training candidates, evaluate, then apply accepted candidates. '
              'Raw outputs are evidence; explanations and hypotheses are interpretations. '
              'Training consumes explicit JSONL artifacts; prepare_data stores that text. '
              'For configure_prompt/evaluate/apply_candidate set input_ids=[]; only train consumes data refs. '
              'Default training restart initializer is fixed, independent of the incumbent. '
              'Use the existing train artifact id: ' + self.systems.get('initial').components['data'].artifact_id)
        steps = []
        for _ in range(settings.max_runner_steps):
            step = self.runner.step(objective)
            steps.append(step)
            if step.kind == 'finished':
                return MedicalRunResult('done', tuple(steps), None, 'Researcher finished or run policy stopped')
            if step.kind == 'waiting' and settings.poll_interval_seconds:
                time.sleep(settings.poll_interval_seconds)
        return MedicalRunResult('needs_input', tuple(steps), self.runner.exploration.pending(),
            'Runner step budget exhausted; retain this assembly and resume explicitly; pending work was not cancelled')


def build(task, *, artifacts, decision_backend, decision_model, initial_model,
          inference, system_activator, learner=None, trusted_exclusions=(),
          training_initializer=None, bind_training_inputs=None,
          run_policy=None, action_permissions=None, evaluator_factory=MedicalEvaluator, prepare_inference=None,
          modules=None):
    """Construct only. Adapter owners configure native devices, recipes and serving.

    inference(GenerationRequest)->GenerationResponse must route the requested
    candidate model, including before incumbent selection. system_activator uses
    the core SystemActivator protocol. Native learner owns its recipe/data contract;
    this example passes JSONL bytes without a universal source conversion layer.
    trusted_exclusions is operator-only and never put in researcher context.
    """
    modules = module_assembly(modules)
    if learner is not None and not callable(bind_training_inputs):
        raise TypeError('Training requires an explicit native input binding/validation callable')
    development = task.check()
    validate_training(task.train.read(), (*development, *trusted_exclusions))
    artifacts.put(encoded({'task': asdict(task), 'protocol_ref': task.protocol_ref}), kind='medical_frozen_task')
    prompt = artifacts.put(b'', kind='medical_prompt')
    train = artifacts.put(Path(task.train.path).read_bytes(), kind='training_data')
    initial = SystemVersion('initial', {'prompt': prompt, 'data': train}, initial_model)
    systems, candidates = SystemRegistry(initial, system_activator), {}
    evaluator = evaluator_factory(task, artifacts, systems, candidates, inference, prepare_inference=prepare_inference)
    required = ('establish_anchor', 'measurement', 'interpret', 'completed_evaluation',
                'submit', 'poll', 'cancel', 'result')
    if any(not callable(getattr(evaluator, name, None)) for name in required):
        raise TypeError('Medical evaluator factory returned an incomplete public contract')
    if evaluator.task != task or not isinstance(evaluator.evidence, dict):
        raise TypeError('Medical evaluator requires the frozen task and public evidence reference mapping')
    store = InMemoryExperienceStore()
    experience = BasicExperienceSystem(store)
    handlers = {'configure_prompt': _PromptCandidate(artifacts, systems, candidates),
                'apply_candidate': _ApplyMedical(systems, evaluator, candidates, artifacts)}
    definitions = [DirectiveDefinition(replace(d.spec, accepts_input_refs=False,
        description=('Propose a versioned prompt candidate without changing the incumbent' if d.spec.mode == 'configure_prompt'
                     else 'Evaluate a candidate using the frozen medical development protocol; input_ids must be []')),
        d.parser) for d in builtin_directives() if d.spec.mode in ('configure_prompt', 'evaluate')]
    application = apply_candidate_directive()
    definitions.append(DirectiveDefinition(replace(application.spec, accepts_input_refs=False), application.parser))
    # The domain fixes the protocol; a sole candidate is an unambiguous reference.
    # Multiple candidates always require an explicit choice by the researcher.
    def with_medical_defaults(definition):
        mode = definition.spec.mode
        schema = deepcopy(definition.spec.input_schema)
        schema['required'] = []
        schema['properties']['candidate_id']['description'] = (
            'May be omitted only when exactly one candidate exists; otherwise choose its exact ID.')
        schema['properties']['candidate_id']['minLength'] = 1
        if mode == 'evaluate':
            schema['properties']['protocol_ref'] = {'type': 'string',
                'enum': [task.protocol_ref], 'default': task.protocol_ref}
        def parse(arguments, available):
            fields = dict(arguments)
            if 'candidate_id' not in fields:
                ids = sorted({candidate.candidate_id for candidate in available})
                if len(ids) != 1:
                    raise ValueError(f'candidate_id is required: choose exactly one from {ids}')
                fields['candidate_id'] = ids[0]
            if mode == 'evaluate':
                fields.setdefault('protocol_ref', task.protocol_ref)
                if fields['protocol_ref'] != task.protocol_ref:
                    raise ValueError('Evaluation must use the frozen medical protocol')
            return definition.parser(fields, available)
        return DirectiveDefinition(replace(definition.spec, input_schema=schema,
            description=definition.spec.description +
                ' candidate_id defaults only for a sole candidate.' +
                (' The frozen protocol defaults automatically.' if mode == 'evaluate' else '')), parse)
    definitions = [with_medical_defaults(d) if d.spec.mode in ('evaluate', 'apply_candidate') else d
                   for d in definitions]
    if learner is not None:
        initializer = training_initializer or initial_model
        handlers['train'] = _MedicalTraining(task, artifacts, systems, candidates, tuple(trusted_exclusions),
            training_initializer or initial_model, bind_training_inputs)
        capabilities = learner.capabilities() if callable(getattr(learner, 'capabilities', None)) else None
        if capabilities is not None and ResumeMode.RESTART not in capabilities.resume_modes:
            raise ValueError('Medical training requires backend restart support')
        native_training = training_directive(capabilities)
        schema = deepcopy(native_training.spec.input_schema)
        # Initialization is an operator choice, not a model-authored object.
        schema['properties'].pop('origin')
        schema['required'] = [name for name in schema['required'] if name != 'origin']
        schema['properties']['apply_result'] = {'type': 'boolean', 'enum': [False], 'default': False}
        recipe_ref = getattr(learner, 'recipe_ref', None)
        if isinstance(recipe_ref, str) and recipe_ref:
            schema['properties']['recipe_ref'] = {'type': 'string', 'enum': [recipe_ref]}
        def parse_training(arguments, available):
            if 'origin' in arguments:
                raise ValueError('arguments.origin is host-owned; omit it to use the current parent and fixed initializer')
            if arguments.get('apply_result', False) is not False:
                raise ValueError('Medical training requires apply_result=false; evaluate before application')
            allowed_recipes = schema['properties']['recipe_ref'].get('enum')
            if allowed_recipes is not None and arguments.get('recipe_ref') not in allowed_recipes:
                raise ValueError('recipe_ref must name the configured native recipe')
            methods = schema['properties']['method'].get('enum')
            if methods is not None and arguments.get('method') not in methods:
                raise ValueError(f'method must be one of {methods}')
            spec = native_training.parser(arguments, available)
            return replace(spec, origin=TrainingOrigin(systems.current.version, initializer))
        definitions.append(DirectiveDefinition(replace(native_training.spec, input_schema=schema,
            description='Create a training candidate from explicit input_ids. The host binds the current parent '
                'and fixed restart initializer; origin is not a model argument. Evaluate before applying.'), parse_training))
    directives = DirectiveDefinitions(definitions)
    exploration = ExplorationLoop(DefaultExplorationHarness(_ModelBinding(initial_model), object(), experience,
        learner=learner, evaluator=evaluator, command_handlers=handlers), EchoActionSpace())
    def read_visible(arguments):
        visible = {r.artifact_id: r for r in initial.components.values()}
        visible.update(prepared_refs)
        visible.update(evaluator.evidence)
        for candidate in candidates.values():
            visible.update({r.artifact_id: r for r in candidate.changes.values()})
        ref = visible[arguments['artifact_id']]
        if not ref.uri.startswith(('artifact://', 'memory://')):
            return OperationResult('External component reference; an explicit resolver is required', (ref,), False,
                data={'reference': asdict(ref), 'requires_external_resolver': True,
                      'content': None})
        return OperationResult('Visible artifact content', (ref,), False,
                               data={'content': artifacts.get(ref).decode('utf-8')})
    prepared_refs = {}
    memory, agenda = DecisionMemory(), ResearchAgenda()
    research_tools = ResearchTools(agenda, memory, experience=experience)
    def prepare_data(arguments):
        ref = artifacts.put(arguments['text'].encode(), kind='training_data')
        prepared_refs[ref.artifact_id] = ref
        return OperationResult('Prepared JSONL training data', (ref,), False)
    tools = BoundDecisionTools({'prepare_data': prepare_data, 'read_artifact': read_visible,
                                **research_tools.bindings})
    tools.research_tools = research_tools
    tools.artifacts = artifacts
    # Publish initial data through the existing artifact context, not hidden filesystem access.
    decision = DefaultDecisionHarness(ModelDecisionPolicy(decision_backend, decision_model,
        directives=directives, max_validation_attempts=3), tools, experience, memory=memory, research_agenda=agenda)
    decision.state.operation_results.append(OperationResult('Frozen training data', (train,), False))
    decision_loop = DecisionMakingLoop(decision, ModularDecisionActionSpace(
        (_PrepareData(), _ReadArtifact(), *research_tools.definitions), directives=directives,
        permissions=action_permissions))
    context = AssemblyContext(experience, artifacts, {decision_model.backend_id: decision_backend},
        MockSandboxBackend(), {'decision_maker': decision_model, 'actor': initial_model},
        exploration_directives=directives)
    # Domain factories retain the frozen medical protocol. The common entry owns
    # optional feedback, state ports, run control and lifecycle for every launcher.
    modules = replace(modules,
        decision_factory=modules.decision_factory if modules.decision_factory is not None else
            (lambda ctx, lab: decision_loop),
        exploration_factory=modules.exploration_factory if modules.exploration_factory is not None else
            (lambda ctx: exploration))
    try:
        runner = assemble_runner(context, modules=modules, run_policy=run_policy)
    except BaseException as exc:
        try:
            context.close()
        except BaseException as cleanup_error:
            exc.add_note(f'Medical assembly cleanup also failed: {cleanup_error}')
        raise
    return MedicalAssembly(task, runner, evaluator, systems, artifacts, store, candidates)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only preparation/check and explicit medical model runs')
    parser.add_argument('command', choices=('prepare', 'check', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--backends', type=Path)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args(argv)
    task = prepare(json.loads(args.config.read_text()), base_directory=args.config.resolve().parent)
    if args.command == 'run':
        if not all((args.backends, args.settings, args.output_directory)):
            parser.error('run requires --backends, --settings, and --output-directory')
        from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
        from rsisys.runtime.model import RemoteModelBackend
        from rsisys.runtime.artifacts import FileArtifactStore
        from rsisys.runtime.systems import ReferenceSystemActivator
        from .evaluation import backend_inference
        config = json.loads(args.backends.read_text())
        actor = ModelRef(**config['initial_model'])
        settings = RunSettings(**json.loads(args.settings.read_text()))
        with build(task, artifacts=FileArtifactStore(args.output_directory),
            decision_backend=RemoteModelBackend(BackendBinding(**config['decision_binding'])),
            decision_model=ModelRef(**config['decision_model']), initial_model=actor,
            inference=backend_inference(RemoteModelBackend(BackendBinding(**config['actor_binding']))),
            system_activator=ReferenceSystemActivator(actor)) as assembly:
            result = assembly.run(settings)
            assembly.artifacts.put(encoded(asdict(result)), kind='medical_run_result')
        print(json.dumps({'status': result.status, 'message': result.message}, indent=2))
        return 0 if result.status == 'done' else 2
    print(json.dumps({'task': asdict(task), 'protocol_ref': task.protocol_ref}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
