"""Run model-directed research with external capabilities, permissions and budgets.

PYTHONPATH=src python experiments/qwen_toy/run.py --baseline-url http://HOST:18095 \
    --worker-url http://HOST:8766

Run prepare.py in the worker's run directory before starting worker.py. The
controller does not upload data. Both services must load the same base weights.
Baseline revision is operator-declared: its service does not attest a revision.
Research order and candidate acceptance are chosen by the model. Budget and
protocol validation never synthesize a fallback decision.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

if __package__:
    from .monitor import Monitor
    from .adapters import (Deadline, HTTPClient, PROTOCOL, RECIPE, RevisionBackend,
                           WorkerActivator, WorkerEvaluator, WorkerTrainer)
else:
    from monitor import Monitor
    from adapters import (Deadline, HTTPClient, PROTOCOL, RECIPE, RevisionBackend,
                          WorkerActivator, WorkerEvaluator, WorkerTrainer)

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.assembly import AssemblyContext, ModuleAssembly, assemble_runner
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinitions, apply_candidate_directive, builtin_directives, training_directive,
)
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.harness.tools.queries import QueryTools
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.loops.decision_making.harness.tools.workspace import WorkspaceTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import ActionProposal, EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.environment import Transition
from rsisys.loops.exploration.harness.learner.command import ApplyCandidateHandler, TrainingCommandHandler
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.provenance import RunSource
from rsisys.runtime.sandbox import MockSandboxBackend
from rsisys.runtime.run_policy import GoalRunPolicy, MetricGoal, ResearcherRunPolicy, TimeRunPolicy


OBJECTIVE = 'Improve parity JSON answers using bounded LoRA SFT rounds and heldout NLL evaluation.'
EXECUTION_PROMPT = 'Return only JSON with key "parity" and value "even" or "odd" for integer 113.'


@dataclass(frozen=True)
class ResearchBudget:
    max_trainings: int = 2
    max_directives: int = 16
    max_model_calls: int = 32
    max_output_tokens: int = 200
    max_context_chars: int = 20000

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError('Research budgets must be positive integers')


class BudgetedModel:
    """Count attempted researcher requests, including malformed-response retries."""
    def __init__(self, backend, budget):
        self.backend, self.budget, self.calls = backend, budget, 0

    def generate(self, request):
        # Bounded model view; the full evidence remains in memory/Experience/files.
        payload = json.loads(request.prompt)
        # Optional named-model ancestry is additive. Keep the old bounded wire
        # view when the extension is empty; populated ancestry is never omitted.
        for candidate in payload.get('candidates', []):
            if candidate.get('model_bases') == {}:
                candidate.pop('model_bases')
        omissions = {}
        def render():
            return json.dumps({**payload, 'context_omissions': omissions}, separators=(',', ':'))
        for field in ('history', 'experience', 'directive_provenance', 'memories', 'operation_results'):
            values = payload.get(field, [])
            while len(values) > 1 and len(render()) > self.budget.max_context_chars:
                values.pop(0)
                omissions[field] = omissions.get(field, 0) + 1
        if len(render()) > self.budget.max_context_chars:
            for field in ('history', 'experience'):
                for row in payload.get(field, []):
                    row['details'] = {'omitted': 'Context budget; query source record for full details'}
            for row in payload.get('operation_results', []):
                row['data'] = {'omitted': 'Context budget; repeat query with a smaller page'}
        prompt = render()
        if len(prompt) > self.budget.max_context_chars:
            raise TimeoutError('Researcher context-character budget exhausted')
        request = replace(request, prompt=prompt)
        if self.calls >= self.budget.max_model_calls:
            raise TimeoutError('Researcher model-call budget exhausted')
        self.calls += 1
        response = self.backend.generate(replace(request, max_output_tokens=min(
            request.max_output_tokens, self.budget.max_output_tokens)))
        self.last_response = response.text
        return response


class ResearchDecisionPolicy(ModelDecisionPolicy):
    """Model-selected research; only capabilities, authorization and budgets constrain actions."""
    def __init__(self, backend, base, catalog, monitor, action_space, *, budget, attempts=3,
                 review_hint=None):
        super().__init__(BudgetedModel(backend, budget), base, directives=catalog)
        self.monitor, self.action_space, self.budget = monitor, action_space, budget
        self.attempts, self.review_hint = attempts, review_hint
        self.trainings = self.directives_issued = 0
        self.phase, self.round, self.reviews = 'researching', 0, []
        self.metrics, self.candidate_id, self.selection = {}, None, 'not_selected'

    def decide(self, context):
        remaining = {'trainings': self.budget.max_trainings - self.trainings,
                     'directives': self.budget.max_directives - self.directives_issued,
                     'model_calls': self.budget.max_model_calls - self.backend.calls}
        instruction = (
            'You are the researcher. Choose your own research steps and stopping point. '
            'You may inspect evidence, execute the current Actor, train, evaluate candidates repeatedly, '
            'configure its prompt, or apply a candidate. There is no prescribed order or acceptance threshold. '
            'Assess evidence and explain your choices in strategy/reason. Finish is allowed at any point. '
            'Finish ends the current research cycle; run_control determines whether the run ends or restarts. '
            f'The installed training backend supports method=sft and recipe_ref={RECIPE}; '
            f'the evaluation protocol is {PROTOCOL}. Its metrics compare candidate and parent NLL; '
            'lower NLL is better. The backend owns its training/evaluation data; empty input_ids are valid. '
            'Use the actual mode schemas and available tools; do not invent capabilities or candidate IDs. '
            'Model application must satisfy version/readiness checks. '
            'Workspace roots, if installed: run (this experiment), source (framework), example (toycase). '
            'Set review_after_seconds yourself when a running experiment needs monitoring. '
            f'An optional monitoring interval suggestion is {self.review_hint}; it is not mandatory. '
            f'Remaining resource budget: {json.dumps(remaining)}. '
            'Return only one compact JSON object.'
        )
        modes = tuple(mode for mode in context.directive_modes
                      if (mode != 'train' or remaining['trainings'] > 0) and remaining['directives'] > 0)
        hint = ''
        for attempt in range(self.attempts):
            self.monitor.event('model_decision_requested', step=attempt + 1)
            try:
                decision = super().decide(replace(context, directive_modes=modes,
                    objective=context.objective + '\n' + instruction + hint))
                space = authorized_space(self.action_space)
                if decision.kind == 'invoke':
                    space.validate(decision.operation)
                elif decision.kind == 'explore':
                    space.validate_directive(decision.directive)
                    decision = replace(decision, directive=replace(decision.directive, objective=context.objective))
                    self.directives_issued += 1
                    if decision.directive.mode == 'train':
                        self.trainings += 1
                        self.round = self.trainings
                    self.phase = decision.directive.mode
                    self.monitor.stage(self.phase)
                elif decision.kind == 'wait':
                    raise ValueError('No external research jobs are installed; use finish or an available capability')
                self.monitor.decision(decision, context, self.round)
                self.monitor.event('model_decision_validated', kind=decision.kind)
                return decision
            except (ValueError, KeyError, TypeError, PermissionError) as exc:
                print(json.dumps({'event': 'decision_validation_error', 'attempt': attempt + 1,
                                  'error_type': type(exc).__name__, 'error_summary': str(exc)[:500],
                                  'response_excerpt': getattr(self.backend, 'last_response', '')[:1200]}), flush=True)
                self.monitor.event('model_decision_rejected', step=attempt + 1)
                hint = '\nPrevious response validation error (data): ' + json.dumps(str(exc)[:500])
        raise ValueError('Model failed bounded decision validation attempts')

    def review(self, context):
        from rsisys.loops.contracts.coordination import ReviewDecision
        from rsisys.loops.decision_making.harness.decision import Decision
        progress = context.execution.progress
        if progress is None or progress.terminal:
            raise ValueError('Review requires an active experiment')
        for attempt in range(self.attempts):
            self.monitor.event('training_review_requested', source='llm', round=self.round,
                               directive_id=progress.directive_id, attempt=attempt + 1)
            try:
                reviewed = super().review(context)
                if isinstance(reviewed, Decision):
                    return reviewed
                if not isinstance(reviewed, ReviewDecision):
                    raise TypeError('Expected an operation or ReviewDecision')
                entry = {'source': 'llm', 'round': self.round, 'action': reviewed.action,
                         'mode': progress.directive.mode,
                         'next_review_seconds': reviewed.next_review_seconds,
                         'directive_id': progress.directive_id,
                         'job_id': progress.job.job_id if progress.job else None,
                         'phase': progress.phase, 'elapsed_seconds': progress.elapsed_seconds,
                         'decision_revision': self.model.revision,
                         'actor_revision': context.execution.model.revision if context.execution.model else None}
                self.reviews.append(entry)
                self.monitor.review(reviewed, context, self.round)
                self.monitor.event('training_review_completed', **entry)
                return reviewed
            except (ValueError, KeyError, TypeError):
                self.monitor.event('training_review_rejected', source='llm', round=self.round,
                                   directive_id=progress.directive_id, attempt=attempt + 1)
        raise ValueError('Model failed bounded review validation attempts')

    def feedback(self, directive, feedback):
        # Observations update evidence; they never choose the next research action.
        if feedback.simulated:
            raise ValueError('A real-model run received simulated feedback')
        if feedback.candidates:
            self.candidate_id = feedback.candidates[-1].candidate_id
        if directive.mode == 'evaluate':
            self.metrics = dict(feedback.quality_metrics)
        if directive.mode == 'apply_candidate' and feedback.status == ExecutionStatus.COMPLETED:
            self.selection = 'accepted'
        self.phase = 'researching'


class ParityActorPolicy:
    """The model returns the task's JSON answer; echo merely transports that text."""
    def __init__(self, backend, model):
        self.backend, self.model = backend, model

    def act(self, context):
        response = self.backend.generate(GenerationRequest('actor', self.model, context.observation,
            system='Answer the parity task with only the requested JSON. ' + context.instructions, max_output_tokens=80))
        return ActionProposal('echo', {'text': response.text}, False, response.model, response.usage)

    def update_model(self, model):
        if model.backend_id != self.model.backend_id:
            raise ValueError('Actor backend cannot change')
        self.model = model


class ParityEnvironment:
    def __init__(self, artifacts):
        self.artifacts, self.answer = artifacts, None

    def reset(self, directive):
        return EXECUTION_PROMPT

    def step(self, action):
        value = json.loads(action.text)
        if not isinstance(value, dict) or set(value) != {'parity'} or value['parity'] not in ('even', 'odd'):
            raise ValueError('Actor output must be a parity JSON object')
        self.answer = value
        ref = self.artifacts.put(json_bytes(value), kind='execution_output')
        return Transition(action.text, True, False, reward=float(value['parity'] == 'odd'), artifacts=(ref,))

    def close(self):
        pass


def make_run_policy(args):
    if args.run_mode == 'time':
        return TimeRunPolicy(args.run_seconds, restart_delay_seconds=args.restart_delay_seconds)
    if args.run_mode == 'goal':
        goal = MetricGoal(args.goal_metric, args.goal_threshold, PROTOCOL,
                          higher_is_better=args.goal_direction == 'higher', allow_simulated=False)
        return GoalRunPolicy(goal, restart_delay_seconds=args.restart_delay_seconds,
                             max_seconds=args.run_seconds)
    return ResearcherRunPolicy()


def run(args):
    root = args.run_dir.resolve()
    source = RunSource(root.name, getattr(args, 'parent_run_id', None))
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ('run.json', 'result.json', 'experience.sqlite3')):
        raise ValueError('Run directory already has controller records; use a fresh directory')
    lock = root / '.controller.lock'
    lock_fd = lock.open('x')
    store = None
    context = None
    try:
        atomic_write(root / 'run.json', json_bytes({**source.to_dict(), 'objective': OBJECTIVE}))
        deadline = Deadline(args.timeout)
        baseline = HTTPClient(args.baseline_url, deadline, timeout=args.request_timeout)
        worker = HTTPClient(args.worker_url, deadline, timeout=args.request_timeout)
        base = ModelRef('qwen-toy', args.model_name, args.base_revision)
        monitor = Monitor(root, base, OBJECTIVE, rounds=args.rounds, source=source)
        cancel_attempted, cancel_requested, cancellation_confirmed = False, False, False
        policy_cancelled_jobs = set()

        def notify(event, **details):
            nonlocal cancel_attempted, cancel_requested
            if event == 'job' and details.get('status') == 'cancel_requested':
                cancel_attempted = cancel_requested = True
                policy_cancelled_jobs.add(details['job_id'])
            monitor.event(event, **details)

        models = RevisionBackend(baseline, worker, base, notify=notify)
        trainer = WorkerTrainer(worker, models, notify=notify)
        evaluator = WorkerEvaluator(worker, models, notify=notify)
        builtin = {definition.spec.mode: definition for definition in builtin_directives()}
        catalog = DirectiveDefinitions((training_directive(), builtin['evaluate'],
                                        apply_candidate_directive(), builtin['execute'], builtin['configure_prompt']))
        store = SQLiteExperienceStore(root / 'experience.sqlite3')
        experience = BasicExperienceSystem(store, run_id=root.name)
        actor_policy = ParityActorPolicy(models, base)
        environment = ParityEnvironment(FileArtifactStore(root / 'artifacts'))
        exploration = DefaultExplorationHarness(actor_policy, environment, experience, learner=trainer,
            evaluator=evaluator, model_activator=WorkerActivator(models),
            memory=ExplorationMemory(root / 'exploration_memory.json'),
            command_handlers={'train': TrainingCommandHandler(), 'apply_candidate': ApplyCandidateHandler()})
        exploration_loop = ExplorationLoop(exploration, EchoActionSpace())
        queries = QueryTools(exploration_loop, artifacts=environment.artifacts)
        permissions = (ActionPermissions.from_dict(json.loads(args.permissions.read_text()))
                       if args.permissions else ActionPermissions.unrestricted())
        workspace = WorkspaceTools({'run': root, 'source': Path(__file__).resolve().parents[2] / 'src',
                                    'example': Path(__file__).resolve().parent}, max_bytes=2048, max_output_bytes=8192)
        memory = DecisionMemory(root / 'decision_memory.json')
        agenda = ResearchAgenda(root / 'decision_research.json')
        research = ResearchTools(agenda, memory, experience=experience)
        definitions = (*queries.definitions, *workspace.definitions, *research.definitions)
        bindings = {**queries.bindings, **workspace.bindings, **research.bindings}
        space = ModularDecisionActionSpace(definitions, directives=catalog, permissions=permissions)
        budget = ResearchBudget(args.rounds, args.max_directives, args.max_model_calls,
                                max_output_tokens=args.max_output_tokens,
                                max_context_chars=args.max_context_chars)
        policy = ResearchDecisionPolicy(models, base, catalog, monitor, space, budget=budget,
                                        review_hint=args.review_after_seconds)
        decision = DefaultDecisionHarness(policy, BoundDecisionTools(bindings), experience,
            memory=memory, research_agenda=agenda, memory_limit=16, experience_limit=16)
        context = AssemblyContext(experience, environment.artifacts, {base.backend_id: models},
            MockSandboxBackend(), {'decision_maker': base, 'actor': base}, data_dir=root,
            exploration_directives=catalog, action_permissions=permissions)
        context.add_cleanup(environment.close)
        runner = assemble_runner(context, modules=ModuleAssembly(
            decision_factory=lambda ctx, lab: DecisionMakingLoop(decision, space),
            exploration_factory=lambda ctx: exploration_loop),
            run_policy=make_run_policy(args), cleanup_timeout_seconds=30)
        status, error_code, remote_may_run = 'running', None, False
        run_stop = None
        monitor.data['run_mode'] = args.run_mode
        service_devices = {}
        round_results, experiments = [], []
        started = time.monotonic()
        try:
            health = baseline.call('/health')
            service_devices['baseline'] = health.get('device')
            if health.get('status') != 'ok' or health.get('model') != base.model_id:
                raise ValueError('Baseline health/model check failed')
            worker_health = worker.call('/health')
            service_devices['worker'] = worker_health.get('device')
            if worker_health.get('ready') is not True:
                raise ValueError('Worker health check failed')
            if worker_health.get('max_rounds', 1) < args.rounds:
                raise ValueError('Worker does not support the requested round count')
            monitor.stage(policy.phase)
            for _ in range(args.max_steps):
                deadline.remaining()
                result = runner.step(OBJECTIVE)
                monitor.sync_research_attempts(decision.state.decision_records)
                # A policy stop can wrap the last completed tick as finished and
                # omit its directive_id. Preserve that feedback in the monitor too.
                has_feedback = result.event.get('feedback') is not None
                monitor.advance(replace(result, kind='completed',
                    directive_id=result.event['feedback']['directive_id']) if has_feedback else result)
                monitor.data['models']['actor']['revision'] = actor_policy.model.revision
                if result.event.get('run_stop') is not None:
                    run_stop = result.event['run_stop']
                    monitor.data['run_stop'] = run_stop
                    monitor.event('run_stop', **run_stop)
                if has_feedback:
                    directive, feedback = decision.state.last_directive, decision.state.last_feedback
                    policy.feedback(directive, feedback)
                    experiments.append({'directive_id': directive.directive_id, 'mode': directive.mode,
                                        'strategy': directive.strategy, 'status': feedback.status.value,
                                        'candidate_ids': [c.candidate_id for c in feedback.candidates],
                                        'metrics': {**feedback.training_metrics, **feedback.quality_metrics,
                                                    **feedback.execution_metrics}})
                    for ident, training in models.candidates.items():
                        evaluation = decision.state.candidate_evaluations.get(ident)
                        metrics = evaluation.quality_metrics if evaluation else training.metrics
                        monitor.candidates[ident] = {
                            'candidate_id': ident, 'base_revision': training.base.revision,
                            'revision': training.model.revision,
                            'status': 'applied' if ident in exploration.state.applied_candidates else 'proposed',
                            'metrics': dict(metrics),
                        }
                    if directive.mode == 'execute':
                        ident = policy.candidate_id
                        training = models.candidates.get(ident)
                        row = {'round': len(round_results) + 1,
                               'base_revision': training.base.revision if training else base.revision,
                               'candidate_id': ident, 'candidate_revision': training.model.revision if training else None,
                               'selection': policy.selection, 'metrics': dict(policy.metrics),
                               'actor_revision': actor_policy.model.revision,
                               'applied': ident in exploration.state.applied_candidates,
                               'answer': environment.answer,
                               'answer_correct': environment.answer == {'parity': 'odd'},
                               'training_review_count': len(policy.reviews)}
                        round_results.append(row)
                        atomic_write(root / f'round-{len(round_results)}.json', json_bytes(row))
                        monitor.data['round_results'] = list(round_results)
                    monitor.data['workflow'].update(round=policy.trainings, rounds=args.rounds,
                                                     budget=asdict(budget))
                    monitor.stage(policy.phase)
                else:
                    monitor.event('progress', status=result.kind)
                if result.kind == 'finished':
                    remote_may_run = (not run_stop['shutdown_complete'] or run_stop['cleanup_timed_out']
                                      or bool(run_stop['unresolved_directives'])
                                      or run_stop['unresolved_research_job'] is not None)
                    status = ('timed_out' if remote_may_run else
                              'stopped' if run_stop['reason'] == 'budget_exhausted' else 'completed')
                    if (status in ('completed', 'stopped') and args.require_training_review
                            and not any(r['mode'] == 'train' for r in policy.reviews)):
                        raise ValueError('Requested verification requires a real training review')
                    break
                if result.kind == 'waiting':
                    time.sleep(min(args.poll_interval, deadline.remaining()))
            else:
                raise TimeoutError('Controller step budget exhausted')
        except (Exception, KeyboardInterrupt) as exc:
            print(json.dumps({'event': 'controller_error', 'error_type': type(exc).__name__,
                              'error_summary': str(exc)[:500]}), flush=True)
            status = 'timed_out' if isinstance(exc, TimeoutError) else 'failed'
            error_code = type(exc).__name__
            # Cleanup gets a separate five-second budget. Do not claim cancellation
            # until /jobs confirms it; a running GPU job may stop cooperatively.
            cleanup = HTTPClient(args.worker_url, Deadline(5), timeout=5)
            cancellation_statuses = []
            for backend in (trainer, evaluator):
                if backend.uncertain_submission:
                    try:
                        # Same frozen request and key: recover the original job
                        # before cancelling, within the shared five-second budget.
                        backend.recover_submission(cleanup)
                    except Exception:
                        monitor.event('submission_recovery_unconfirmed', kind=backend.kind)
                remote_may_run = remote_may_run or backend.uncertain_submission
                for job in backend.requests:
                    if job in backend.outcomes:
                        continue
                    cancel_attempted = True
                    pending = True
                    try:
                        cleanup.call('/cancel', {'job_id': job.job_id})
                        cancel_requested = True
                        monitor.event('job', job_id=job.job_id, kind=backend.kind, status='cancel_requested')
                        state = cleanup.call('/jobs/' + job.job_id)
                        if state.get('job_id') != job.job_id:
                            raise ValueError('Cancellation response job mismatch')
                        cancellation_statuses.append(state.get('status'))
                        pending = state.get('status') not in ('succeeded', 'failed', 'cancelled')
                        monitor.event('job', job_id=job.job_id, kind=backend.kind,
                                      status=state.get('status', 'cancel_requested'))
                    except Exception:
                        monitor.event('cancellation_unconfirmed', job_id=job.job_id)
                    remote_may_run = remote_may_run or pending
            cancellation_confirmed = (cancel_requested and not remote_may_run
                                      and all(value == 'cancelled' for value in cancellation_statuses))
        if policy_cancelled_jobs:
            cancellation_confirmed = (not remote_may_run and all(
                monitor.jobs.get(ident, {}).get('status') == 'cancelled' for ident in policy_cancelled_jobs))
        # A hard controller deadline may interrupt the Runner's cleanup phase.
        # Its last unconfirmed stop must remain visible even if fallback cleanup ran.
        if run_stop is not None and not run_stop['shutdown_complete']:
            remote_may_run = True
            cancellation_confirmed = False
        monitor.data['models']['actor']['revision'] = actor_policy.model.revision
        goal_reached = run_stop is not None and run_stop['reason'] == 'goal_reached'
        stop_reason = run_stop['reason'] if run_stop is not None else None
        monitor.data.update(run_stop=run_stop, stop_reason=stop_reason, goal_reached=goal_reached,
                            remote_may_run=remote_may_run, remote_job_may_still_run=remote_may_run)
        result = {
            'schema_version': 1, **source.to_dict(), 'status': status, 'simulated': False,
            'run_mode': args.run_mode, 'run_stop': run_stop, 'stop_reason': stop_reason,
            'goal_reached': goal_reached,
            'remote_may_run': remote_may_run,
            'model_name': args.model_name, 'service_devices': service_devices,
            'rounds_requested': args.rounds, 'rounds_completed': len(round_results),
            'round_results': round_results,
            'training_reviews': policy.reviews, 'training_review_source': 'llm',
            'training_review_required': args.require_training_review,
            'phase': 'finished' if status in ('completed', 'stopped') else monitor.phase,
            'selection': policy.selection, 'metrics': policy.metrics,
            'candidate_id': policy.candidate_id, 'applied': bool(exploration.state.applied_candidates),
            'base_revision': base.revision, 'actor_revision': actor_policy.model.revision,
            'decision_revision': policy.model.revision, 'baseline_revision_verified': False,
            'answer': environment.answer, 'answer_correct': environment.answer == {'parity': 'odd'},
            'completed_directives': decision.state.completed,
            'duration_seconds': round(time.monotonic() - started, 3),
            'error_code': error_code, 'remote_job_may_still_run': remote_may_run,
            'cancel_attempted': cancel_attempted, 'cancel_requested': cancel_requested,
            'cancellation_confirmed': cancellation_confirmed,
            'criterion': 'model-selected research and acceptance; no fixed workflow',
            'experiments': experiments, 'budget': asdict(budget),
            'model_calls': policy.backend.calls, 'training_submissions': policy.trainings,
            'decision_source': 'ModelDecisionPolicy; capability, permission and budget checks only',
        }
        atomic_write(root / 'result.json', json_bytes(result))
        monitor.stage(result['phase'], status=status)
        return result
    finally:
        try:
            if context is not None:
                context.close()
        finally:
            if context is None or not context.cleanup_pending:
                if store is not None:
                    store.close()
                lock_fd.close()
                lock.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-url', required=True)
    parser.add_argument('--worker-url', required=True)
    parser.add_argument('--model-name', default='Qwen3.5-9B')
    parser.add_argument('--base-revision', default='baseline')
    parser.add_argument('--max-trainings', '--rounds', dest='rounds', type=int, choices=(1, 2), default=2,
                        help='Training submission budget, not a prescribed number of rounds')
    parser.add_argument('--max-directives', type=int, default=16)
    parser.add_argument('--max-model-calls', type=int, default=32)
    parser.add_argument('--max-output-tokens', type=int, default=200,
                        help='Positive researcher output-token limit per model call')
    parser.add_argument('--max-context-chars', type=int, default=20000)
    parser.add_argument('--permissions', type=Path, help='External ActionPermissions JSON; default all installed capabilities')
    parser.add_argument('--run-dir', type=Path, default=Path('runs/qwen-toy'))
    parser.add_argument('--parent-run-id', help='Declare ancestry only; does not restore model or memory state')
    parser.add_argument('--run-mode', choices=('researcher', 'time', 'goal'), default='researcher')
    parser.add_argument('--run-seconds', type=float,
                        help='RunPolicy duration: required for time; optional maximum for goal')
    parser.add_argument('--goal-metric', help=f'Evaluation metric under protocol {PROTOCOL}')
    parser.add_argument('--goal-threshold', type=float)
    parser.add_argument('--goal-direction', choices=('higher', 'lower'), default='higher',
                        help='Goal comparison: higher >= threshold (default), lower <= threshold')
    parser.add_argument('--restart-delay-seconds', type=float, default=1,
                        help='Positive pause before restarting a finished research cycle')
    parser.add_argument('--timeout', type=float, default=3600,
                        help='Hard controller/service deadline, including policy cleanup; plus at most 5s emergency cancellation')
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--review-after-seconds', type=float, default=2,
                        help='Suggest a monitoring interval; the model decides whether and when to review')
    parser.add_argument('--require-training-review', action='store_true',
                        help='Fail verification if a round completes without an LLM training review')
    parser.add_argument('--poll-interval', type=float, default=2)
    parser.add_argument('--max-steps', type=int, default=4096)
    args = parser.parse_args(argv)
    for name in ('timeout', 'request_timeout', 'poll_interval', 'review_after_seconds', 'restart_delay_seconds'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be finite and positive')
    if min(args.max_steps, args.max_directives, args.max_model_calls,
           args.max_output_tokens, args.max_context_chars) < 1:
        parser.error('step, directive, model-call, output-token and context-character budgets must be positive')
    if args.run_seconds is not None and (not math.isfinite(args.run_seconds) or args.run_seconds <= 0):
        parser.error('--run-seconds must be finite and positive')
    if args.run_mode == 'time' and args.run_seconds is None:
        parser.error('--run-mode time requires --run-seconds')
    if args.run_mode == 'researcher' and args.run_seconds is not None:
        parser.error('--run-seconds applies only to time or goal mode')
    if args.goal_threshold is not None and not math.isfinite(args.goal_threshold):
        parser.error('--goal-threshold must be finite')
    if args.run_mode == 'goal':
        if not args.goal_metric or not args.goal_metric.strip() or args.goal_threshold is None:
            parser.error('--run-mode goal requires --goal-metric and --goal-threshold')
    elif args.goal_metric is not None or args.goal_threshold is not None or args.goal_direction != 'higher':
        parser.error('goal options require --run-mode goal')
    return args


if __name__ == '__main__':
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result['status'] in ('completed', 'stopped') else 1)
