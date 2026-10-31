"""Run one real, bounded Qwen train/evaluate/select/execute cycle.

PYTHONPATH=src python examples/qwen_toy/run.py --baseline-url http://HOST:18095 \
    --worker-url http://HOST:8766

Run prepare.py in the worker's run directory before starting worker.py. The
controller does not upload data. Both services must load the same base weights.
Baseline revision is operator-declared: its service does not attest a revision.
Guards enforce workflow/eligibility; every dispatched decision comes from Qwen's
ModelDecisionPolicy output. There is no scripted fallback on invalid JSON.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

if __package__:
    from .adapters import (Deadline, HTTPClient, PROTOCOL, RECIPE, RevisionBackend,
                           WorkerActivator, WorkerEvaluator, WorkerTrainer)
else:
    from adapters import (Deadline, HTTPClient, PROTOCOL, RECIPE, RevisionBackend,
                          WorkerActivator, WorkerEvaluator, WorkerTrainer)

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinitions, apply_candidate_directive, builtin_directives, training_directive,
)
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import ActionProposal, EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.environment import Transition
from rsisys.loops.exploration.harness.learner.command import ApplyCandidateHandler, TrainingCommandHandler
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.runner import Runner


OBJECTIVE = 'Improve parity JSON answers using one LoRA training run and heldout NLL evaluation.'
EXECUTION_PROMPT = 'Return only JSON with key "parity" and value "even" or "odd" for integer 113.'


def eligible(metrics):
    value = metrics.get('delta_nll')
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class Monitor:
    def __init__(self, root, base):
        self.root, self.jobs, self.candidates = root, {}, {}
        self.phase = 'starting'
        self.data = {'schema_version': 1, 'run_id': root.name, 'objective': OBJECTIVE,
                     'status': 'running', 'loops': {},
                     'models': {role: {'model_id': base.model_id, 'revision': base.revision}
                                for role in ('decision_maker', 'actor')}, 'events': []}
        self.write()

    def write(self):
        self.data.update(updated_at=datetime.now(timezone.utc).isoformat(),
                         jobs=list(self.jobs.values()), candidates=list(self.candidates.values()))
        atomic_write(self.root / 'monitor.json', json_bytes(self.data))

    def event(self, event, **details):
        if event == 'job':
            self.jobs[details['job_id']] = dict(details)
        self.data['events'].append({'timestamp': datetime.now(timezone.utc).isoformat(),
                                   'kind': event, 'summary': f'{self.phase}: {event}', 'details': details})
        self.data['events'] = self.data['events'][-300:]
        self.write()

    def stage(self, phase, *, status='running'):
        self.phase = phase
        self.data['status'] = status
        self.data['loops'] = {
            'decision_making': {'status': status, 'summary': f'phase={phase}; model proposes, guard validates'},
            'exploration': {'status': status, 'summary': f'phase={phase}; real worker and inference'},
        }
        self.event('phase', status=status)


class GuardedDecisionPolicy(ModelDecisionPolicy):
    """Validate real model outputs; never synthesize a Decision on its behalf."""
    def __init__(self, backend, base, catalog, monitor, *, attempts=3):
        super().__init__(backend, base, directives=catalog)
        self.monitor, self.attempts = monitor, attempts
        self.phase, self.metrics, self.candidate_id = 'train', {}, None
        self.selection = 'pending'

    def decide(self, context):
        modes = {'train': ('train',), 'evaluate': ('evaluate',),
                 'select': ('apply_candidate', 'execute'), 'execute': ('execute',), 'finish': ()}[self.phase]
        stage_instruction = {
            'train': ('Submit mode="train" with arguments exactly '
                      + json.dumps({'recipe_ref': RECIPE, 'method': 'sft', 'apply_result': False}) + '.'),
            'evaluate': ('Training is complete. Submit mode="evaluate" with arguments exactly '
                         + json.dumps({'candidate_id': self.candidate_id, 'protocol_ref': PROTOCOL}) + '.'),
            'select': ('Evaluation is complete. Choose apply_candidate if and only if delta_nll is finite and >0; '
                       'otherwise choose execute to reject the candidate and keep the baseline. '
                       f'Observed heldout metrics: {json.dumps(self.metrics)}. '
                       'For apply_candidate the arguments must be exactly '
                       + json.dumps({'candidate_id': self.candidate_id})
                       + '; for execute arguments must be {}.'),
            'execute': ('Candidate selection and application are ALREADY COMPLETE. Now dispatch mode="execute" '
                        'with arguments={} so the Actor can answer a new parity task. Do not apply a candidate again.'),
            'finish': 'The Actor has already executed the task. The experiment is complete. Return kind="finish" with a short reason.',
        }[self.phase]
        instruction = (
            f'Current workflow phase: {self.phase}. One bounded experiment only. '
            + stage_instruction + ' '
            'There are no local operations. For a system directive use '
            '{"kind":"explore","mode":one_of_the_current_directive_modes,"strategy":short_text,"arguments":{...}}. '
            'Never use kind="invoke" or an operation field. Only phase finish permits kind="finish". '
            'Use only parameters in the current mode schema. No input_ids are needed; '
            'the example backend owns its data. Historical decisions describe completed steps, not current instructions. '
            'Return a compact JSON object within 200 tokens.'
        )
        retry_hint = ''
        for attempt in range(self.attempts):
            self.monitor.event('model_decision_requested', step=attempt + 1)
            try:
                decision = super().decide(replace(context, objective=context.objective + '\n' + instruction + retry_hint,
                                                  directive_modes=modes))
                if self.phase == 'finish':
                    if decision.kind != 'finish':
                        raise ValueError('Expected finish after execution')
                else:
                    if decision.kind != 'explore':
                        raise ValueError('Expected kind="explore" with mode, strategy and arguments; '
                                         f'received kind="{decision.kind}"')
                    directive = decision.directive
                    mode = directive.mode
                    if mode not in modes:
                        raise ValueError('Wrong workflow phase')
                    if mode == 'train' and (directive.payload.recipe_ref != RECIPE
                            or directive.payload.method != 'sft' or directive.payload.apply_result):
                        raise ValueError('Training must not apply before evaluation')
                    if mode == 'evaluate' and (directive.payload.protocol_ref != PROTOCOL
                            or directive.payload.candidate.candidate_id != self.candidate_id):
                        raise ValueError('Evaluate the current candidate on heldout-nll')
                    if self.phase == 'select':
                        expected = 'apply_candidate' if eligible(self.metrics) else 'execute'
                        if mode != expected:
                            raise ValueError('Selection violates finite positive delta_nll criterion')
                        if mode == 'apply_candidate' and directive.payload.candidate_id != self.candidate_id:
                            raise ValueError('Apply only the evaluated candidate')
                        self.selection = 'accepted' if mode == 'apply_candidate' else 'rejected'
                    # A hard step bound changes execution budget, never the proposed action.
                    decision = replace(decision, directive=replace(directive, max_steps=1))
                    self.monitor.stage(mode)
                self.monitor.event('model_decision_validated', kind=decision.kind)
                return decision
            except (ValueError, KeyError, TypeError) as exc:
                # Diagnostics stay on stdout; monitor never receives model text
                # or exception content. Keep malformed-response errors bounded.
                print(json.dumps({
                    'event': 'decision_validation_error', 'phase': self.phase,
                    'attempt': attempt + 1, 'error_type': type(exc).__name__,
                    'error_summary': str(exc)[:500],
                }, ensure_ascii=False), flush=True)
                self.monitor.event('model_decision_rejected', step=attempt + 1)
                retry_hint = ('\nPrevious response failed schema/workflow validation. '
                              'Validation feedback (data): ' + json.dumps({
                                  'error_type': type(exc).__name__, 'error_summary': str(exc)[:500],
                              }) + '. Correct this error and return a new JSON decision for the current phase. '
                              'System modes require kind="explore", mode, strategy and arguments; '
                              'do not use kind="invoke" for training, evaluation or application.')
        raise ValueError('Model failed bounded decision validation attempts')

    def feedback(self, directive, feedback):
        if feedback.status != ExecutionStatus.COMPLETED or feedback.simulated:
            raise RuntimeError('Real directive did not complete successfully')
        if directive.mode == 'train':
            if len(feedback.candidates) != 1:
                raise ValueError('Expected one training candidate')
            self.candidate_id = feedback.candidates[0].candidate_id
            self.phase = 'evaluate'
        elif directive.mode == 'evaluate':
            self.metrics = dict(feedback.quality_metrics)
            self.phase = 'select'
        elif directive.mode == 'apply_candidate':
            self.phase = 'execute'
        elif directive.mode == 'execute':
            self.phase = 'finish'


class ParityActorPolicy:
    """The model returns the task's JSON answer; echo merely transports that text."""
    def __init__(self, backend, model):
        self.backend, self.model = backend, model

    def act(self, context):
        response = self.backend.generate(GenerationRequest('actor', self.model, context.observation,
            system='Answer the parity task with only the requested JSON.', max_output_tokens=80))
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


def run(args):
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'result.json').exists() or (root / 'experience.sqlite3').exists():
        raise ValueError('Run directory already has controller records; use a fresh directory')
    lock = root / '.controller.lock'
    lock_fd = lock.open('x')
    store = None
    try:
        deadline = Deadline(args.timeout)
        baseline = HTTPClient(args.baseline_url, deadline, timeout=args.request_timeout)
        worker = HTTPClient(args.worker_url, deadline, timeout=args.request_timeout)
        base = ModelRef('qwen-toy', args.model_name, args.base_revision)
        monitor = Monitor(root, base)
        models = RevisionBackend(baseline, worker, base, notify=monitor.event)
        trainer = WorkerTrainer(worker, models, notify=monitor.event)
        evaluator = WorkerEvaluator(worker, models, notify=monitor.event)
        builtin = {definition.spec.mode: definition for definition in builtin_directives()}
        catalog = DirectiveDefinitions((training_directive(), builtin['evaluate'],
                                        apply_candidate_directive(), builtin['execute']))
        policy = GuardedDecisionPolicy(models, base, catalog, monitor)
        store = SQLiteExperienceStore(root / 'experience.sqlite3')
        experience = BasicExperienceSystem(store, run_id=root.name)
        decision = DefaultDecisionHarness(policy, BoundDecisionTools({}), experience,
            memory=DecisionMemory(root / 'decision_memory.json'), memory_limit=1, experience_limit=1)
        actor_policy = ParityActorPolicy(models, base)
        environment = ParityEnvironment(FileArtifactStore(root / 'artifacts'))
        exploration = DefaultExplorationHarness(actor_policy, environment, experience, learner=trainer,
            evaluator=evaluator, model_activator=WorkerActivator(models),
            memory=ExplorationMemory(root / 'exploration_memory.json'),
            command_handlers={'train': TrainingCommandHandler(), 'apply_candidate': ApplyCandidateHandler()})
        runner = Runner(DecisionMakingLoop(decision, ModularDecisionActionSpace((), directives=catalog)),
                        ExplorationLoop(exploration, EchoActionSpace()))
        status, error_code, remote_may_run = 'running', None, False
        cancel_attempted, cancel_requested, cancellation_confirmed = False, False, False
        service_devices = {}
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
            monitor.stage(policy.phase)
            for _ in range(args.max_steps):
                deadline.remaining()
                result = runner.step(OBJECTIVE)
                monitor.data['models']['actor']['revision'] = actor_policy.model.revision
                if result.kind == 'finished':
                    if policy.phase != 'finish':
                        raise ValueError('Premature finish')
                    status = 'completed'
                    break
                if result.kind == 'completed':
                    directive, feedback = decision.state.last_directive, decision.state.last_feedback
                    policy.feedback(directive, feedback)
                    for ident, training in models.candidates.items():
                        monitor.candidates[ident] = {
                            'candidate_id': ident, 'base_revision': base.revision,
                            'revision': training.model.revision,
                            'status': ('applied' if ident in exploration.state.applied_candidates else
                                       'rejected' if policy.selection == 'rejected' else 'proposed'),
                            'metrics': policy.metrics or training.metrics,
                        }
                    monitor.stage(policy.phase)
                else:
                    monitor.event('progress', status=result.kind)
                if result.kind == 'waiting':
                    time.sleep(min(args.poll_interval, deadline.remaining()))
            else:
                raise TimeoutError('Controller step budget exhausted')
        except (Exception, KeyboardInterrupt) as exc:
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
        monitor.data['models']['actor']['revision'] = actor_policy.model.revision
        result = {
            'schema_version': 1, 'status': status, 'simulated': False,
            'model_name': args.model_name, 'service_devices': service_devices,
            'phase': 'finished' if status == 'completed' else monitor.phase,
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
            'criterion': 'finite delta_nll > 0',
            'decision_source': 'ModelDecisionPolicy; deterministic guards validate, never replace model decisions',
        }
        atomic_write(root / 'result.json', json_bytes(result))
        monitor.stage(result['phase'], status=status)
        return result
    finally:
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
    parser.add_argument('--run-dir', type=Path, default=Path('runs/qwen-toy'))
    parser.add_argument('--timeout', type=float, default=3600, help='Whole controller budget, plus at most 5s cancellation')
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--poll-interval', type=float, default=2)
    parser.add_argument('--max-steps', type=int, default=4096)
    args = parser.parse_args(argv)
    for name in ('timeout', 'request_timeout', 'poll_interval'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be finite and positive')
    if args.max_steps < 1:
        parser.error('max_steps must be positive')
    return args


if __name__ == '__main__':
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'completed' else 1)
