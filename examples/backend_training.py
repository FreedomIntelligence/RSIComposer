"""Real, tiny CPU training behind ActorLearner; no LLM training or verl adapter.

Run with Python 3.12+: python examples/backend_training.py
The same file is launched with --worker for each independent training process.
Only stdlib and this checkout are needed. Job receipts are in-process, and the
demo's in-memory checkpoint artifacts live only as long as its artifact store.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

# Allow direct execution from an uninstalled source checkout, from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import TrainSpec
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinitions, training_directive,
)
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.environment import SandboxEchoEnvironment
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import ArtifactStore


def worker(directory: Path) -> None:
    """Fit y = weight * x; all numerical optimization happens in this process."""
    # Each data artifact is UTF-8 JSON: {"samples": [[x, y], ...]}.
    # Multiple artifacts are concatenated; samples must be finite numbers.
    samples = []
    for filename in json.loads((directory / 'inputs.json').read_text()):
        rows = json.loads((directory / filename).read_text())['samples']
        for x, y in rows:
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y)):
                raise ValueError('Samples must contain finite numeric x/y pairs')
            samples.append((x, y))
    if not samples:
        raise ValueError('Training requires at least one sample')

    # toy-recipe is a fixed local recipe name, not a path or remote config:
    # initialize w=0, then perform 100 full-batch MSE steps with learning rate .1.
    weight, learning_rate, steps = 0.0, 0.1, 100

    def mse(w):
        return sum((w * x - y) ** 2 for x, y in samples) / len(samples)

    loss_before = mse(weight)
    for _ in range(steps):
        gradient = 2 * sum((weight * x - y) * x for x, y in samples) / len(samples)
        weight -= learning_rate * gradient
    loss_after = mse(weight)
    if not all(math.isfinite(v) for v in (weight, loss_before, loss_after)):
        raise ValueError('Training produced non-finite weights or losses')
    checkpoint = {
        'method': 'linear_regression', 'recipe_ref': 'toy-recipe',
        'weight': weight, 'initial_weight': 0.0, 'learning_rate': learning_rate,
        'steps': steps, 'sample_count': len(samples),
        'loss_before': loss_before, 'loss_after': loss_after,
    }
    (directory / 'checkpoint.json').write_text(json.dumps(checkpoint, allow_nan=False))


@dataclass
class _Job:
    request: ActorTrainingRequest
    directory: Path
    outcome: JobOutcome[TrainingResult]
    process: subprocess.Popen | None = None


class SubprocessActorLearner:
    """Single-controller demo backend with stable receipts and explicit polling."""

    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts
        self.backend = f'toy-cpu-{uuid4().hex}'
        self._temporary = tempfile.TemporaryDirectory(prefix='rsisys-toy-training-')
        self._jobs: dict[JobRef, _Job] = {}
        self._keys: dict[str, JobRef] = {}
        self._closed = False

    def submit(self, request: ActorTrainingRequest, *, idempotency_key: str) -> JobRef:
        if self._closed:
            raise RuntimeError('Learner is closed')
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError('An idempotency key is required')
        if idempotency_key in self._keys:
            job = self._keys[idempotency_key]
            if self._jobs[job].request != request:
                raise ValueError('Idempotency key was reused with a different request')
            return job
        if not isinstance(request, ActorTrainingRequest):
            raise TypeError('Expected ActorTrainingRequest')
        if request.method != 'linear_regression' or request.recipe_ref != 'toy-recipe':
            raise ValueError('Only toy-recipe / linear_regression is supported')
        # No trajectory requirement: this backend consumes data artifacts directly.
        job = JobRef(uuid4().hex, self.backend)
        directory = Path(self._temporary.name) / job.job_id
        state = _Job(deepcopy(request), directory, JobOutcome(job, JobStatus.QUEUED))
        self._jobs[job], self._keys[idempotency_key] = state, job
        try:
            directory.mkdir()
            filenames = []
            for index, ref in enumerate(request.data_refs):
                filename = f'data-{index}.json'
                (directory / filename).write_bytes(self.artifacts.get(ref))
                filenames.append(filename)
            (directory / 'inputs.json').write_text(json.dumps(filenames))
            # A file, not PIPE: a failing worker can never fill an unread pipe.
            # Popen duplicates this handle; the parent closes its copy immediately.
            with (directory / 'stderr.txt').open('wb') as stderr:
                state.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
                )
            state.outcome = JobOutcome(job, JobStatus.RUNNING)
        except Exception as exc:
            state.outcome = JobOutcome(job, JobStatus.FAILED, error=f'{type(exc).__name__}: {exc}')
        return job

    def _get(self, job: JobRef) -> _Job:
        if job not in self._jobs:
            raise ValueError('Job does not belong to this backend')
        return self._jobs[job]

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]:
        state = self._get(job)
        if state.outcome.terminal:
            return deepcopy(state.outcome)
        code = state.process.poll()
        if code is None:
            return deepcopy(state.outcome)
        if code != 0:
            with (state.directory / 'stderr.txt').open('rb') as stderr:
                stderr.seek(0, 2)
                stderr.seek(max(0, stderr.tell() - 4096))
                detail = stderr.read().decode('utf-8', errors='replace').strip()
            state.outcome = JobOutcome(job, JobStatus.FAILED,
                                       error=f'Worker exited with code {code}: {detail}')
        else:
            try:
                content = (state.directory / 'checkpoint.json').read_bytes()
                checkpoint = json.loads(content)
                metrics = {key: float(checkpoint[key]) for key in ('loss_before', 'loss_after')}
                if not all(math.isfinite(v) for v in (*metrics.values(), checkpoint['weight'])):
                    raise ValueError('Invalid checkpoint')
                ref = self.artifacts.put(content, kind='linear_regression_weights')
                model = replace(state.request.base, revision=f'toy-trained-{job.job_id}')
                result = TrainingResult(
                    state.request.base, model,
                    Candidate(job.job_id, state.request.base.revision, {'model': ref}, ('actor',)),
                    metrics=metrics, simulated=False,
                )
                state.outcome = JobOutcome(job, JobStatus.SUCCEEDED, result)
            except Exception as exc:
                state.outcome = JobOutcome(job, JobStatus.FAILED, error=f'{type(exc).__name__}: {exc}')
        return deepcopy(state.outcome)

    def cancel(self, job: JobRef) -> None:
        state = self._get(job)
        if self.poll(job).terminal:
            return
        process = state.process
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        state.outcome = JobOutcome(job, JobStatus.CANCELLED)

    def result(self, job: JobRef) -> TrainingResult:
        # Reading a result neither polls nor submits another process.
        outcome = self._get(job).outcome
        if outcome.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f'Job has no successful result: {outcome.status}')
        return deepcopy(outcome.result)

    def close(self) -> None:
        if self._closed:
            return
        for job in self._jobs:
            self.cancel(job)
        self._temporary.cleanup()
        self._closed = True


class ScalarModelService:
    """Tiny in-process serving fixture, explicitly separate from training."""
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.weights = {}
        self.receipts = {}

    def activate(self, result, *, idempotency_key):
        previous = self.receipts.get(idempotency_key)
        if previous is not None:
            if previous != result:
                raise ValueError('Activation key reused with different result')
            return result.model
        checkpoint = json.loads(self.artifacts.get(result.candidate.changes['model']))
        weight = checkpoint['weight']
        if type(weight) not in (int, float) or not math.isfinite(weight):
            raise ValueError('Invalid scalar weights')
        self.weights[result.model] = weight
        self.receipts[idempotency_key] = deepcopy(result)
        return result.model

    def predict(self, model, value):
        return self.weights[model] * value


class TrainingOnlyPolicy:
    # This model identifies the scalar initialization, not the demo's mock LLM.
    model = ModelRef('toy-cpu', 'scalar-linear-regression', 'zero-weight')

    def update_model(self, model):
        self.model = model

    def act(self, context):
        raise AssertionError('Backend-owned training must not call actor.act')


def exploration_factory(context) -> ExplorationLoop:
    learner = SubprocessActorLearner(context.artifacts)
    context.add_cleanup(learner.close)
    environment = SandboxEchoEnvironment(context.sandbox, context.artifacts)
    context.add_cleanup(environment.close)
    context.exploration_directives = DirectiveDefinitions((training_directive(),))
    # Direct construction also creates the training JobBook with this learner.
    harness = DefaultExplorationHarness(
        TrainingOnlyPolicy(), environment, context.experience, learner=learner,
        command_handlers={'train': TrainingCommandHandler()},
        model_activator=ScalarModelService(context.artifacts),
    )
    return ExplorationLoop(harness, EchoActionSpace())


def main() -> dict:
    config = DemoConfig.load(ROOT / 'configs/demo.json')
    runner, _ = build_demo(config, exploration_factory=exploration_factory)
    with runner:
        artifacts = runner.assembly.artifacts
        data = artifacts.put(json.dumps({
            'samples': [[-1.0, -2.0], [-0.5, -1.0], [0.5, 1.0], [1.0, 2.0]],
        }).encode(), kind='training_data')
        directive = Directive(
            'toy-training', 'Fit y = 2*x on CPU', 'delegate', mode='train',
            payload=TrainSpec('toy-recipe', method='linear_regression', apply_result=True), input_refs=(data,),
        )
        exploration = runner.exploration
        progress = exploration.accept(directive)  # prepare needs no trajectory.
        deadline = time.monotonic() + 15
        while not progress.terminal:
            if time.monotonic() >= deadline:
                exploration.cancel(directive.directive_id)
                exploration.advance(directive.directive_id)
                raise TimeoutError('Toy training did not finish within 15 seconds')
            progress = exploration.advance(directive.directive_id)
            if not progress.terminal:
                time.sleep(0.01)
        if progress.feedback.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(progress.feedback.error or progress.phase)
        result = exploration.harness.learner.result(progress.job)
        checkpoint_ref = result.candidate.changes['model']
        checkpoint = json.loads(artifacts.get(checkpoint_ref))
        if not result.metrics['loss_after'] < result.metrics['loss_before']:
            raise AssertionError('Training did not reduce MSE')
        exploration.acknowledge(directive.directive_id)
        # Include actual weights as well as their artifact reference because the
        # demo store is in memory and worker files are removed by context cleanup.
        return {
            'training_performed': True, 'llm_training': False, 'simulated': False,
            **result.metrics, 'checkpoint': checkpoint,
            'checkpoint_artifact': asdict(checkpoint_ref),
            'training_result': asdict(result),
            'active_model': asdict(exploration.harness.policy.model),
            'prediction_at_3': exploration.harness.model_activator.predict(exploration.harness.policy.model, 3),
        }


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        print(json.dumps(main(), indent=2, ensure_ascii=False))
    else:
        raise SystemExit('Usage: backend_training.py [--worker DIRECTORY]')
