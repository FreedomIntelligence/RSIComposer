"""Fixed-model math harness measurement, with no search policy: a MathLab whose candidates are programs.

A candidate is a sandboxed harness program, step(problem, history), that asks a
fixed model pool for messages until it returns an answer. The Lab runs it with the
step-program runtime (labs/math/runtimes.py) for `trials` independent trials per
problem; MathLab verifies the answers, aggregates metrics and stores the trace.
This Lab keeps the pool, the sandbox executor, the call budget and the candidate
checks (source validity, no embedded problems, no edits after measuring). The dataset manifest has a
development `search` split and a sealed `heldout` split and selects the answer
verifier (math_verify in the presets); grading_workers grades isolated
verifiers in parallel processes.

Config: {"manifest", "endpoints", "model", "max_calls", "max_tokens", "workers",
"grading_workers", "sandbox_workers", "seed", "temperature", "timeout", "trials"}.
"""
from dataclasses import asdict
from pathlib import Path
import threading

from rsisys.api import (
    HarnessExecutionError,
    ModelRef,
    StepProgramRuntime,
    file_hash as digest,
    sandbox_step,
    validate_source,
)

from .math_search import MathSearchLab


_COUNTS = ('max_calls', 'max_tokens', 'workers', 'grading_workers', 'sandbox_workers', 'timeout', 'trials')


def validate_config(config):
    allowed = {'manifest', 'endpoints', 'model', 'seed', 'temperature', *_COUNTS}
    if set(config) - allowed:
        raise ValueError('Unknown harness_math option')
    for key in ('manifest', 'endpoints', 'model'):
        if key not in config:
            raise ValueError('Missing harness_math ' + key)
    for key in _COUNTS:
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError('Invalid ' + key)
    if not 1 <= config.get('trials', 2) <= 16:
        raise ValueError('Invalid trials')


class FixedModelHarnessLab(MathSearchLab):
    source_budgets = False  # Generation is set by this Lab, not by a benchmark.
    search_splits = ('search',)
    sealed_split = 'heldout'
    receipt_kind = 'harness'

    def __init__(self, context, config, *, pool=None, executor=sandbox_step):
        validate_config(config)
        self.pool = pool or context.models.pool(config, purpose='measurement',
            max_concurrent=config.get('workers', 1))
        self.executor = executor
        self._sandbox_slots = threading.BoundedSemaphore(config.get('sandbox_workers', 32))
        # Harness paths measured so far, with their hashes; a measured file must not change.
        self._harnesses = {}
        self.seed = config.get('seed', 73)
        self.workers = config.get('workers', 32)
        super().__init__(context, config)

    def math_config(self, config):
        return {**super().math_config(config), 'max_output_tokens': config.get('max_tokens', 2048),
                'grading_workers': config.get('grading_workers', 1)}

    def make_runtime(self, config):
        return StepProgramRuntime(max_calls=config.get('max_calls', 3))

    def runtime_options(self, config):
        return {'max_calls': self.runtime.max_calls}

    # Errors that mean the candidate program failed on a task: observed failed trials, not Lab failures.
    candidate_errors = (HarnessExecutionError,)

    @property
    def sandbox_slots(self):
        return self._sandbox_slots

    @property
    def trials_per_problem(self):
        return self._trials

    def bind_actor(self, context, config):
        # The measured model is the pool serving this Lab's model from its endpoints.
        return ModelRef('harness-pool', self.config['model'], 'served')

    def state(self):
        return {**super().state(), 'harnesses': dict(self._harnesses)}

    def restore(self, state):
        super().restore(state)
        self._harnesses = dict(state['harnesses'])

    def _candidate_id(self, candidate):
        if (not isinstance(candidate, dict) or set(candidate) != {'harness'}
                or not isinstance(candidate['harness'], str)):
            raise ValueError('Expected an immutable harness path')
        path = Path(candidate['harness'])
        validate_source(path.read_text())
        return digest(path)

    def describe(self):
        specs = self._search_specs()
        schema = specs[1].input_schema
        schema['properties'].update(trials={'type': 'integer'}, round={'type': 'integer'}, baseline={'type': 'boolean'})
        return specs

    def trials(self, request):
        trials = request.get('trials', self.config.get('trials', 2))
        if type(trials) is not int or not 1 <= trials <= 16:
            raise ValueError('Invalid trial count')
        return trials

    def problems(self, arguments):
        # Each selected problem once per trial, trials adjacent.
        trials = self.trials(arguments['request'])
        return [problem for problem in super().problems(arguments) for _ in range(trials)]

    def solve(self, problems, arguments):
        harness, request = arguments['candidate']['harness'], arguments['request']
        if digest(harness) != request['candidate_id']:
            raise ValueError('Registered harness changed')
        source = Path(harness).read_text()
        validate_source(source)
        harness_hash = digest(harness)
        key = str(Path(harness).resolve())
        if self._harnesses.setdefault(key, harness_hash) != harness_hash:
            raise ValueError('Previously measured harness changed')
        if any(len(problem.prompt) > 20 and problem.prompt in source for problem in problems):
            raise ValueError('Task-specific problem in harness source')
        self._trials = self.trials(request)
        attempts = self.run_runtime({'harness': harness}, problems)
        if digest(harness) != harness_hash:
            raise ValueError('Harness changed during evaluation')
        return attempts

    def feedback(self, record):
        return 'Correct' if record['correct'] else 'Incorrect'

    def result_data(self, records, ref, arguments):
        request, harness = arguments['request'], arguments['candidate']['harness']
        trials = self.trials(request)
        groups = [records[i:i + trials] for i in range(0, len(records), trials)]
        rewards = [[int(row['correct']) for row in group] for group in groups]
        return {**super().result_data(records, ref, arguments),
                'harness': harness, 'candidate_id': request['candidate_id'], 'harness_sha256': digest(harness),
                'score': sum(map(sum, rewards)) / len(records),
                'cost': sum(row['tokens'] for row in records) / len(records),
                'rewards': rewards, 'samples': len(records), 'traces': asdict(ref),
                'harness_errors': sum(row['error'] is not None for row in records),
                'training_performed': False,
                'endpoints_used': sorted({reply['endpoint'] for row in records for reply in row['history']}),
                # One score per problem: the fraction of its trials that were correct.
                'scores': [sum(group) / len(group) for group in rewards],
                'trace_refs': [asdict(ref)] * len(groups),
                'feedback': [f'{sum(group)}/{len(group)} trials correct' for group in rewards]}
