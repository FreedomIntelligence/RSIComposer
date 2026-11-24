"""Run the artifact's single DSPy CoT predictor inside MathLab."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from rsicomposer.api import AnswerCheck, AnswerVerifier, DomainAgentSpec, GenerationRequest

from ..shared.experiment import ModelCalls
from ..shared.math_search import MathSearchLab
from .controller import candidate_id
from .protocol import ARTIFACT_COMMIT, DSPY_COMMIT, aime_protocol


class AIMEInteger(AnswerVerifier):
    name = 'gepa-artifact-aime-integer/v1'
    spec = {'name': 'gepa_aime_integer'}

    @staticmethod
    def parse(text):
        try:
            return int(text)
        except (TypeError, ValueError):
            return None

    parse_reference = parse


class AIMECotRuntime:
    name = 'gepa-aime-dspy-cot'
    version = '1'
    model_access = 'lab-mediated'

    def __init__(self, component, predictor, adapter, config):
        self.component, self.predictor, self.adapter = component, predictor, adapter
        self.accepts = DomainAgentSpec('instruction', {component: {'type': 'string', 'required': True}})
        self.workers = config['workers']
        model = config['model']
        self.temperature, self.top_p = model['temperature'], model['top_p']
        self.max_tokens = model['max_output_tokens']
        self.settings = {'artifact_commit': ARTIFACT_COMMIT, 'dspy_commit': DSPY_COMMIT,
                         'temperature': self.temperature, 'top_p': self.top_p,
                         'seed': None, 'max_output_tokens': self.max_tokens,
                         'max_model_calls': config['max_model_calls'],
                         'sealed_trials': config['sealed_trials']}

    def run(self, domain_agent, problems, run):
        signature = self.predictor.signature.with_instructions(domain_agent[self.component])

        def answer(problem):
            messages = self.adapter.format(signature, [], {'problem': problem.prompt})
            attempt = run.ask(messages, temperature=self.temperature, top_p=self.top_p,
                              max_output_tokens=self.max_tokens)
            if attempt.status != 'answered':
                return attempt
            try:
                prediction = self.adapter.parse(signature, attempt.completion)
            except ValueError:
                return replace(attempt, status='invalid_format')
            return replace(attempt, extra={'prediction': prediction})

        with ThreadPoolExecutor(max_workers=min(len(problems), self.workers)) as pool:
            futures = [pool.submit(copy_context().run, answer, problem) for problem in problems]
            return [future.result() for future in futures]


class GEPAMathLab(MathSearchLab):
    receipt_kind = 'gepa'
    source_budgets = False
    verifiers = {**MathSearchLab.verifiers, 'gepa_aime_integer': AIMEInteger}

    def __init__(self, context, config):
        self.official, self.component, self.predictor, self.adapter, _ = aime_protocol(
            config['artifact_dir'], config['dspy_dir'])
        self.calls = ModelCalls(context, config['max_model_calls'])
        self.sealed_trials = config['sealed_trials']
        if any(type(config[key]) is not int or config[key] < 1 for key in ('workers', 'sealed_trials')):
            raise ValueError('GEPA workers and sealed_trials must be positive integers')
        super().__init__(context, config)
        path = Path(config['manifest'])
        search = json.loads(path.read_bytes())['splits']['search']
        contents = (path.parent / search['file']).read_bytes()
        if sha256(contents).hexdigest() != search['sha256']:
            raise ValueError('GEPA search reference solutions differ from the manifest')
        rows = (json.loads(line) for line in contents.splitlines() if line.strip())
        self.solutions = {row['id']: row['solution'] for row in rows}

    def make_runtime(self, config):
        return AIMECotRuntime(self.component, self.predictor, self.adapter, self.config)

    def _candidate_id(self, candidate):
        return candidate_id(candidate, self.component)

    def describe(self):
        register, evaluate, select = super().describe()
        return register, replace(evaluate, idempotent=False), select

    def offsets(self, request, size):
        indices = request.get('indices')
        if indices is None:
            return list(range(size))
        # Upstream can pad an epoch with repeated training indices.
        if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < size for i in indices):
            raise ValueError('Invalid GEPA example offsets')
        return indices

    def problems(self, arguments):
        request = arguments['request']
        split = arguments['split']
        if request.get('capture_traces') and split != 'search':
            raise ValueError('Only GEPA search may supply reflection evidence')
        if split == self.sealed_split and request.get('indices') is not None:
            raise ValueError('GEPA sealed evaluation requires the full split')
        rows = super().problems(arguments)
        return rows * self.sealed_trials if split == self.sealed_split else rows

    def solve(self, problems, arguments):
        return self.run_runtime(arguments['candidate'], problems, arguments['request'])

    def ask_actor(self, messages, **sampling):
        request = GenerationRequest('actor', self.actor, '', messages=messages, **sampling)
        return self.attempt(request, self.calls.generate)

    def verify(self, problems, attempts):
        checks = []
        for problem, attempt in zip(problems, attempts, strict=True):
            prediction = attempt.extra.get('prediction')
            if prediction is None:
                checks.append(None)
                continue
            correct = bool(self.official.metric({'answer': problem.answer}, SimpleNamespace(**prediction)))
            answer = AIMEInteger.parse(prediction['answer'])
            checks.append(AnswerCheck(answer is not None, correct, None if answer is None else str(answer)))
        return checks

    def result_data(self, records, ref, arguments):
        data = super().result_data(records, ref, arguments)
        request = arguments['request']
        data['candidate_id'] = request['candidate_id']
        if request.get('capture_traces'):
            trajectories = []
            for row in records:
                prediction = row.get('prediction')
                if prediction is None:
                    trajectories.append(None)
                    continue
                feedback = self.official.metric_with_feedback(
                    {'answer': row['expected'], 'solution': self.solutions[row['id']]}, SimpleNamespace(**prediction))
                trajectories.append({'inputs': {'problem': row['question']}, 'generated_output': prediction,
                                     'feedback': feedback.feedback, 'score': feedback.score})
            data['trajectories'] = trajectories
        return data

    def state(self):
        return {**super().state(), 'model_calls': self.calls.count}

    def restore(self, state):
        super().restore(state)
        self.calls.count = state['model_calls']
