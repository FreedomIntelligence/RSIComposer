"""Run the pinned GEPA optimizer behind the Controller/Lab receipt boundary."""
from contextlib import contextmanager
from hashlib import sha256
import json
from types import SimpleNamespace

from rsisys.api import Decision, ExecutionStatus, OperationCall, ReplayMismatchError

from .protocol import SOURCE_COMMIT, merge_api


COMPONENTS = ('strategy', 'verification')


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def candidate_id(candidate):
    if (not isinstance(candidate, dict) or set(candidate) != set(COMPONENTS)
            or any(not isinstance(text, str) for text in candidate.values())):
        raise ValueError('GEPA requires string strategy and verification components')
    return _digest(candidate)


class _RequestLab(BaseException):
    # Upstream catches Exception during reflection; suspending for a Lab is not a failure.
    def __init__(self, operation, arguments):
        self.operation, self.arguments = operation, arguments


class _UpstreamAdapter:
    propose_new_texts = None

    def __init__(self, exchange, batch_type):
        self.exchange, self.batch_type = exchange, batch_type
        self.iteration = 0
        self.trace = []

    def on_iteration_start(self, event):
        self.iteration = event['iteration']

    def on_candidate_selected(self, event):
        self.trace.append(['parent', self.iteration, event['candidate_idx']])

    def on_merge_attempted(self, event):
        self.trace.append(['merge_attempted', self.iteration, list(event['parent_ids'])])

    def on_merge_accepted(self, event):
        self.trace.append(['merge_accepted', self.iteration, event['new_candidate_idx']])

    def on_merge_rejected(self, event):
        self.trace.append(['merge_rejected', self.iteration, list(event['parent_ids'])])

    def evaluate(self, batch, candidate, capture_traces=False):
        split = batch[0][0]
        indices = [index for _, index in batch]
        response = self.exchange('evaluate', {'candidate': candidate, 'split': split,
            'indices': indices, 'capture_traces': capture_traces, 'round': self.iteration,
            'baseline': self.iteration == 0})
        return self.batch_type(outputs=[None] * len(batch), scores=response['scores'],
                               trajectories=response.get('trajectories'))

    def make_reflective_dataset(self, candidate, evaluation, components):
        return {component: evaluation.trajectories for component in components}

    def reflect(self, prompt):
        return self.exchange('reflect', {'prompt': prompt, 'round': self.iteration})['text']


class GEPAMergeController:
    def __init__(self, context, config):
        self.optimize, self.batch_type = merge_api(config['source_dir'])
        p, data = config['parameters'], config['data']
        initial = p['initial_components']
        candidate_id(initial)
        if any(not text.strip() for text in initial.values()):
            raise ValueError('Initial GEPA components must be nonempty')
        self.initial = {name: initial[name] for name in COMPONENTS}
        self.train = [('search', i) for i in range(data['search_count'])]
        self.validation = [('validation', i) for i in range(data['validation_count'])]
        if (not self.train or len(self.validation) < 5
                or any(type(p[key]) is not int or p[key] < 1 for key in
                       ('rounds', 'minibatch_size', 'max_metric_calls', 'max_merge_invocations'))
                or p['minibatch_size'] > len(self.train) or type(p['use_merge']) is not bool):
            raise ValueError('GEPA requires valid positive budgets, a train batch and at least five validation examples')
        self.options = dict(seed=config['seed'], reflection_minibatch_size=p['minibatch_size'],
            use_merge=p['use_merge'], max_merge_invocations=p['max_merge_invocations'],
            max_metric_calls=p['max_metric_calls'], stop_callbacks=lambda state: state.i + 1 >= p['rounds'])
        metric = context.lab_catalog.primary_metric('gepa')
        if metric.direction != 'maximize':
            raise ValueError('GEPA requires a metric to maximize')
        self.pending = self.previous = None
        self.registered, self.journal = set(), []
        self.summary = {'method': 'gepa_merge', 'status': 'running'}

    def required_operations(self):
        return ('gepa.register', 'gepa.evaluate', 'gepa.reflect', 'gepa.select')

    def required_profiles(self):
        return {'gepa': 'artifact-search/v1'}

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        yield

    def state(self):
        return {'version': 1, 'pending': self.pending, 'previous': self.previous,
                'registered': sorted(self.registered), 'journal': self.journal, 'summary': self.summary}

    def restore(self, state):
        if set(state) != {'version', 'pending', 'previous', 'registered', 'journal', 'summary'} or state['version'] != 1:
            raise ValueError('Unsupported GEPA merge checkpoint')
        self.pending, self.previous = state['pending'], state['previous']
        self.registered, self.journal = set(state['registered']), state['journal']
        self.summary = state['summary']

    def report(self):
        return self.summary

    def invoke(self, operation, arguments):
        self.pending = Decision.invoke(OperationCall('gepa.' + operation, arguments),
            reason=f'GEPA {operation}', depends_on_decision_ids=(self.previous,) if self.previous else ())
        return self.pending

    def consume(self, window):
        decision = self.pending
        if decision is None:
            return False
        call = decision.operations[0]
        operation = call.name.removeprefix('gepa.')
        feedback = next((item for item in window.feedbacks if item.call_id == call.call_id), None)
        if feedback is None or feedback.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f'GEPA {operation} did not complete: {getattr(feedback, "error", None)}')
        expected = call.arguments.get('candidate_id')
        if operation == 'register':
            expected = candidate_id(call.arguments['candidate'])
        if expected is not None and feedback.data.get('candidate_id') != expected:
            raise RuntimeError('GEPA candidate identity differs from receipt')
        if operation == 'register':
            self.registered.add(expected)
        elif operation in ('evaluate', 'reflect'):
            if operation == 'evaluate':
                count = len(call.arguments['indices'])
                data = feedback.data
                if (data.get('count') != count or len(data.get('scores', ())) != count
                        or any(type(score) is not int or score not in (0, 1) for score in data['scores'])):
                    raise RuntimeError('Invalid GEPA evaluation receipt')
                response = {'scores': data['scores']}
                if call.arguments['capture_traces']:
                    traces = data.get('trajectories')
                    if not isinstance(traces, list) or len(traces) != count:
                        raise RuntimeError('Missing GEPA trajectories')
                    response['trajectories'] = traces
            else:
                text = feedback.data.get('text')
                if not isinstance(text, str):
                    raise RuntimeError('Invalid GEPA reflection receipt')
                response = {'text': text}
            self.journal.append([call.arguments['exchange'], response])
        self.pending, self.previous = None, decision.decision_id
        return operation == 'select'

    def decide(self, window):
        if self.consume(window):
            return Decision('finish', reason='GEPA selected a candidate; host runs sealed evaluation',
                            depends_on_decision_ids=(self.previous,))

        cursor = 0
        metric_calls = reflection_calls = 0

        def exchange(operation, arguments):
            nonlocal cursor, metric_calls, reflection_calls
            digest = _digest([operation, arguments])
            if cursor < len(self.journal):
                recorded, response = self.journal[cursor]
                if recorded != digest:
                    # Do not let upstream convert a replay error into a skipped proposal.
                    raise _ReplayDiverged(cursor, operation)
                cursor += 1
                if operation == 'evaluate':
                    metric_calls += len(response['scores'])
                else:
                    reflection_calls += 1
                return response
            raise _RequestLab(operation, {**arguments, 'exchange': digest})

        # Only deterministic CPU decisions replay. Every external effect is a host Lab operation.
        adapter = _UpstreamAdapter(exchange, self.batch_type)
        try:
            result = self.optimize(self.initial, self.train, self.validation, adapter,
                reflection_lm=adapter.reflect, callbacks=[adapter],
                logger=SimpleNamespace(log=lambda message: None), **self.options)
        except _ReplayDiverged as error:
            raise ReplayMismatchError(f'GEPA diverged at exchange {error.args[0]} ({error.args[1]})') from error
        except _RequestLab as request:
            arguments = request.arguments
            if request.operation == 'evaluate':
                candidate = arguments.pop('candidate')
                key = candidate_id(candidate)
                if key not in self.registered:
                    return self.invoke('register', {'candidate': candidate})
                arguments['candidate_id'] = key
            return self.invoke(request.operation, arguments)

        if cursor != len(self.journal) or metric_calls != result.total_metric_calls:
            raise ReplayMismatchError('GEPA finished with a different evaluation budget or unused receipts')
        self.summary = {'method': 'gepa_merge', 'status': 'completed', 'source_commit': SOURCE_COMMIT,
            'candidates': result.candidates, 'parents': result.parents, 'trace': adapter.trace,
            'metric_calls': metric_calls, 'reflection_calls': reflection_calls,
            'selected_candidate': result.best_candidate,
            'validation_score': result.val_aggregate_scores[result.best_idx], 'sealed_evaluation': 'host_after_finish'}
        return self.invoke('select', {'candidate_id': candidate_id(result.best_candidate)})


class _ReplayDiverged(BaseException):
    pass
