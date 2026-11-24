"""Run the pinned GEPA optimizer behind the Controller/Lab receipt boundary.

Evaluations are Lab operations. Reflection is the Controller's own research
step: the decision model answers it here and the reply joins the replay journal.
"""
from contextlib import contextmanager
from collections import Counter
from hashlib import sha256
import json

from rsicomposer.api import (Decision, ExecutionStatus, OperationCall, ReplayMismatchError,
    emit_observation, observation_id, report_version, report_version_evaluation, report_version_selection)

from .proposer import GEPAReflector
from .paper_bridge import PaperGEPA
from .protocol import ARTIFACT_COMMIT, aime_protocol


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def candidate_id(candidate, component):
    if (not isinstance(candidate, dict) or set(candidate) != {component}
            or not isinstance(candidate[component], str) or not candidate[component].strip()):
        raise ValueError(f'GEPA requires one nonempty instruction for {component}')
    return _digest(candidate)


class _RequestLab(BaseException):
    # Upstream catches Exception during reflection; suspending for a Lab is not a failure.
    def __init__(self, operation, arguments, parents):
        self.operation, self.arguments, self.parents = operation, arguments, parents


class GEPAController:
    def __init__(self, context, config, *, reflector=None):
        p, data = config['parameters'], config['data']
        use_merge = p.get('use_merge', False)
        if type(use_merge) is not bool:
            raise ValueError('GEPA use_merge must be a boolean')
        if 'max_merge_invocations' in p:
            limit = p['max_merge_invocations']
            if type(limit) is not int or limit < 1:
                raise ValueError('GEPA max_merge_invocations must be a positive integer')
        protocol = aime_protocol(config['artifact_dir'], config['dspy_dir'])
        self.paper = PaperGEPA(config, protocol)
        self.component = self.paper.component
        self.initial = self.paper.candidate(self.paper.benchmark.program_cot)
        self.train = [('search', i) for i in range(data['search_count'])]
        self.validation = [('validation', i) for i in range(data['validation_count'])]
        if (not self.train or len(self.validation) < 5
                or any(type(p[key]) is not int or p[key] < 1 for key in
                       ('minibatch_size', 'max_metric_calls'))
                or type(p.get('max_reflection_calls')) is not int or p['max_reflection_calls'] < 1
                or p['minibatch_size'] > len(self.train)):
            raise ValueError('GEPA requires valid positive budgets, a train batch and at least five validation examples')
        self.use_merge = use_merge
        self.max_lab_calls = p['max_lab_calls']
        if type(self.max_lab_calls) is not int or self.max_lab_calls < p['max_metric_calls']:
            raise ValueError('GEPA max_lab_calls must cover the optimizer budget')
        metric = context.lab_catalog.primary_metric('gepa')
        if metric.direction != 'maximize':
            raise ValueError('GEPA requires a metric to maximize')
        self.reflector = reflector or GEPAReflector(context, config, max_calls=p['max_reflection_calls'])
        self.pending = self.previous = None
        self.registered, self.journal = set(), []
        self._reported_screening = set()
        self.summary = {'method': 'gepa', 'use_merge': use_merge, 'status': 'running'}

    def required_operations(self):
        return ('gepa.register', 'gepa.evaluate', 'gepa.select')

    def required_profiles(self):
        return {'gepa': 'artifact-search/v1'}

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        yield

    def state(self):
        return {'version': 4, 'pending': self.pending, 'previous': self.previous,
                'registered': sorted(self.registered), 'journal': self.journal, 'summary': self.summary}

    def restore(self, state):
        if set(state) != {'version', 'pending', 'previous', 'registered', 'journal', 'summary'} or state['version'] != 4:
            raise ValueError('Unsupported GEPA checkpoint')
        self.pending, self.previous = state['pending'], state['previous']
        self.registered, self.journal = set(state['registered']), state['journal']
        self.summary = state['summary']
        self.reflector.calls.count = sum('text' in response for _, response in self.journal)

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
            expected = candidate_id(call.arguments['candidate'], self.component)
        if expected is not None and feedback.data.get('candidate_id') != expected:
            raise RuntimeError('GEPA candidate identity differs from receipt')
        if operation == 'register':
            self.registered.add(expected)
            initial = expected == candidate_id(self.initial, self.component)
            report_version(expected, label='Baseline' if initial else f'Candidate {expected[:8]}',
                           round_index=0 if initial else None, parent_ids=() if initial else None)
            emit_observation('gepa.prompt', observation_id('gepa.prompt', expected), subject=expected,
                             fields={'component': self.component, 'baseline': initial},
                             components={self.component: call.arguments['candidate'][self.component]})
        elif operation == 'evaluate':
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
            self.journal.append([call.arguments['exchange'], response])
            report_version_evaluation(expected, feedback.call_id)
        elif operation == 'select':
            report_version_selection(expected, step=0, event_id=call.call_id)
        self.pending, self.previous = None, decision.decision_id
        return operation == 'select'

    def observe_screening(self, fields, parents, child):
        round_index, status = fields['round'], fields['status']
        key = (round_index, status)
        if key in self._reported_screening:
            return
        attempt = observation_id('gepa.screening', round_index)
        if emit_observation('gepa.screening', observation_id(attempt, status), scope=f'round-{round_index}',
                            subject=attempt, fields=fields, links={
                                'derived_from': [candidate_id(parent, self.component) for parent in parents],
                                'content': [candidate_id(child, self.component)] if child is not None else [],
                            }):
            self._reported_screening.add(key)

    def decide(self, window):
        if self.consume(window):
            return Decision('finish', reason='GEPA selected a candidate; host runs sealed evaluation',
                            depends_on_decision_ids=(self.previous,))

        cursor = 0
        metric_calls = reflection_calls = 0

        def exchange(operation, arguments, parents):
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
            if operation == 'reflect':
                # A new reflection: ask the decision model now and journal the reply.
                try:
                    response = {'text': self.reflector.reflect(arguments['prompt'], round_index=arguments['round'])}
                except Exception as error:
                    # Upstream skips invalid proposals, but an unavailable model is a run failure.
                    raise _ReflectionFailed() from error
                self.journal.append([digest, response])
                cursor += 1
                reflection_calls += 1
                return response
            if metric_calls + len(arguments['indices']) > self.max_lab_calls:
                raise _LabBudgetReached()
            raise _RequestLab(operation, {**arguments, 'exchange': digest}, parents)

        # Only deterministic CPU decisions and journaled reflections replay; evaluations are Lab operations.
        try:
            result = self.paper.run(self.train, self.validation, exchange, self.observe_screening)
        except _ReplayDiverged as error:
            raise ReplayMismatchError(f'GEPA diverged at exchange {error.args[0]} ({error.args[1]})') from error
        except _ReflectionFailed as error:
            raise RuntimeError('GEPA reflection model call failed') from error.__cause__
        except _LabBudgetReached:
            raise RuntimeError('GEPA Lab safety budget reached before the paper optimizer counted enough scores') from None
        except _RequestLab as request:
            arguments = request.arguments
            if request.operation == 'evaluate':
                candidate = arguments.pop('candidate')
                key = candidate_id(candidate, self.component)
                parents = tuple(candidate_id(parent, self.component) for parent in request.parents)
                round_index = arguments['round']
                if not arguments['baseline'] and key not in parents:
                    attempt = observation_id('gepa.proposal', round_index, key, parents)
                    emit_observation('gepa.proposal', attempt, scope=f'round-{round_index}', subject=attempt,
                                     links={'derived_from': parents, 'content': [key]},
                                     fields={'round': round_index, 'merge': len(parents) > 1})
                if key not in self.registered:
                    if parents and key not in parents and self.registered.issuperset(parents):
                        report_version(key, label=f'Candidate {key[:8]}',
                                       round_index=arguments['round'], parent_ids=parents)
                    return self.invoke('register', {'candidate': candidate})
                arguments['candidate_id'] = key
            return self.invoke(request.operation, arguments)

        if cursor != len(self.journal):
            raise ReplayMismatchError('GEPA finished with unused receipts')
        # The final population can also fill ancestry for candidates restored from older runs.
        keys = [candidate_id(candidate, self.component) for candidate in result['candidates']]
        counts = Counter(keys)
        for index, (key, parents) in enumerate(zip(keys, result['parents'], strict=True)):
            refs = tuple(keys[parent] for parent in parents if parent is not None)
            attempt = observation_id('gepa.population', index, key)
            emit_observation('gepa.population', attempt, subject=attempt,
                             links={'content': [key], 'derived_from': [observation_id('gepa.population', parent, keys[parent])
                                    for parent in parents if parent is not None]}, fields={'index': index})
            if counts[key] == 1 and all(counts[ref] == 1 for ref in refs):
                report_version(key, label='Baseline' if index == 0 else f'Candidate {key[:8]}', parent_ids=refs)
        self.summary = {**result, 'method': 'gepa', 'use_merge': self.use_merge,
            'status': 'completed', 'artifact_commit': ARTIFACT_COMMIT,
            'metric_calls': metric_calls, 'reflection_calls': reflection_calls, 'sealed_evaluation': 'host_after_finish'}
        return self.invoke('select', {'candidate_id': candidate_id(result['selected_candidate'], self.component)})


class _ReplayDiverged(BaseException):
    pass


class _ReflectionFailed(BaseException):
    pass


class _LabBudgetReached(BaseException):
    pass
