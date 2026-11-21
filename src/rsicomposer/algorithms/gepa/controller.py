"""Choose one GEPA parent, mutate it with the decision model, and act on one matched Lab receipt per decision."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from random import Random
from types import SimpleNamespace

from rsicomposer.api import Decision, ExecutionStatus, OperationCall

from ..shared.experiment import MethodEvents, random_state, set_random_state

from .proposer import GEPAProposer
from .protocol import budget, upstream


def candidate_id(instruction):
    return sha256(instruction.encode()).hexdigest()


@dataclass(frozen=True)
class ValidatedCandidate:
    instruction: str
    scores: tuple[int, ...]
    receipt_id: str
    decision_id: str
    metric_value: float


class GEPAController:
    def __init__(self, context, config, *, proposer=None):
        self.metric = context.lab_catalog.primary_metric('gepa')
        self.parameters = config['parameters']
        self.counts = {name: config['data'][name + '_count'] for name in ('search', 'validation', 'test')}
        self.initial = self.parameters['initial_instruction']
        self.emit = MethodEvents(context, 'gepa')
        # One mutation per round.
        self.proposer = proposer or GEPAProposer(context, config, max_calls=self.parameters['rounds'])
        ensure_loader, self.select_parent, sampler_type, _ = upstream(config['source_dir'])
        self.search_indices = ensure_loader(list(range(self.counts['search'])))
        self.sampler = sampler_type(self.parameters['minibatch_size'], Random(config['seed']))
        self.parent_rng = Random(config['seed'])
        actor, mutation = budget(config)
        self.planned_calls = (actor - 2 * self.counts['test'], mutation)
        self.candidates = []
        self.receipts = set()
        self.registered = {}
        self.deferred = None
        self.pending = None
        self.last = None
        self.selected = None
        self.iteration = 0
        self.sequence = 0
        self.actor_calls = 0
        self.mutation_calls = 0
        self.indices = self.proposal_valid = self.parent_correct = self.accepted = None
        # Each upstream sampler call as [round, indices]; restore replays them on a fresh sampler.
        self.sampler_calls = []

    _SCALARS = ('registered', 'deferred', 'pending', 'last', 'iteration', 'sequence', 'actor_calls', 'mutation_calls',
                'indices', 'proposal_valid', 'parent_correct', 'accepted')

    def state(self):
        return {**{name: getattr(self, name) for name in self._SCALARS},
                'candidates': [asdict(item) for item in self.candidates],
                'selected': None if self.selected is None else self.candidates.index(self.selected),
                'receipts': sorted(self.receipts), 'sampler_calls': self.sampler_calls,
                'parent_rng': random_state(self.parent_rng), 'summary': self.emit.summary}

    def restore(self, state):
        if self.sampler_calls or self.candidates:
            raise RuntimeError('GEPA state restores into a newly constructed controller')
        if set(state) != {*self._SCALARS, 'candidates', 'selected', 'receipts', 'sampler_calls',
                          'parent_rng', 'summary'}:
            raise ValueError('Unexpected GEPA controller state')
        # The upstream sampler owns its shuffle RNG and epoch; replaying its calls rebuilds both exactly.
        for round_index, indices in state['sampler_calls']:
            if self.next_minibatch(round_index) != indices:
                raise ValueError('Rebuilt GEPA sampler diverged from the saved minibatches')
        for name in self._SCALARS:
            setattr(self, name, state[name])
        self.candidates = [ValidatedCandidate(row['instruction'], tuple(row['scores']), row['receipt_id'],
                                              row['decision_id'], row['metric_value']) for row in state['candidates']]
        self.selected = None if state['selected'] is None else self.candidates[state['selected']]
        self.receipts = set(state['receipts'])
        set_random_state(self.parent_rng, state['parent_rng'])
        self.emit.summary = state['summary']
        self.proposer.calls.count = self.mutation_calls

    def next_minibatch(self, round_index):
        indices = list(self.sampler.next_minibatch_ids(self.search_indices, SimpleNamespace(i=round_index)))
        self.sampler_calls.append([round_index, indices])
        return indices

    def required_operations(self):
        return ('gepa.register', 'gepa.evaluate', 'gepa.select')

    def required_profiles(self):
        return {'gepa': 'artifact-search/v1'}

    def utility(self, value):
        return value if self.metric.direction == 'maximize' else -value

    def report(self):
        return self.emit.report()

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        # Exact typed receipts carry the score vector and immutable trajectory reference.
        yield

    def invoke(self, operation, arguments, *, sources=()):
        self.sequence += 1
        dependencies = [item for item in (self.last.decision_id if self.last else None, *sources) if item]
        call = OperationCall('gepa.' + operation, {'request_id': f'gepa-{self.sequence}',
                                                    'round': self.iteration, **arguments})
        self.pending = Decision.invoke(call, reason=f"GEPA {arguments.get('label', operation)}", depends_on_decision_ids=tuple(dict.fromkeys(dependencies)))
        self.record(self.pending)
        return self.pending

    def score(self, instruction, split, label, *, indices=None, sources=()):
        if instruction not in self.registered:
            self.deferred = {'instruction': instruction, 'split': split, 'label': label,
                             'indices': indices, 'sources': tuple(sources)}
            return self.invoke('register', {'candidate': {'instruction': instruction}}, sources=sources)
        return self.invoke('evaluate', {'candidate_id': self.registered[instruction],
            'split': split, 'label': label, 'indices': indices,
            'baseline': label == 'validation_initial'}, sources=sources)

    def record(self, decision):
        self.emit('decision', {'decision_id': decision.decision_id, 'kind': decision.kind, 'reason': decision.reason,
                              'depends_on_decision_ids': list(decision.depends_on_decision_ids),
                              'operation': decision.operations[0].name if decision.operations else None,
                              'request': decision.operations[0].arguments if decision.operations else None})

    def consume(self, window):
        feedback = window.feedbacks[0] if window.feedbacks else None
        decision = self.pending
        request = decision.operations[0].arguments
        if ((feedback is None or feedback.call_id != decision.operations[0].call_id)
                or feedback.status != ExecutionStatus.COMPLETED):
            raise RuntimeError(f"GEPA operation {request['request_id']} did not complete: {getattr(feedback, 'error', None)}")
        data = feedback.data
        receipt_id = data.get('receipt_id')
        if (data.get('request') != request or not isinstance(receipt_id, str)
                or not receipt_id or receipt_id in self.receipts or len(feedback.artifacts) != 1):
            raise RuntimeError('GEPA received an unmatched or duplicate Lab receipt')
        if decision.operations[0].name == 'gepa.evaluate':
            count = len(request['indices']) if request['indices'] is not None else self.counts[request['split']]
            scores = data.get('scores')
            if (not isinstance(scores, list) or len(scores) != count or data.get('count') != count
                    or any(type(value) is not int or value not in (0, 1) for value in scores)):
                raise RuntimeError('Invalid GEPA score receipt')
            self.actor_calls += count
        self.receipts.add(receipt_id)
        self.last, self.pending = decision, None
        self.emit('receipt', {'request_id': request['request_id'], 'receipt_id': receipt_id,
                              'decision_id': decision.decision_id,
                              'artifacts': [asdict(ref) for ref in feedback.artifacts]})
        if decision.operations[0].name == 'gepa.evaluate':
            self.emit('evaluation', {'candidate_id': request['candidate_id'],
                'instruction': next(text for text, key in self.registered.items() if key == request['candidate_id']), 'phase': request['label'], 'score': sum(scores) / count,
                'count': count, 'round': self.iteration, 'receipt_id': receipt_id})
        return request, feedback

    def next_parent(self):
        if self.iteration == self.parameters['rounds']:
            self.selected = max(self.candidates, key=lambda item: self.utility(item.metric_value))
            self.emit('selection', {'candidate_id': candidate_id(self.selected.instruction),
                'instruction': self.selected.instruction, 'phase': 'validation',
                'score': self.selected.metric_value,
                'receipt_id': self.selected.receipt_id})
            return self.invoke('select', {'candidate_id': candidate_id(self.selected.instruction)},
                               sources=(self.selected.decision_id,))

        frontier = {}
        for index, scores in enumerate(zip(*(item.scores for item in self.candidates), strict=True)):
            best = (max if self.metric.direction == 'maximize' else min)(scores)
            frontier[index] = {candidate for candidate, score in enumerate(scores) if score == best}
        selected = self.select_parent(frontier, [self.utility(item.metric_value) for item in self.candidates],
                                      self.parent_rng)
        parent = self.candidates[selected]
        self.indices = self.next_minibatch(self.iteration)
        self.emit('parent', {'round': self.iteration, 'candidate_id': candidate_id(parent.instruction),
                             'frontier_receipt_id': parent.receipt_id, 'indices': self.indices})
        return self.score(parent.instruction, 'search', 'parent', indices=self.indices,
                          sources=(parent.decision_id,))

    def mutate(self, parent, request, feedback):
        """Research step: propose a child of the parent from its minibatch trace, then score it."""
        records = self.proposer.parent_records(feedback.artifacts[0], feedback.data['receipt_id'], request)
        proposal = self.proposer.propose(parent, records, round_index=self.iteration)
        self.mutation_calls += 1
        self.proposal_valid = proposal.candidate is not None
        child = proposal.candidate if self.proposal_valid else parent
        self.emit('candidate', {'candidate_id': candidate_id(child), 'instruction': child,
            'parent_id': candidate_id(parent), 'round': self.iteration,
            'valid_mutation': self.proposal_valid, 'meta_prompt_sha256': proposal.meta_prompt_sha256,
            'proposal': asdict(proposal.evidence)})
        return self.score(child, 'search', 'child', indices=self.indices)

    def admit(self, instruction, data):
        self.candidates.append(ValidatedCandidate(
            instruction, tuple(data['scores']), data['receipt_id'], self.last.decision_id, self.metric.value(data)))

    def decide(self, window):
        if self.pending is None:
            return self.score(self.initial, 'validation', 'validation_initial')
        request, feedback = self.consume(window)
        data = feedback.data
        if self.last.operations[0].name == 'gepa.register':
            instruction = request['candidate']['instruction']
            if data['candidate_id'] != candidate_id(instruction):
                raise RuntimeError('GEPA registration identity mismatch')
            self.registered[instruction] = data['candidate_id']
            deferred, self.deferred = self.deferred, None
            return self.score(**deferred)
        if self.last.operations[0].name == 'gepa.select':
            return self.finish()
        label = request['label']
        instruction = next(text for text, key in self.registered.items() if key == request['candidate_id'])
        if label == 'validation_initial':
            self.admit(instruction, data)
            return self.next_parent()
        if label == 'parent':
            self.parent_correct = self.metric.value(data)
            return self.mutate(instruction, request, feedback)
        if label == 'child':
            self.accepted = (self.proposal_valid and self.utility(self.metric.value(data)) > self.utility(self.parent_correct)
                             and all(item.instruction != instruction for item in self.candidates))
            self.emit('acceptance', {'round': self.iteration, 'candidate_id': candidate_id(instruction),
                'parent_correct': self.parent_correct, 'child_correct': self.metric.value(data),
                'accepted': self.accepted, 'receipt_id': data['receipt_id']})
            # Run the same validation calls in both arms on every attempt.
            return self.score(instruction, 'validation', 'validation_child')
        if label == 'validation_child':
            if self.accepted:
                self.admit(instruction, data)
            self.iteration += 1
            return self.next_parent()
        raise RuntimeError(f'Unexpected GEPA evaluation label: {label}')

    def finish(self):
        if (self.actor_calls, self.mutation_calls) != self.planned_calls:
            raise RuntimeError('GEPA did not consume the paired experiment budget')
        decision = Decision('finish', reason='GEPA prompt selection completed; host sealed evaluation follows',
                            depends_on_decision_ids=(self.last.decision_id,))
        self.record(decision)
        self.emit('finish', {'method': 'gepa', 'condition': self.parameters['condition'],
            'rounds': self.iteration, 'accepted_candidates': len(self.candidates) - 1,
            'initial_instruction': self.initial, 'selected_instruction': self.selected.instruction,
            'selected_candidate_id': candidate_id(self.selected.instruction),
            'validation_score': self.selected.metric_value,
            'sealed_evaluation': 'host_after_finish',
            'test_count': self.counts['test'], 'actor_calls': self.actor_calls, 'mutation_calls': self.mutation_calls,
            'training_submissions': 0})
        return decision
