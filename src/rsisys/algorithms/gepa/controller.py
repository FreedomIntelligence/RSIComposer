"""Choose one GEPA parent and act on one matched Lab receipt per decision."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from random import Random
from types import SimpleNamespace

from rsisys.loops.contracts.coordination import ExecutionStatus
from rsisys.loops.contracts.decision import Decision, OperationCall

from ..shared.experiment import MethodEvents

from .protocol import budget, upstream, valid_instruction


def candidate_id(instruction):
    return sha256(instruction.encode()).hexdigest()


@dataclass(frozen=True)
class ValidatedCandidate:
    instruction: str
    scores: tuple[int, ...]
    receipt_id: str
    decision_id: str


class GEPAController:
    def __init__(self, context, config):
        self.parameters = config['parameters']
        self.counts = {name: config['data'][name + '_count'] for name in ('search', 'validation', 'test')}
        self.initial = self.parameters['initial_instruction']
        self.emit = MethodEvents(context, 'gepa')
        ensure_loader, self.select_parent, sampler_type, _ = upstream(config['source_dir'])
        self.search_indices = ensure_loader(list(range(self.counts['search'])))
        self.sampler = sampler_type(self.parameters['minibatch_size'], Random(config['seed']))
        self.parent_rng = Random(config['seed'])
        self.planned_calls = budget(config)
        self.candidates = []
        self.receipts = set()
        self.test_scores = {}
        self.pending = None
        self.last = None
        self.selected = None
        self.iteration = 0
        self.sequence = 0
        self.actor_calls = 0
        self.mutation_calls = 0

    def required_operations(self):
        return ('gepa.score', 'gepa.mutate')

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
        self.pending = Decision('invoke', operation=call, reason=f"GEPA {arguments.get('label', operation)}",
                                depends_on_decision_ids=tuple(dict.fromkeys(dependencies)))
        self.record(self.pending)
        return self.pending

    def score(self, instruction, split, label, *, indices=None, sources=()):
        return self.invoke('score', {'instruction': instruction, 'split': split, 'label': label,
                                    'indices': indices}, sources=sources)

    def record(self, decision):
        self.emit('decision', {'decision_id': decision.decision_id, 'kind': decision.kind, 'reason': decision.reason,
                              'depends_on_decision_ids': list(decision.depends_on_decision_ids),
                              'operation': decision.operation.name if decision.operation else None,
                              'request': decision.operation.arguments if decision.operation else None})

    def consume(self, window):
        decision = self.pending
        request = decision.operation.arguments
        feedback = window.feedback
        if (window.last_operation != decision.operation or feedback is None
                or feedback.status != ExecutionStatus.COMPLETED):
            raise RuntimeError(f"GEPA operation {request['request_id']} did not complete: {getattr(feedback, 'error', None)}")
        data = feedback.data
        receipt_id = data.get('receipt_id')
        if (data.get('request') != request or not isinstance(receipt_id, str)
                or not receipt_id or receipt_id in self.receipts or len(feedback.artifacts) != 1):
            raise RuntimeError('GEPA received an unmatched or duplicate Lab receipt')
        if decision.operation.name == 'gepa.score':
            count = len(request['indices']) if request['indices'] is not None else self.counts[request['split']]
            scores = data.get('scores')
            if (not isinstance(scores, list) or len(scores) != count or data.get('count') != count
                    or any(type(value) is not int or value not in (0, 1) for value in scores)):
                raise RuntimeError('Invalid GEPA score receipt')
            self.actor_calls += count
        else:
            instruction = data.get('candidate')
            if instruction is not None and not valid_instruction(instruction):
                raise RuntimeError('Invalid GEPA mutation receipt')
            self.mutation_calls += 1
        self.receipts.add(receipt_id)
        self.last, self.pending = decision, None
        self.emit('receipt', {'request_id': request['request_id'], 'receipt_id': receipt_id,
                              'decision_id': decision.decision_id,
                              'artifacts': [asdict(ref) for ref in feedback.artifacts],
                              **({'meta_prompt_sha256': data['meta_prompt_sha256']}
                                 if 'meta_prompt_sha256' in data else {})})
        if decision.operation.name == 'gepa.score':
            self.emit('evaluation', {'candidate_id': candidate_id(request['instruction']),
                'instruction': request['instruction'], 'phase': request['label'], 'score': sum(scores) / count,
                'count': count, 'round': self.iteration, 'receipt_id': receipt_id})
        return request, feedback

    def next_parent(self):
        if self.iteration == self.parameters['rounds']:
            self.selected = max(self.candidates, key=lambda item: sum(item.scores))
            self.emit('selection', {'candidate_id': candidate_id(self.selected.instruction),
                'instruction': self.selected.instruction, 'phase': 'validation',
                'score': sum(self.selected.scores) / self.counts['validation'],
                'receipt_id': self.selected.receipt_id})
            return self.score(self.initial, 'test', 'test_initial', sources=(self.selected.decision_id,))

        frontier = {}
        for index, scores in enumerate(zip(*(item.scores for item in self.candidates), strict=True)):
            best = max(scores)
            frontier[index] = {candidate for candidate, score in enumerate(scores) if score == best}
        selected = self.select_parent(frontier, [sum(item.scores) / len(item.scores) for item in self.candidates],
                                      self.parent_rng)
        parent = self.candidates[selected]
        self.indices = self.sampler.next_minibatch_ids(self.search_indices, SimpleNamespace(i=self.iteration))
        self.emit('parent', {'round': self.iteration, 'candidate_id': candidate_id(parent.instruction),
                             'frontier_receipt_id': parent.receipt_id, 'indices': self.indices})
        return self.score(parent.instruction, 'search', 'parent', indices=self.indices,
                          sources=(parent.decision_id,))

    def admit(self, instruction, data):
        self.candidates.append(ValidatedCandidate(
            instruction, tuple(data['scores']), data['receipt_id'], self.last.decision_id))

    def decide(self, window):
        if self.pending is None:
            return self.score(self.initial, 'validation', 'validation_initial')
        request, feedback = self.consume(window)
        data = feedback.data
        if self.last.operation.name == 'gepa.mutate':
            proposal = data['candidate']
            self.proposal_valid = proposal is not None
            instruction = proposal if self.proposal_valid else request['instruction']
            self.emit('candidate', {'candidate_id': candidate_id(instruction), 'instruction': instruction,
                'parent_id': candidate_id(request['instruction']), 'round': self.iteration,
                'valid_mutation': self.proposal_valid, 'receipt_id': data['receipt_id']})
            return self.score(instruction, 'search', 'child', indices=self.indices)
        label = request['label']
        instruction = request['instruction']
        if label == 'validation_initial':
            self.admit(instruction, data)
            return self.next_parent()
        if label == 'parent':
            self.parent_correct = sum(data['scores'])
            return self.invoke('mutate', {'instruction': instruction, 'parent_receipt_id': data['receipt_id'],
                                         'parent_artifact': asdict(feedback.artifacts[0]),
                                         'indices': self.indices})
        if label == 'child':
            self.accepted = (self.proposal_valid and sum(data['scores']) > self.parent_correct
                             and all(item.instruction != instruction for item in self.candidates))
            self.emit('acceptance', {'round': self.iteration, 'candidate_id': candidate_id(instruction),
                'parent_correct': self.parent_correct, 'child_correct': sum(data['scores']),
                'accepted': self.accepted, 'receipt_id': data['receipt_id']})
            # Run the same validation calls in both arms on every attempt.
            return self.score(instruction, 'validation', 'validation_child')
        if label == 'validation_child':
            if self.accepted:
                self.admit(instruction, data)
            self.iteration += 1
            return self.next_parent()
        self.test_scores[label] = sum(data['scores']) / self.counts['test']
        if label == 'test_initial':
            return self.score(self.selected.instruction, 'test', 'test_selected',
                              sources=(self.selected.decision_id,))
        return self.finish()

    def finish(self):
        if (self.actor_calls, self.mutation_calls) != self.planned_calls:
            raise RuntimeError('GEPA did not consume the paired experiment budget')
        decision = Decision('finish', reason='GEPA prompt selection and held-out evaluation completed',
                            depends_on_decision_ids=(self.last.decision_id,))
        self.record(decision)
        self.emit('finish', {'method': 'gepa', 'condition': self.parameters['condition'],
            'rounds': self.iteration, 'accepted_candidates': len(self.candidates) - 1,
            'initial_instruction': self.initial, 'selected_instruction': self.selected.instruction,
            'selected_candidate_id': candidate_id(self.selected.instruction),
            'validation_score': sum(self.selected.scores) / self.counts['validation'],
            'test_initial_score': self.test_scores['test_initial'], 'test_selected_score': self.test_scores['test_selected'],
            'test_count': self.counts['test'], 'actor_calls': self.actor_calls, 'mutation_calls': self.mutation_calls,
            'training_submissions': 0})
        return decision
