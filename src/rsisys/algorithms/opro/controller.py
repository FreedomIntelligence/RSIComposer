"""OPRO policy over individual, matched Lab receipts."""
from collections import deque
from dataclasses import asdict
from contextlib import contextmanager
import math

from rsisys.loops.contracts.coordination import ExecutionStatus
from rsisys.loops.contracts.decision import Decision, OperationCall

from ..shared.experiment import MethodEvents

from .protocol import candidate_id


class OPROController:
    def __init__(self, context, config):
        self.parameters = config['parameters']
        self.counts = {split: config['data'][split + '_count']
                       for split in ('search', 'validation', 'test')}
        self.emit = MethodEvents(context, 'opro')
        self.initial = self.parameters['initial_instruction']
        self.round = 0
        self.sequence = 0
        self.pending = None
        self.receipts = set()
        self.seen = {self.initial}
        self.history = []
        self.scores = {'validation': {}, 'test': {}}
        self.queue = deque([self.score(self.initial, 'search', -1)])
        self.last_decision_id = None
        self.selected = None
        self.candidate_count = 0

    def required_operations(self):
        return ('opro.generate', 'opro.score')

    def report(self):
        return self.emit.report()

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        # The optimizer sees an explicit score history, independent of the
        # Harness's conversational context budget.
        yield

    def score(self, instruction, split, round_index, *, label=None):
        return ('score', {'instruction': instruction, 'split': split,
                          'round': round_index, 'label': label or split})

    def consume(self, window):
        decision = self.pending
        feedback = window.feedback
        request = decision.operation.arguments
        if (window.last_operation != decision.operation or feedback is None
                or feedback.status != ExecutionStatus.COMPLETED):
            error = feedback.error if feedback is not None else None
            raise RuntimeError(f"OPRO operation {request['request_id']} did not complete: {error}")
        data = feedback.data
        receipt_id = data.get('receipt_id')
        if (data.get('request') != request or not isinstance(receipt_id, str)
                or not receipt_id or receipt_id in self.receipts):
            raise RuntimeError('OPRO received an unmatched or duplicate Lab receipt')
        self.receipts.add(receipt_id)
        self.emit('receipt', {'request_id': request['request_id'], 'receipt_id': receipt_id,
                              'decision_id': decision.decision_id,
                              'artifacts': [asdict(ref) for ref in feedback.artifacts],
                              **({'meta_prompt_sha256': data['meta_prompt_sha256']}
                                 if 'meta_prompt_sha256' in data else {})})
        if decision.operation.name == 'opro.generate':
            candidates = data.get('candidates')
            if (not isinstance(candidates, list)
                    or len(candidates) > self.parameters['candidates_per_round']
                    or any(not isinstance(item, str) or not item or len(item) > 500 for item in candidates)):
                raise RuntimeError('Invalid OPRO candidate receipt')
            for instruction in candidates:
                if instruction in self.seen:
                    continue
                self.seen.add(instruction)
                self.candidate_count += 1
                self.emit('candidate', {'candidate_id': candidate_id(instruction),
                                        'instruction': instruction, 'round': self.round})
                self.queue.append(self.score(instruction, 'search', self.round))
                self.queue.append(self.score(instruction, 'validation', self.round))
            self.round += 1
        else:
            split, label, instruction = (request[key] for key in ('split', 'label', 'instruction'))
            score, count = data.get('score'), data.get('count')
            if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                    or type(count) is not int or count != self.counts[split]):
                raise RuntimeError('Invalid OPRO score receipt')
            self.emit('evaluation', {'candidate_id': candidate_id(instruction), 'instruction': instruction,
                'phase': label, 'score': score, 'count': count, 'round': request['round'],
                'protocol': 'opro-official-gsm8k', 'receipt_id': receipt_id})
            if split == 'search':
                self.history.append(({'instruction': instruction, 'score': score, 'round': request['round'],
                                      'receipt_id': receipt_id}, decision.decision_id))
            else:
                key = instruction if split == 'validation' else label
                self.scores[split][key] = score
        self.last_decision_id = decision.decision_id
        self.pending = None

    def decide(self, window):
        if self.pending is not None:
            self.consume(window)
        dependencies = [self.last_decision_id] if self.last_decision_id else []
        if self.queue:
            operation, arguments = self.queue.popleft()
        elif self.round < self.parameters['rounds']:
            import numpy as np

            history = self.history[:1] if self.parameters['condition'] == 'independent_candidates' else self.history
            indices = np.sort(np.random.RandomState(self.round).choice(
                self.counts['search'], self.parameters['exemplars'], replace=False)).tolist()
            operation = 'generate'
            arguments = {'round': self.round, 'history': [record for record, _ in history],
                         'exemplar_indices': indices}
            dependencies.extend(ident for _, ident in history)
        elif self.initial not in self.scores['validation']:
            operation, arguments = self.score(self.initial, 'validation', -1)
        elif self.selected is None:
            scores = self.scores['validation']
            self.selected = max([self.initial, *scores], key=scores.__getitem__)
            self.emit('selection', {'candidate_id': candidate_id(self.selected), 'instruction': self.selected,
                                    'phase': 'validation', 'score': scores[self.selected]})
            operation, arguments = self.score(self.initial, 'test', self.round, label='test_initial')
            self.queue.append(self.score(self.selected, 'test', self.round, label='test_selected'))
        else:
            decision = Decision('finish', reason='OPRO selection and held-out evaluation completed',
                                depends_on_decision_ids=tuple(dependencies))
            self.record(decision)
            self.emit('finish', {'method': 'opro', 'condition': self.parameters['condition'],
                'rounds': self.round, 'unique_candidates': self.candidate_count,
                'initial_instruction': self.initial, 'selected_instruction': self.selected,
                'selected_candidate_id': candidate_id(self.selected),
                'validation_score': self.scores['validation'][self.selected],
                'test_initial_score': self.scores['test']['test_initial'],
                'test_selected_score': self.scores['test']['test_selected'],
                'training_submissions': 0})
            return decision
        self.sequence += 1
        call = OperationCall('opro.' + operation, {'request_id': f'op-{self.sequence}', **arguments})
        self.pending = Decision('invoke', operation=call, reason=f"OPRO {operation} / {arguments.get('label', self.round)}",
                                depends_on_decision_ids=tuple(dict.fromkeys(dependencies)))
        self.record(self.pending)
        return self.pending

    def record(self, decision):
        self.emit('decision', {'decision_id': decision.decision_id, 'kind': decision.kind,
            'reason': decision.reason, 'depends_on_decision_ids': list(decision.depends_on_decision_ids),
            'operation': decision.operation.name if decision.operation else None,
            'request': decision.operation.arguments if decision.operation else None})
