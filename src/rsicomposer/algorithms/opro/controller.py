"""OPRO policy over individual, matched Lab receipts; the optimizer step runs in the Controller."""
from collections import deque
from dataclasses import asdict
from contextlib import contextmanager
import math

from rsicomposer.api import Decision, ExecutionStatus, OperationCall

from ..shared.experiment import MethodEvents

from .proposer import OPROProposer
from .protocol import candidate_id


class OPROController:
    def __init__(self, context, config, *, proposer=None):
        self.metric = context.lab_catalog.primary_metric('opro')
        self.parameters = config['parameters']
        self.proposer = proposer or OPROProposer(context, config)
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
        self.scores = {'validation': {}}
        self.queue = deque([('register', {'candidate': {'instruction': self.initial}}),
                            self.score(self.initial, 'search', -1)])
        self.last_decision_id = None
        self.selected = None
        self.candidate_count = 0
        self.optimizer_calls = 0

    def state(self):
        return {'round': self.round, 'sequence': self.sequence, 'pending': self.pending,
                'receipts': sorted(self.receipts), 'seen': sorted(self.seen),
                'history': [[record, ident] for record, ident in self.history],
                'scores': {split: dict(rows) for split, rows in self.scores.items()},
                'queue': [[operation, arguments] for operation, arguments in self.queue],
                'last_decision_id': self.last_decision_id, 'selected': self.selected,
                'candidate_count': self.candidate_count, 'optimizer_calls': self.optimizer_calls,
                'summary': self.emit.summary}

    def restore(self, state):
        self.round, self.sequence, self.pending = state['round'], state['sequence'], state['pending']
        self.receipts, self.seen = set(state['receipts']), set(state['seen'])
        self.history = [(record, ident) for record, ident in state['history']]
        self.scores = {split: dict(rows) for split, rows in state['scores'].items()}
        self.queue = deque((operation, arguments) for operation, arguments in state['queue'])
        self.last_decision_id, self.selected = state['last_decision_id'], state['selected']
        self.candidate_count, self.emit.summary = state['candidate_count'], state['summary']
        self.optimizer_calls = self.proposer.calls.count = state['optimizer_calls']

    def required_operations(self):
        return ('opro.exemplars', 'opro.register', 'opro.evaluate', 'opro.select')

    def required_profiles(self):
        return {'opro': 'artifact-search/v1'}

    def report(self):
        return self.emit.report()

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        # The optimizer sees an explicit score history, independent of the
        # Harness's conversational context budget.
        yield

    def score(self, instruction, split, round_index, *, label=None):
        return ('evaluate', {'candidate_id': candidate_id(instruction), 'indices': None,
                             'baseline': instruction == self.initial, 'split': split,
                          'round': round_index, 'label': label or split})

    def consume(self, window):
        feedback = window.feedbacks[0] if window.feedbacks else None
        decision = self.pending
        request = decision.operations[0].arguments
        if ((feedback is None or feedback.call_id != decision.operations[0].call_id)
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
                              'artifacts': [asdict(ref) for ref in feedback.artifacts]})
        if decision.operations[0].name == 'opro.exemplars':
            exemplars = data.get('exemplars')
            if not isinstance(exemplars, list) or len(exemplars) != len(request['indices']):
                raise RuntimeError('Invalid OPRO exemplar receipt')
            proposal = self.proposer.propose([record for record, _ in self.optimizer_history()], exemplars,
                                             round_index=self.round)
            self.optimizer_calls += self.parameters['candidates_per_round']
            self.emit('proposal', {'round': self.round, 'meta_prompt_sha256': proposal.meta_prompt_sha256,
                                   'evidence': asdict(proposal.evidence), 'candidates': list(proposal.candidates)})
            for instruction in proposal.candidates:
                if instruction in self.seen:
                    continue
                self.seen.add(instruction)
                self.candidate_count += 1
                self.emit('candidate', {'candidate_id': candidate_id(instruction),
                                        'instruction': instruction, 'round': self.round})
                self.queue.append(('register', {'candidate': {'instruction': instruction}}))
                self.queue.append(self.score(instruction, 'search', self.round))
                self.queue.append(self.score(instruction, 'validation', self.round))
            self.round += 1
        elif decision.operations[0].name == 'opro.evaluate':
            split, label = (request[key] for key in ('split', 'label'))
            instruction = next(item for item in self.seen if candidate_id(item) == request['candidate_id'])
            score, count = self.metric.value(data), data.get('count')
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

    def optimizer_history(self):
        """Scored search records the optimizer sees, with the decisions that produced them."""
        return self.history[:1] if self.parameters['condition'] == 'independent_candidates' else self.history

    def decide(self, window):
        if self.pending is not None:
            self.consume(window)
        dependencies = [self.last_decision_id] if self.last_decision_id else []
        if self.queue:
            operation, arguments = self.queue.popleft()
        elif self.round < self.parameters['rounds']:
            import numpy as np

            history = self.optimizer_history()
            indices = np.sort(np.random.RandomState(self.round).choice(
                self.counts['search'], self.parameters['exemplars'], replace=False)).tolist()
            # The Lab returns the exemplar problems; the optimizer step follows in consume().
            operation = 'exemplars'
            arguments = {'round': self.round, 'indices': indices}
            dependencies.extend(ident for _, ident in history)
        elif self.initial not in self.scores['validation']:
            operation, arguments = self.score(self.initial, 'validation', -1)
        elif self.selected is None:
            scores = self.scores['validation']
            self.selected = (max if self.metric.direction == 'maximize' else min)(
                [self.initial, *scores], key=scores.__getitem__)
            self.emit('selection', {'candidate_id': candidate_id(self.selected), 'instruction': self.selected,
                                    'phase': 'validation', 'score': scores[self.selected]})
            operation, arguments = 'select', {'candidate_id': candidate_id(self.selected)}
        else:
            decision = Decision('finish', reason='OPRO selection completed; host sealed evaluation follows',
                                depends_on_decision_ids=tuple(dependencies))
            self.record(decision)
            self.emit('finish', {'method': 'opro', 'condition': self.parameters['condition'],
                'rounds': self.round, 'unique_candidates': self.candidate_count,
                'initial_instruction': self.initial, 'selected_instruction': self.selected,
                'selected_candidate_id': candidate_id(self.selected),
                'validation_score': self.scores['validation'][self.selected],
                'sealed_evaluation': 'host_after_finish',
                'training_submissions': 0})
            return decision
        self.sequence += 1
        call = OperationCall('opro.' + operation, {'request_id': f'op-{self.sequence}', **arguments})
        self.pending = Decision.invoke(call, reason=f"OPRO {operation} / {arguments.get('label', self.round)}", depends_on_decision_ids=tuple(dict.fromkeys(dependencies)))
        self.record(self.pending)
        return self.pending

    def record(self, decision):
        self.emit('decision', {'decision_id': decision.decision_id, 'kind': decision.kind,
            'reason': decision.reason, 'depends_on_decision_ids': list(decision.depends_on_decision_ids),
            'operation': decision.operations[0].name if decision.operations else None,
            'request': decision.operations[0].arguments if decision.operations else None})
