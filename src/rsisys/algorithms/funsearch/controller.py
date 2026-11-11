"""FunSearch policy. Only the Controller can change population and incumbent."""

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict

from rsisys.api import Decision, ExecutionStatus, OperationCall

from ..shared.experiment import MethodEvents
from .packing import BASES
from .upstream import Population, REVISION


class FunSearchController:
    def __init__(self, context, config):
        self.config = config
        self.emit = MethodEvents(context, 'funsearch')
        self.population = Population(seed=config['seed'], islands=config['islands'],
                                     reset_every=config['reset_every'], source_dir=config['source_dir'])
        self.queue = deque([('seed', {})])
        self.pending = None
        self.receipts = set()
        self.seen = set()
        self.stats = dict(proposals=0, parse_rejections=0, duplicates=0, invalid_scores=0,
                          valid_candidates=0, population_additions=0, adoptions=0, resets=0)
        self.baselines = {}
        self.development = {}
        self.heldout = {}
        self.best = None
        self.selected = False
        self.sequence = 0

    def required_operations(self):
        return ('funsearch.seed', 'funsearch.generate', 'funsearch.score')

    def report(self):
        return self.emit.report()

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        yield

    def score_request(self, record, split, label):
        return ('score', {'version': record['version'], 'split': split, 'label': label,
                          'candidate': record.get('index'), 'island': record.get('island'),
                          'generated_by': record.get('receipt_id')})

    def consume(self, window):
        decision = self.pending
        call, feedback = decision.operation, window.feedback
        if (window.last_operation != call or feedback is None
                or feedback.status != ExecutionStatus.COMPLETED):
            error = feedback.error if feedback is not None else None
            raise RuntimeError(f'FunSearch operation {call.name} did not complete: {error}')
        data = feedback.data
        receipt = data.get('receipt_id')
        if (data.get('request') != call.arguments or not isinstance(receipt, str)
                or not receipt or receipt in self.receipts):
            raise RuntimeError('FunSearch received an unmatched or duplicate Lab receipt')
        self.receipts.add(receipt)
        self.emit('receipt', {'receipt_id': receipt, 'decision_id': decision.decision_id,
                              'request_id': call.arguments['request_id'], 'operation': call.name,
                              'artifacts': [asdict(ref) for ref in feedback.artifacts]})
        operation = call.name.removeprefix('funsearch.')
        if operation == 'seed':
            self.baselines = data['versions']
            if set(self.baselines) != set(BASES):
                raise RuntimeError('Unexpected FunSearch seed receipt')
            self.seen.update(row['version']['digest'] for row in self.baselines.values())
            for label, row in self.baselines.items():
                self.queue.append(self.score_request(row, 'development', label))
        elif operation == 'generate':
            self.consume_generation(call.arguments, data, receipt)
        else:
            self.consume_score(call.arguments, data, receipt, decision.decision_id)
        return receipt

    def consume_generation(self, request, data, receipt):
        candidates = data['proposals']
        start, count = request['start'], request['count']
        if (len(candidates) != count or [row['index'] for row in candidates] != list(range(start, start + count))
                or start != self.stats['proposals']):
            raise RuntimeError('FunSearch receipt does not match proposal budget')
        self.stats['proposals'] += count
        for row in candidates:
            index, version = row['index'], row['version']
            if version is None:
                self.stats['parse_rejections'] += 1
                self.emit('candidate', {'index': index, 'status': 'parse_rejected',
                                       'generation_receipt_id': receipt})
                continue
            key = version['digest']
            is_duplicate = key in self.seen
            self.emit('candidate', {'index': index, 'version': version,
                                    'status': 'duplicate' if is_duplicate else 'queued',
                                    'generation_receipt_id': receipt, 'parents': request['parents']})
            if is_duplicate:
                self.stats['duplicates'] += 1
                continue
            self.seen.add(key)
            self.queue.append(self.score_request({**row, 'island': request['island'],
                                                  'receipt_id': receipt}, 'development', 'candidate'))

    def consume_score(self, request, data, receipt, decision_id):
        split, label = request['split'], request['label']
        score = data['score']
        self.emit('evaluation', {'receipt_id': receipt, 'phase': split, 'label': label,
            'candidate': request['candidate'], 'candidate_id': request['version']['digest'],
            'version': request['version'], 'score': score})
        if score is None:
            if split != 'development' or label != 'candidate':
                raise RuntimeError('Selected or baseline priority failed evaluation')
            self.stats['invalid_scores'] += 1
            return
        count = self.config['development_count' if split == 'development' else 'heldout_count']
        if (score['count'] != count or len(score['signature']) != count
                or any(type(value) is not int or value < 1 for value in score['signature'])
                or score['bins'] != sum(score['signature'])):
            raise RuntimeError('Malformed FunSearch score receipt')
        if split == 'heldout':
            self.heldout[label] = {'score': score, 'receipt_id': receipt}
            return
        row = {'version': request['version'], 'score': score, 'receipt_id': receipt,
               'candidate': request['candidate']}
        if label == 'first_fit':
            self.development[label] = row
            return
        if label == 'best_fit':
            self.best = self.development[label] = row
        else:
            self.stats['valid_candidates'] += 1
            if score['bins'] < self.best['score']['bins']:
                self.stats['adoptions'] += 1
                self.emit('adoption', {'previous': self.best['version'], 'current': row['version'],
                                       'receipt_id': receipt, 'score': score['bins']})
                self.best = row
        if label == 'best_fit' or self.config['condition'] == 'evolution':
            self.population.register(data['source'], row['version'], receipt, decision_id,
                                     score['signature'], request['island'])
            if label == 'candidate':
                self.stats['population_additions'] += 1

    def next_operation(self):
        if self.queue:
            return self.queue.popleft()
        completed = self.stats['proposals']
        if completed < self.config['proposals']:
            if self.config['condition'] == 'evolution' and self.population.reset(completed):
                self.stats['resets'] += 1
                self.emit('reset', {'after_proposals': completed})
            prompt, island, parents = self.population.prompt()
            count = min(self.config['samples_per_prompt'], self.config['proposals'] - completed)
            return 'generate', {'start': completed, 'count': count, 'island': island,
                                'prompt': prompt, 'parents': parents}
        if self.selected:
            return None, None
        self.selected = True
        self.emit('selection', {'source_split': 'development', **self.best})
        for label, row in (*self.baselines.items(), ('selected', self.best)):
            self.queue.append(self.score_request(row, 'heldout', label))
        return self.queue.popleft()

    def decide(self, window):
        previous = self.pending
        receipt = self.consume(window) if previous is not None else None
        dependencies = [previous.decision_id] if previous is not None else []
        operation, arguments = self.next_operation()
        if operation == 'generate':
            dependencies.extend(row['score_decision_id'] for row in arguments['parents'])
        dependencies = tuple(dict.fromkeys(dependencies))
        if operation is None:
            decision = Decision('finish', reason='FunSearch budgets and final held-out scores complete',
                                depends_on_decision_ids=dependencies)
            request = None
        else:
            self.sequence += 1
            request = {'request_id': f'op-{self.sequence}', **arguments}
            decision = Decision('invoke', operation=OperationCall('funsearch.' + operation, request),
                                reason=f'FunSearch {operation}', depends_on_decision_ids=dependencies)
        self.emit('decision', {'decision_id': decision.decision_id, 'consumed_receipt_id': receipt,
                               'reason': decision.reason,
                               'depends_on_decision_ids': list(dependencies), 'request': request,
                               'operation': decision.operation.name if request else None})
        self.pending = decision if operation is not None else None
        if operation is None:
            self.emit('finish', {'method': 'funsearch', 'condition': self.config['condition'],
                'seed': self.config['seed'], 'upstream_revision': REVISION,
                'proposal_statistics': self.stats, 'development_baselines': self.development,
                'selected': self.best, 'heldout': self.heldout, 'training_submissions': 0})
        return decision
