"""FunSearch policy. Only the Controller can change population and incumbent.

The Controller samples new priority programs from the decision model with its
own concurrent samplers (sampler.py) while the Lab evaluates earlier ones.
inference_ready() lets the host call decide as soon as a sampling request
finishes, even while Lab evaluations are still running. With no Lab work
outstanding, a managed research wait gives control back to the host.
"""

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4

from rsicomposer.api import (Decision, ExecutionStatus, JobOutcome, JobRef, JobStatus, OperationCall,
                             report_version, report_version_evaluation, report_version_selection)

from ..shared.experiment import MethodEvents
from .packing import BASES
from .sampler import FunSearchSampler
from .upstream import Population, REVISION


class FunSearchController:
    def __init__(self, context, config, *, sampler=None):
        self.metric = context.lab_catalog.primary_metric('funsearch')
        self.config = config
        self.emit = MethodEvents(context, 'funsearch')
        self.population = Population(seed=config['seed'], islands=config['islands'],
                                     reset_every=config['reset_every'], source_dir=config['source_dir'])
        self.queue = deque([('task_spec', {})])
        self.pending = {}
        self.receipts = set()
        self.seen = set()
        self.registered = set()
        self.stats = dict(proposals=0, parse_rejections=0, duplicates=0, invalid_scores=0,
                          valid_candidates=0, population_additions=0, adoptions=0, resets=0)
        self.baselines = {}
        self.development = {}
        self.best = None
        self.selected = False
        self.sequence = 0
        self.issued_proposals = 0
        self.reset_cursor = 0
        self.sampling = {}  # request_id -> {'request', 'decision_id'} for in-flight sampling requests.
        self.samplers = config.get('samplers', 2)
        self.evaluators = config.get('evaluators', 2)
        for value in (self.samplers, self.evaluators, config['samples_per_prompt'], config['reset_every']):
            if type(value) is not int or value < 1:
                raise ValueError('FunSearch sampler/evaluator and batch/reset counts must be positive integers')
        self.sampler = sampler or FunSearchSampler(context, config, workers=self.samplers)
        self._research_job = JobRef(str(uuid4()), 'funsearch-sampler')
        self._stopping = False

    _FIELDS = ('pending', 'stats', 'baselines', 'development', 'best', 'selected', 'sequence',
               'issued_proposals', 'reset_cursor', 'sampling')

    def state(self):
        return {**{name: getattr(self, name) for name in self._FIELDS},
                'queue': [[operation, arguments] for operation, arguments in self.queue],
                'receipts': sorted(self.receipts), 'seen': sorted(self.seen),
                'registered': sorted(self.registered), 'population': self.population.state(), 'summary': self.emit.summary,
                'research_job_id': self._research_job.job_id, 'stopping': self._stopping}

    def restore(self, state):
        previous = {*self._FIELDS, 'queue', 'receipts', 'seen', 'registered', 'population', 'summary'}
        if set(state) not in (previous, previous | {'research_job_id', 'stopping'}):
            raise ValueError('Unexpected FunSearch controller state')
        job_id = state.get('research_job_id', self._research_job.job_id)
        stopping = state.get('stopping', False)
        if (not isinstance(job_id, str) or not job_id.strip() or type(stopping) is not bool
                or (stopping and state['sampling'])):
            raise ValueError('Invalid FunSearch research state')
        self._research_job = JobRef(job_id, 'funsearch-sampler')
        self.population.restore(state['population'])
        for name in self._FIELDS:
            setattr(self, name, deepcopy(state[name]))
        self.queue = deque((operation, deepcopy(arguments)) for operation, arguments in state['queue'])
        self.receipts, self.seen = set(state['receipts']), set(state['seen'])
        self.registered = set(state['registered'])
        self.emit.summary = state['summary']
        # Completed proposals were charged; in-flight requests are asked again with the same seeds.
        self.sampler.calls.count = self.stats['proposals']
        if stopping:
            self._stop_research()
        else:
            for key, row in self.sampling.items():
                self.sampler.submit(key, row['request'])

    def required_operations(self):
        return ('funsearch.task_spec', 'funsearch.register', 'funsearch.evaluate', 'funsearch.select')

    def required_profiles(self):
        return {'funsearch': 'artifact-search/v1'}

    def report(self):
        return self.emit.report()

    def inference_ready(self):
        return not self._stopping and bool(self.sampler.finished())

    def pending_research_job(self):
        return self._research_job if self.sampler.jobs else None

    def can_poll_job(self, job):
        return job == self._research_job

    def poll_job(self, job):
        if not self.can_poll_job(job):
            raise ValueError('FunSearch does not own this research job')
        if self._stopping:
            status = JobStatus.CANCELLED if self.sampler.stop() else JobStatus.RUNNING
            return JobOutcome(job, status)
        if self.inference_ready():
            return JobOutcome(job, JobStatus.SUCCEEDED, True)
        return JobOutcome(job, JobStatus.RUNNING)

    def cancel_job(self, job):
        if not self.can_poll_job(job):
            raise ValueError('FunSearch does not own this research job')
        status = JobStatus.CANCELLED if self._stop_research() else JobStatus.RUNNING
        return JobOutcome(job, status)

    def _mark_stopping(self):
        self._stopping = True
        self.sampling.clear()  # Unpublished batches must not restart on durable recovery.

    def _stop_research(self):
        self._mark_stopping()
        return self.sampler.stop()

    # Both stop modes allow an already admitted provider call to exit but
    # discard uncommitted results and never admit another proposal.
    drain_research = cancel_research = _stop_research

    def close(self):
        self._mark_stopping()
        self.sampler.close()

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        yield

    def score_request(self, record, split, label):
        return ('evaluate', {'candidate_id': record['version']['digest'], 'version': record['version'],
                             'indices': None, 'baseline': label in BASES, 'split': split, 'label': label,
                          'candidate': record.get('index'), 'island': record.get('island'),
                          'generated_by': record.get('receipt_id')})

    def consume(self, feedback):
        pending = self.pending.pop(feedback.call_id, None)
        if pending is None:
            raise RuntimeError('FunSearch received an unknown or already consumed call_id')
        call, decision_id = pending['call'], pending['decision_id']
        if feedback.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f'FunSearch operation {call.name} did not complete: {feedback.error}')
        data = feedback.data
        receipt = data.get('receipt_id')
        if (data.get('request') != call.arguments or not isinstance(receipt, str)
                or not receipt or receipt in self.receipts):
            raise RuntimeError('FunSearch received an unmatched or duplicate Lab receipt')
        self.receipts.add(receipt)
        self.emit('receipt', {'receipt_id': receipt, 'decision_id': decision_id,
                              'request_id': call.arguments['request_id'], 'operation': call.name,
                              'artifacts': [asdict(ref) for ref in feedback.artifacts]})
        operation = call.name.removeprefix('funsearch.')
        if operation == 'task_spec':
            self.baselines = data['versions']
            if set(self.baselines) != set(BASES):
                raise RuntimeError('Unexpected FunSearch seed receipt')
            self.seen.update(row['version']['digest'] for row in self.baselines.values())
            for label, row in self.baselines.items():
                report_version(row['version']['digest'], label=label, parent_ids=(),
                               source_decision_ids=(decision_id,))
                self.queue.append(self.score_request(row, 'development', label))
        elif operation == 'register':
            self.registered.add(data['candidate_id'])
            self.queue.appendleft(pending['followup'])
        elif operation == 'evaluate':
            self.consume_score(call.arguments, data, receipt, decision_id)
            if data['score'] is not None:
                report_version_evaluation(call.arguments['candidate_id'], feedback.call_id)
        elif operation == 'select':
            report_version_selection(call.arguments['candidate_id'], step=self.stats['proposals'],
                                     event_id=feedback.call_id)
        return receipt

    def consume_generation(self, request, data, receipt):
        candidates = data['proposals']
        start, count = request['start'], request['count']
        if (len(candidates) != count or [row['index'] for row in candidates] != list(range(start, start + count))
                or start < 0 or start + count > self.issued_proposals):
            raise RuntimeError('FunSearch receipt does not match proposal budget')
        self.stats['proposals'] += count
        for row in candidates:
            index, version = row['index'], row['version']
            if version is None:
                event = {'index': index, 'generation_receipt_id': receipt}
                if 'generation_error' in row:
                    event.update(status='generation_rejected', reason=row['generation_error'])
                else:
                    self.stats['parse_rejections'] += 1
                    event['status'] = 'parse_rejected'
                self.emit('candidate', event)
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
            parents = request['parents']
            report_version(key, label=f'Candidate {index + 1}',
                           parent_ids=tuple(dict.fromkeys(parent['version']['digest'] for parent in parents)),
                           source_decision_ids=tuple(dict.fromkeys(parent['score_decision_id'] for parent in parents)))
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
        row = {'version': request['version'], 'score': score, 'receipt_id': receipt,
               'candidate': request['candidate'], 'metric_value': self.metric.value(data)}
        if label == 'first_fit':
            self.development[label] = row
            return
        if label == 'best_fit':
            self.best = self.development[label] = row
        else:
            self.stats['valid_candidates'] += 1
            if ((row['metric_value'] < self.best['metric_value']) if self.metric.direction == 'minimize'
                    else (row['metric_value'] > self.best['metric_value'])):
                self.stats['adoptions'] += 1
                self.emit('adoption', {'previous': self.best['version'], 'current': row['version'],
                                       'receipt_id': receipt, 'score': score['bins']})
                self.best = row
        if label == 'best_fit' or self.config['condition'] == 'evolution':
            self.population.register(data['source'], row['version'], receipt, decision_id,
                                     score['signature'], request['island'])
            if label == 'candidate':
                self.stats['population_additions'] += 1

    def consume_samples(self, sources, receipts):
        for key in self.sampler.finished():
            row, result = self.sampling.pop(key), self.sampler.take(key)
            receipt = result['receipt_id']
            if receipt in self.receipts:
                raise RuntimeError('FunSearch received a duplicate sampling result')
            self.receipts.add(receipt)
            self.emit('receipt', {'receipt_id': receipt, 'decision_id': row['decision_id'],
                                  'request_id': key, 'operation': 'sample',
                                  'artifacts': [asdict(ref) for ref in result['artifacts']]})
            self.consume_generation(row['request'], {'proposals': result['proposals']}, receipt)
            sources.append(row['decision_id'])
            receipts.append(receipt)

    def decide(self, window):
        if self._stopping:
            raise RuntimeError('Cannot decide after FunSearch research has stopped')
        sources, receipts, started = [], [], []
        for feedback in window.feedbacks:
            pending = self.pending.get(feedback.call_id)
            if pending is not None:
                sources.append(pending['decision_id'])
            receipts.append(self.consume(feedback))
        if set(window.pending) != set(self.pending):
            raise RuntimeError('FunSearch host pending calls differ from its durable controller state')
        operations, metadata = [], []

        def issue(operation, arguments, followup=None):
            self.sequence += 1
            call = OperationCall('funsearch.' + operation, {'request_id': f'op-{self.sequence}', **arguments})
            operations.append(call)
            metadata.append({'call': call, 'followup': followup})

        self.consume_samples(sources, receipts)
        self.dispatch(issue, sources, started)
        sources = tuple(dict.fromkeys(ident for ident in sources if ident is not None))
        if operations:
            decision = Decision('invoke', operations=tuple(operations), reason='FunSearch asynchronous dispatch',
                                source_decision_ids=sources)
            for row in metadata:
                self.pending[row['call'].call_id] = {**row, 'decision_id': decision.decision_id}
        elif self.pending:
            decision = Decision('wait', reason='FunSearch awaiting evaluator receipts', source_decision_ids=sources)
        elif self.sampling:
            decision = Decision('wait', job=self._research_job, reason='FunSearch awaiting sampling',
                                source_decision_ids=sources)
        else:
            decision = Decision('finish', reason='FunSearch search complete; host sealed evaluation follows',
                                source_decision_ids=sources)
            self.emit('finish', {'method': 'funsearch', 'condition': self.config['condition'],
                'seed': self.config['seed'], 'upstream_revision': REVISION,
                'proposal_statistics': self.stats, 'development_baselines': self.development,
                'selected': self.best, 'sealed_evaluation': 'host_after_finish', 'training_submissions': 0})
        for key in started:
            if key in self.sampling:
                self.sampling[key]['decision_id'] = decision.decision_id
        self.emit('decision', {'decision_id': decision.decision_id, 'consumed_receipt_ids': receipts,
            'reason': decision.reason, 'source_decision_ids': list(sources),
            'operations': [asdict(call) for call in operations], 'sampling': started})
        return decision

    def dispatch(self, issue, sources, started):
        """Queue evaluations up to the evaluator limit and start samplers up to the sampler limit."""
        if not self.baselines:
            if not self.pending:
                self.queue.popleft()
                issue('task_spec', {})
            return
        active_evaluators = sum(row['call'].name.endswith(('.evaluate', '.register'))
                                for row in self.pending.values())
        # Registration occupies an evaluator slot and carries its follow-up;
        # an evaluate can never race ahead of registration.
        while self.queue and active_evaluators < self.evaluators:
            operation, args = self.queue.popleft()
            if operation != 'evaluate':
                raise RuntimeError('Unexpected FunSearch evaluation queue entry')
            if args['candidate_id'] not in self.registered:
                issue('register', {'candidate': {'version': args['version']}}, (operation, args))
            else:
                issue(operation, args)
            active_evaluators += 1
        if set(self.development) == set(BASES):
            completed = self.stats['proposals']
            if self.config['condition'] == 'evolution':
                while (self.reset_cursor + 1) * self.config['reset_every'] <= completed:
                    self.reset_cursor += 1
                    boundary = self.reset_cursor * self.config['reset_every']
                    if self.population.reset(boundary):
                        self.stats['resets'] += 1
                        self.emit('reset', {'after_proposals': boundary})
            while len(self.sampling) < self.samplers and self.issued_proposals < self.config['proposals']:
                # Snapshot the current population once per request. Outstanding
                # samplers deliberately keep their older prompt/database view.
                prompt, island, parents = self.population.prompt()
                count = min(self.config['samples_per_prompt'], self.config['proposals'] - self.issued_proposals)
                self.sequence += 1
                key = f'sample-{self.sequence}'
                request = {'request_id': key, 'start': self.issued_proposals, 'count': count,
                           'island': island, 'prompt': prompt, 'parents': parents}
                sources.extend(row['score_decision_id'] for row in parents)
                self.issued_proposals += count
                # The launching decision's ID is filled in once the decision exists.
                self.sampling[key] = {'request': request, 'decision_id': None}
                self.sampler.submit(key, request)
                started.append(key)
        if (not active_evaluators and not self.pending and not self.sampling and not self.queue and not self.selected
                and self.stats['proposals'] == self.config['proposals']):
            self.selected = True
            self.emit('selection', {'source_split': 'development', **self.best})
            issue('select', {'candidate_id': self.best['version']['digest']})
