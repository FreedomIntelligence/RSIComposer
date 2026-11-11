"""Reflexion retry policy over independently executed HumanEval operations."""
from contextlib import contextmanager
from dataclasses import asdict

from rsisys.loops.contracts.coordination import ExecutionStatus
from rsisys.loops.contracts.decision import Decision, OperationCall

from ..shared.experiment import MethodEvents


class ReflexionController:
    def __init__(self, context, options):
        self.options = options
        self.emit = MethodEvents(context, 'reflexion')
        self.task, self.attempt, self.sequence = 0, 1, 0
        self.program = self.tests = self.feedback = None
        self.pending = None
        self.receipts = set()
        self.results = []

    def required_operations(self):
        return tuple('reflexion.' + name for name in ('tests', 'implement', 'execute', 'reflect', 'hidden'))

    def report(self):
        return self.emit.report()

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        yield

    def decide(self, window):
        last = self.pending
        if last is None:
            operation, args = 'tests', {}
        else:
            operation, args = self.consume(window)
        dependencies = (last.decision_id,) if last else ()
        if operation is None:
            decision = Decision('finish', reason='Reflexion HumanEval evaluation completed',
                                depends_on_decision_ids=dependencies)
            request = None
        else:
            self.sequence += 1
            request = {'request_id': f'op-{self.sequence}', 'task': self.task, 'attempt': self.attempt, **args}
            call = OperationCall('reflexion.' + operation, request)
            decision = Decision('invoke', operation=call,
                                reason=f'Reflexion task {self.task + 1}, attempt {self.attempt}: {operation}',
                                depends_on_decision_ids=dependencies)
        self.emit('decision', {'decision_id': decision.decision_id, 'kind': decision.kind,
            'operation': decision.operation.name if request is not None else None, 'request': request,
            'reason': decision.reason, 'depends_on_decision_ids': list(dependencies)})
        if operation is None:
            self.emit('finish', {'method': 'reflexion', 'condition': self.options['condition'],
                'tasks': len(self.results), 'success_rate': sum(row['passed'] for row in self.results) / len(self.results),
                'task_results': self.results, 'training_submissions': 0})
        else:
            self.pending = decision
        return decision

    def consume(self, window):
        decision = self.pending
        call, feedback = decision.operation, window.feedback
        if (window.last_operation != call or feedback is None
                or feedback.status != ExecutionStatus.COMPLETED):
            raise RuntimeError('Reflexion operation did not complete')
        data = feedback.data
        receipt = data.get('receipt_id')
        if (data.get('request') != call.arguments or not isinstance(receipt, str)
                or not receipt or receipt in self.receipts):
            raise RuntimeError('Reflexion received an unmatched or duplicate Lab receipt')
        self.receipts.add(receipt)
        self.emit('receipt', {'receipt_id': receipt, 'decision_id': decision.decision_id,
            'request_id': call.arguments['request_id'], 'public': data['public'],
            'artifacts': [asdict(ref) for ref in feedback.artifacts]})

        name = call.name.split('.', 1)[1]
        if name == 'tests':
            self.tests = data['tests']
            return 'implement', {}
        if name == 'implement':
            self.program = data['program']
            self.emit('candidate', {'candidate_id': 'code-' + self.program['digest'][:20],
                'task': self.task, 'round': self.attempt})
            return 'execute', {'program': self.program, 'tests': self.tests}
        if name == 'execute':
            if type(data['passed']) is not bool:
                raise RuntimeError('Invalid Reflexion internal-test result')
            if data['passed'] or self.attempt >= self.options['max_iters']:
                return 'hidden', {'program': self.program}
            self.feedback = data['feedback']
            if self.options['condition'] == 'reflexion':
                return 'reflect', {'program': self.program, 'feedback': self.feedback}
            return self.retry(None)
        if name == 'reflect':
            return self.retry(data['reflection'])

        if name != 'hidden' or type(data['passed']) is not bool:
            raise RuntimeError('Invalid Reflexion final-test result')
        self.results.append({'task_id': data['task_id'], 'attempts': self.attempt, 'passed': data['passed']})
        self.emit('evaluation', {'candidate_id': 'code-' + self.program['digest'][:20],
            'task': self.task, 'phase': 'hidden', 'round': self.attempt, 'count': 1,
            'score': int(data['passed']), 'protocol': 'reflexion-official-humaneval', 'receipt_id': receipt})
        self.task += 1
        self.attempt = 1
        self.program = self.tests = self.feedback = None
        return ('tests', {}) if self.task < self.options['count'] else (None, {})

    def retry(self, reflection):
        self.attempt += 1
        return 'implement', {'program': self.program, 'feedback': self.feedback, 'reflection': reflection}
