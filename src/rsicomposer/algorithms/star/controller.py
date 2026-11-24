"""Schedule STaR from matched Lab receipts and select a trained checkpoint."""

from copy import deepcopy

from rsicomposer.api import Decision, ExecutionStatus, OperationCall

from .protocol import curriculum
from ..shared.weight_observation import observe_weight_receipt


class StarController:
    def __init__(self, context, config):
        self.schedule = curriculum(config['rounds'], config['plain_rounds'])
        self.lab = config.get('lab', 'star')
        self.initial_model = config.get('initial_model', 'base')
        self.train_data, self.eval_data = config.get('train_data', 'train'), config.get('eval_data', 'dev')
        self.samples_per_prompt, self.temperature = config.get('samples_per_prompt', 1), config.get('temperature', 0.)
        self.metric = context.lab_catalog.primary_metric(self.lab)
        if self.metric is None:
            raise ValueError('STaR requires the Lab primary metric declaration')
        self.flow = self.operations()
        self.pending = None
        self.result = None
        # Receipts sent into the round generator, in order. A generator cannot be
        # saved, so restore() rebuilds it by replaying these receipts.
        self.replies = []

    def required_profiles(self):
        return {self.lab: 'round-weight-training/v1'}

    def required_operations(self):
        return tuple(f'{self.lab}.{name}' for name in ('evaluate', 'generate', 'filter', 'train', 'select'))

    def report(self):
        return self.result or {'method': 'star', 'status': 'running'}

    def operations(self):
        baseline_dev = yield 'evaluate', {'round': 0, 'split': 'dev', 'model': self.initial_model,
            'dataset': self.eval_data, 'baseline': True}, 'Evaluate the initial development baseline'
        best_model, best_value = self.initial_model, self.metric.value(baseline_dev)
        previous, rounds = self.initial_model, []
        for number, rationalize, steps in self.schedule:
            generated = yield 'generate', {'round': number, 'model': previous,
                'dataset': self.train_data, 'samples_per_prompt': self.samples_per_prompt,
                'temperature': self.temperature, 'condition': 'answer' if rationalize else 'none'}, 'Generate rationales'
            filtered = yield 'filter', {'round': number, 'samples': generated['dataset'],
                'max_accepted_per_prompt': self.samples_per_prompt}, 'Filter accepted rationales'
            fitted = yield 'train', {'round': number, 'dataset': filtered['dataset'],
                'parent': self.initial_model, 'steps': steps}, 'Train from the explicit initial parent'
            previous = fitted['model']
            measured = yield 'evaluate', {'round': number, 'split': 'dev', 'model': previous,
                'dataset': self.eval_data, 'baseline': False}, 'Evaluate trained development quality'
            value = self.metric.value(measured)
            rounds.append({'round': number, 'model': previous,
                'generated_count': generated['count'], 'accepted_count': filtered['count'],
                'development': measured['quality_metrics']})
            if (value > best_value) if self.metric.direction == 'maximize' else (value < best_value):
                best_model, best_value = previous, value
        selected = (max if self.metric.direction == 'maximize' else min)(
            rounds, key=lambda row: self.metric.value({'quality_metrics': row['development']}))
        yield 'select', {'round': self.schedule[-1][0], 'model': selected['model']}, 'Record selection for host final evaluation'
        self.result = {'status': 'completed', 'rounds': rounds, 'best_including_base': best_model,
            'baseline_dev': baseline_dev['quality_metrics'], 'selected': selected['model'],
            'sealed_evaluation': 'host_after_finish'}

    def state(self):
        return {'replies': deepcopy(self.replies), 'pending': deepcopy(self.pending),
                'result': deepcopy(self.result)}

    def restore(self, state):
        if set(state) != {'replies', 'pending', 'result'}:
            raise ValueError('Unexpected STaR controller state')
        replies, pending = state['replies'], state['pending']
        if pending is None and replies:
            raise ValueError('STaR state has receipts but no pending operation')
        self.result = None
        flow, step, finished = self.operations(), None, False
        if pending is not None:
            step = next(flow)
            for index, reply in enumerate(replies):
                if reply['request'] != step[1]:
                    raise ValueError('Saved STaR receipt does not match the replayed request')
                try:
                    step = flow.send(deepcopy(reply))
                except StopIteration:
                    if index != len(replies) - 1:
                        raise ValueError('Saved STaR receipts continue past the end of the schedule') from None
                    finished = True
            if OperationCall(f'{self.lab}.{step[0]}', step[1]) != pending:
                raise ValueError('Saved STaR pending operation differs from the configured schedule')
        if self.result != state['result'] or finished != (state['result'] is not None):
            raise ValueError('Saved STaR result differs from the replayed receipts')
        # pending keeps its call_id, which the host uses to match the in-flight operation.
        self.flow, self.replies, self.pending = flow, deepcopy(replies), deepcopy(pending)
        self.result = deepcopy(state['result'])

    def decide(self, window):
        feedback = window.feedbacks[0] if window.feedbacks else None
        try:
            if self.pending is None:
                name, request, reason = next(self.flow)
            else:
                if ((feedback is None or feedback.call_id != self.pending.call_id) or feedback.status != ExecutionStatus.COMPLETED
                        or feedback.data["request"] != self.pending.arguments):
                    raise RuntimeError("STaR feedback differs from its request")
                observe_weight_receipt('star', self.pending, feedback.data)
                self.replies.append(deepcopy(feedback.data))
                name, request, reason = self.flow.send(feedback.data)
        except StopIteration:
            return Decision("finish", reason="STaR search completed; sealed evaluation belongs to the host")
        self.pending = OperationCall(f"{self.lab}.{name}", request)
        return Decision.invoke(self.pending, reason=reason)
