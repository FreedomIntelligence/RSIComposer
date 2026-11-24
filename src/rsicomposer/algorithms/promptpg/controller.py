"""Choose PromptPG training batches, checkpoint and evaluation arms."""

from copy import deepcopy

from rsicomposer.api import (
    Decision, ExecutionStatus, OperationCall, emit_observation, observation_id,
    report_version, report_version_selection,
)

from ..shared.experiment import MethodEvents


class PromptPGController:
    def __init__(self, context, config):
        self.config = config
        self.events = MethodEvents(context, 'promptpg')
        self.flow = self.operations()
        self.pending = None
        # What the flow received from each receipt, in order. The flow is a generator, so
        # restore() rebuilds it by replaying these values on a fresh one.
        self.sent = []

    def state(self):
        return {'pending': self.pending, 'sent': self.sent, 'summary': self.events.summary}

    def restore(self, state):
        if self.pending is not None or self.sent:
            raise RuntimeError('PromptPG state restores into a newly constructed controller')
        if set(state) != {'pending', 'sent', 'summary'}:
            raise ValueError('Unexpected PromptPG controller state')
        events, self.events = self.events, lambda kind, data: None  # replay writes no events
        try:
            step = None
            if state['pending'] is not None:
                step = next(self.flow)
                for value in state['sent']:
                    step = self.flow.send(deepcopy(value))
        except StopIteration:
            step = None
        finally:
            self.events = events
        pending = state['pending']
        if pending is not None and step is not None and (
                'tabmwp.' + step[0] != pending.name or step[1] != pending.arguments):
            raise ValueError('Replayed PromptPG flow diverged from the saved request')
        self.pending, self.sent = pending, deepcopy(state['sent'])
        self.events.summary = state['summary']

    @staticmethod
    def received(name, data):
        """The receipt fields the flow reads."""
        if name == 'prepare':
            return {'train': data['train'], 'test': data['test']}
        if name in ('train_batch', 'evaluate'):
            return {'correct': data['correct']}
        return {}

    def required_operations(self):
        return tuple(f'tabmwp.{name}' for name in ('prepare', 'train_batch', 'save', 'restore', 'evaluate'))

    def report(self):
        return self.events.report()

    def operations(self):
        groups = yield 'prepare', {}, 'Prepare frozen TabMWP cohorts and selector'
        train, test = groups['train'], groups['test']
        batch, eval_batch = self.config['batch_size'], self.config['evaluation_batch_size']
        best_correct, best_epoch, steps = -1, None, 0
        for epoch in range(1, self.config['epochs'] + 1):
            correct = 0
            for start in range(0, len(train), batch):
                feedback = yield 'train_batch', {'ids': train[start:start + batch]}, f'PromptPG training epoch {epoch}'
                correct += feedback['correct']
                steps += 1
            if correct > best_correct:
                yield 'save', {'epoch': epoch}, 'Save the best training selector'
                best_correct, best_epoch = correct, epoch
        yield 'restore', {'epoch': best_epoch}, 'Restore the selected selector'
        results = {}
        for arm in ('trained', 'random', 'nearest'):
            correct = 0
            for start in range(0, len(test), eval_batch):
                receipt = yield 'evaluate', {'ids': test[start:start + eval_batch], 'arm': arm}, f'TabMWP evaluation: {arm}'
                correct += receipt['correct']
            results[arm] = {'correct': correct, 'total': len(test), 'accuracy': correct / len(test)}
        self.events('finish', {'method': 'promptpg', 'training_steps': steps,
            'best_train_epoch': best_epoch, 'heldout': results})

    def decide(self, window):
        feedback = window.feedbacks[0] if window.feedbacks else None
        try:
            if self.pending is None:
                name, args, reason = next(self.flow)
            else:
                receipt = feedback
                if (feedback is None or feedback.call_id != self.pending.call_id) or receipt is None or receipt.status != ExecutionStatus.COMPLETED:
                    raise RuntimeError('Expected a matching completed PromptPG Lab receipt')
                if receipt.data.get('request') != self.pending.arguments:
                    raise RuntimeError('PromptPG Lab receipt differs from its request')
                operation = self.pending.name.removeprefix('tabmwp.')
                if operation == 'prepare':
                    report_version('promptpg-initial', label='Initial selector', round_index=0, parent_ids=())
                elif operation in ('save', 'restore'):
                    epoch = self.pending.arguments['epoch']
                    version = f'promptpg-selector-{epoch}'
                    if operation == 'save':
                        # Later epochs continue from live weights, which may not have been saved.
                        report_version(version, label=f'Selector epoch {epoch}', round_index=epoch,
                                       parent_ids=('promptpg-initial',) if epoch == 1 else None)
                    else:
                        report_version_selection(version, step=epoch, event_id=receipt.call_id)
                value = self.received(operation, receipt.data)
                self.sent.append(value)
                name, args, reason = self.flow.send(deepcopy(value))
        except StopIteration:
            for arm, score in self.events.report()['heldout'].items():
                emit_observation('promptpg.heldout', observation_id('promptpg.heldout', arm),
                                 subject=arm, fields={'arm': arm, 'correct': score['correct'], 'total': score['total']})
            return Decision('finish', reason='PromptPG and held-out comparisons completed')
        self.pending = OperationCall('tabmwp.' + name, args)
        return Decision.invoke(self.pending, reason=reason)
