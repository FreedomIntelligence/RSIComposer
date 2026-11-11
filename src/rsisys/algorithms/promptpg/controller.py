"""Choose PromptPG training batches, checkpoint and evaluation arms."""

from rsisys.api import Decision, ExecutionStatus, LabRequest

from ..shared.experiment import MethodEvents


class PromptPGController:
    def __init__(self, context, config):
        self.config = config
        self.events = MethodEvents(context, 'promptpg')
        self.flow = self.operations()
        self.pending = None

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
        try:
            if self.pending is None:
                name, args, reason = next(self.flow)
            else:
                receipt = window.feedback
                if window.last_operation != self.pending or receipt is None or receipt.status != ExecutionStatus.COMPLETED:
                    raise RuntimeError('Expected a matching completed PromptPG Lab receipt')
                if receipt.data.get('request') != self.pending.arguments:
                    raise RuntimeError('PromptPG Lab receipt differs from its request')
                name, args, reason = self.flow.send(receipt.data)
        except StopIteration:
            return Decision('finish', reason='PromptPG and held-out comparisons completed')
        self.pending = LabRequest('tabmwp.' + name, args)
        return Decision('invoke', operation=self.pending, reason=reason)
