"""Adapt the existing exploration endpoint to the Lab task/receipt contract.

This adapter adds no actions, budgets or sandbox promises. The declared modes
must match installed handlers. Its task index is process-local; durable users
must include it in their explicit recovery integration.
"""
from copy import deepcopy

from rsisys.loops.contracts.lab import LabSpec, LabTask
from rsisys.loops.contracts.coordination import ExplorationProgress
from rsisys.loops.contracts.receipts import ExecutionReceipt


class ExplorationLab:
    def __init__(self, spec: LabSpec, exploration):
        if not isinstance(spec, LabSpec):
            raise TypeError('Expected LabSpec')
        self._spec, self._endpoint = deepcopy(spec), exploration
        self._tasks, self._keys = {}, {}
        self._accepted = set()

    def describe(self):
        return deepcopy(self._spec)

    def submit(self, task: LabTask):
        if not isinstance(task, LabTask):
            raise TypeError('Expected LabTask')
        task.__post_init__()
        if task.lab_id != self._spec.lab_id or task.directive.mode not in self._spec.capabilities:
            raise ValueError('Task targets a different Lab or unsupported capability')
        previous = self._tasks.get(task.task_id)
        if previous is not None and previous != task:
            raise ValueError('Task identity reused with different content')
        if task.idempotency_key in self._keys and self._keys[task.idempotency_key] != task:
            raise ValueError('Idempotency key reused with different content')
        if task.task_id in self._accepted:
            return self.inspect(task.task_id)
        # Claim before submission: a lost accept response must retry the exact
        # same immutable request/key, never silently mint another task.
        self._tasks[task.task_id] = deepcopy(task)
        self._keys[task.idempotency_key] = deepcopy(task)
        progress = self._endpoint.accept(deepcopy(task.directive))
        progress = self._validate_progress(task, progress)
        self._accepted.add(task.task_id)
        return progress

    def _task(self, task_id):
        return self._tasks[task_id]

    @staticmethod
    def _validate_progress(task, progress):
        if not isinstance(progress, ExplorationProgress):
            raise TypeError('Lab endpoint must return ExplorationProgress')
        progress.__post_init__()
        if progress.directive != task.directive:
            raise ValueError('Lab endpoint returned a different directive')
        return deepcopy(progress)

    def inspect(self, task_id):
        return self._validate_progress(self._task(task_id), self._endpoint.inspect(task_id))

    def advance(self, task_id):
        return self._validate_progress(self._task(task_id), self._endpoint.advance(task_id))

    def cancel(self, task_id):
        return self._validate_progress(self._task(task_id), self._endpoint.cancel(task_id))

    def result(self, task_id):
        task = self._task(task_id)
        progress = self.inspect(task_id)
        if not progress.terminal:
            raise ValueError('Lab task has no matching terminal receipt')
        return ExecutionReceipt.from_feedback(task.directive, progress.feedback,
            decision_id=task.decision_id, lab_id=task.lab_id, run_id=task.run_id, agent_id=task.agent_id,
            issuing_model=task.issuing_model)

    def acknowledge(self, task_id):
        self.result(task_id)  # Do not acknowledge unfinished or mismatched work.
        self._endpoint.acknowledge(task_id)
