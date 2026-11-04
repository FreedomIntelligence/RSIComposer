"""Execution facts shared by Labs, local tools and feedback observers.

The original request/result remain addressable. This envelope is neither a
generated interpretation nor authorization to apply a proposed change.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Literal

from .directive import Directive
from .feedback import Feedback
from .lifecycle import ExecutionStatus
from .metrics import validated_metrics
from .model_service import ModelRef
from .jobs import JobRef


@dataclass(frozen=True)
class ExecutionReceipt:
    task_id: str
    objective: str
    source: Literal['operation', 'exploration']
    request: dict
    result: dict
    decision_id: str | None = None
    lab_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    issuing_model: ModelRef | None = None
    read_only: bool = False
    job: JobRef | None = None

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError('Receipt task_id must be nonempty')
        if not isinstance(self.objective, str):
            raise TypeError('Receipt objective must be a string')
        for name in ('decision_id', 'lab_id', 'run_id', 'agent_id'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'Receipt {name} must be nonempty or None')
        if self.source not in ('operation', 'exploration') or type(self.read_only) is not bool:
            raise ValueError('Invalid receipt source or read_only flag')
        if self.issuing_model is not None and not isinstance(self.issuing_model, ModelRef):
            raise TypeError('issuing_model must be ModelRef or None')
        if self.job is not None and not isinstance(self.job, JobRef):
            raise TypeError('Receipt job must be JobRef or None')
        if not isinstance(self.request, dict) or not isinstance(self.result, dict):
            raise TypeError('Receipt request/result must be mappings')
        # Revalidate typed results after deserialization as well as construction.
        if self.source == 'exploration':
            if self.request.get('directive_id') != self.task_id or self.result.get('directive_id') != self.task_id:
                raise ValueError('Receipt request/result task identities differ')
        status = self.result.get('status')
        if status not in tuple(ExecutionStatus):
            raise ValueError('Receipt must contain an execution terminal status')
        if self.source == 'operation' and status == ExecutionStatus.PARTIAL:
            raise ValueError('An operation receipt cannot be partial')
        if type(self.result.get('simulated')) is not bool or not isinstance(self.result.get('summary'), str):
            raise ValueError('Receipt result requires summary and simulated fields')
        if status == ExecutionStatus.FAILED and not self.result.get('error'):
            raise ValueError('A failed receipt requires an error')
        if status == ExecutionStatus.COMPLETED and self.result.get('error') is not None:
            raise ValueError('A completed receipt cannot carry an error')
        if self.result.get('adopted_candidate_ids') and status != ExecutionStatus.COMPLETED:
            raise ValueError('Only completed receipts may confirm adoption')
        metrics = self.result if self.source == 'exploration' else self.result.get('data', {})
        for name in ('quality_metrics', 'training_metrics', 'execution_metrics'):
            validated_metrics(metrics.get(name, {}), name)
        json.dumps({'request': self.request, 'result': self.result}, allow_nan=False)
        object.__setattr__(self, 'request', deepcopy(self.request))
        object.__setattr__(self, 'result', deepcopy(self.result))

    @classmethod
    def from_feedback(cls, directive: Directive, feedback: Feedback, **identity):
        directive.__post_init__()
        feedback.__post_init__()
        return cls(directive.directive_id, directive.objective, 'exploration',
                   asdict(directive), asdict(feedback), **identity)

    @classmethod
    def from_operation(cls, task_id, objective, call, result, **identity):
        # Importing coordination here avoids making the core coordination
        # protocols depend on the optional observation envelope.
        from .coordination import OperationResult
        if not isinstance(result, OperationResult):
            raise TypeError('Expected OperationResult')
        result.__post_init__()
        return cls(task_id, objective, 'operation', asdict(call) if call is not None else {},
                   asdict(result), **identity)

    def payload(self):
        """Preserve the historical exploration journal format for old readers."""
        if self.source == 'exploration':
            value = {'directive': deepcopy(self.request), 'feedback': deepcopy(self.result)}
        else:
            value = {'operation': deepcopy(self.request), 'result': deepcopy(self.result),
                     'task': {'task_id': self.task_id, 'objective': self.objective,
                              'decision_id': self.decision_id, 'read_only': self.read_only}}
            value['task']['issuing_model'] = asdict(self.issuing_model) if self.issuing_model else None
            value['task']['job'] = asdict(self.job) if self.job else None
        identity = (self.lab_id, self.run_id, self.agent_id)
        if self.source == 'exploration':
            identity += (self.decision_id, self.issuing_model)
        if any(value is not None for value in identity):
            value['identity'] = {name: getattr(self, name) for name in
                                 ('lab_id', 'run_id', 'agent_id', 'decision_id')}
            value['identity']['issuing_model'] = asdict(self.issuing_model) if self.issuing_model else None
        return value


def receipt_task(receipt):
    """Task metadata for both legacy exploration and common operation receipts."""
    if 'directive' in receipt:
        return receipt['directive']
    return {**receipt['task'], 'mode': 'operation', 'directive_id': None,
            'simulated': receipt['result']['simulated']}


def receipt_facts(receipt):
    """Measured aggregates only; arbitrary tool data stays in the raw result.

    Operation metrics use the documented data.<group> layout. They do not imply
    any candidate/protocol binding or turn a query into a new evaluation.
    """
    if 'feedback' in receipt:
        return deepcopy(receipt['feedback'])
    result, task = receipt['result'], receipt['task']
    data = result.get('data', {})
    return {'task_id': task['task_id'], 'decision_id': task.get('decision_id'),
            'source': 'operation', 'status': result['status'], 'summary': result['summary'],
            'simulated': result['simulated'], 'error': result.get('error'),
            'quality_metrics': deepcopy(data.get('quality_metrics', {})),
            'training_metrics': deepcopy(data.get('training_metrics', {})),
            'execution_metrics': deepcopy(data.get('execution_metrics', {})),
            'artifact_refs': deepcopy(result.get('artifacts', ())),
            'evidence_refs': deepcopy(data.get('evidence_refs', ())), 'candidates': [],
            'adopted_candidate_ids': deepcopy(result.get('adopted_candidate_ids', ()))}
