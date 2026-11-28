"""Execution facts shared by Labs, local tools and feedback observers.

The original request/result remain addressable. This envelope is neither a
generated interpretation nor authorization to apply a proposed change.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Literal

from .lifecycle import ExecutionStatus
from .metrics import has_scored_metrics, validated_metrics
from .evaluation import canonical_refs
from .model_service import ModelRef
from .jobs import JobRef
from .delivery import FeedbackDelivery


@dataclass(frozen=True)
class ExecutionReceipt:
    task_id: str
    objective: str
    source: Literal['operation', 'lab']
    request: dict
    result: dict
    decision_id: str | None = None
    lab_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    issuing_model: ModelRef | None = None
    read_only: bool = False
    job: JobRef | None = None
    feedback_delivery: FeedbackDelivery | None = None
    measured_evaluation: bool = False  # Delivery metadata, never read from the operation result.

    def __post_init__(self):
        if type(self.measured_evaluation) is not bool:
            raise TypeError('measured_evaluation must be boolean')
        if self.feedback_delivery is not None:
            if not isinstance(self.feedback_delivery, FeedbackDelivery):
                raise TypeError('feedback_delivery must be FeedbackDelivery or None')
            self.feedback_delivery.__post_init__()
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError('Receipt task_id must be nonempty')
        if not isinstance(self.objective, str):
            raise TypeError('Receipt objective must be a string')
        for name in ('decision_id', 'lab_id', 'run_id', 'agent_id'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'Receipt {name} must be nonempty or None')
        if self.source not in ('operation', 'lab') or type(self.read_only) is not bool:
            raise ValueError('Invalid receipt source or read_only flag')
        if self.issuing_model is not None and not isinstance(self.issuing_model, ModelRef):
            raise TypeError('issuing_model must be ModelRef or None')
        if self.job is not None and not isinstance(self.job, JobRef):
            raise TypeError('Receipt job must be JobRef or None')
        if not isinstance(self.request, dict) or not isinstance(self.result, dict):
            raise TypeError('Receipt request/result must be mappings')
        if (self.source == 'operation' and 'feedback_delivery' in self.result
                and self.result['feedback_delivery'] !=
                (asdict(self.feedback_delivery) if self.feedback_delivery is not None else None)):
            raise ValueError('Operation result and receipt feedback delivery identities differ')
        # Revalidate typed results after deserialization as well as construction.
        status = self.result.get('status')
        if status not in tuple(ExecutionStatus):
            raise ValueError('Receipt must contain an execution terminal status')
        if self.source == 'operation' and status == ExecutionStatus.PARTIAL:
            raise ValueError('An operation receipt cannot be partial')
        if type(self.result.get('simulated')) is not bool or not isinstance(self.result.get('summary'), str):
            raise ValueError('Receipt result requires summary and simulated fields')
        if self.source != 'lab' and status == ExecutionStatus.FAILED and not self.result.get('error'):
            raise ValueError('A failed receipt requires an error')
        if status == ExecutionStatus.COMPLETED and self.result.get('error') is not None:
            raise ValueError('A completed receipt cannot carry an error')
        if self.result.get('adopted_candidate_ids') and status != ExecutionStatus.COMPLETED:
            raise ValueError('Only completed receipts may confirm adoption')
        metrics = self.result.get('data', {})
        for name in ('quality_metrics', 'training_metrics', 'execution_metrics'):
            validated_metrics(metrics.get(name, {}), name)
        if self.measured_evaluation and (self.source != 'operation' or not has_scored_metrics(status, metrics)
                or not self.request.get('call_id') or self.result.get('call_id') != self.request['call_id']
                or not self.request.get('name') or canonical_refs(metrics.get('input_refs')) is None
                or any(not isinstance(metrics.get(k), str) or not metrics[k].strip()
                       for k in ('candidate_id', 'protocol_ref'))):
            raise ValueError('Measured evaluations require a matching scored operation, candidate, protocol and inputs')
        json.dumps({'request': self.request, 'result': self.result}, allow_nan=False)
        object.__setattr__(self, 'request', deepcopy(self.request))
        object.__setattr__(self, 'result', deepcopy(self.result))

    @classmethod
    def from_output(cls, task_id, output=None, *, status, references=(), request=None,
                    objective='', summary='', error=None, simulated=False, **identity):
        """Host envelope for a Lab's native JSON/text or artifact references.

        No business schema, summary, metrics or candidate types are required of
        the producer. Lifecycle/identity come from the executor; native fields
        stay opaque under output, including names that resemble our own schema.
        Non-JSON/binary artifacts can remain behind producer-native references.
        """
        return cls(task_id, objective, 'lab', {'input': deepcopy(request)},
                   {'status': status, 'summary': summary, 'error': error,
                    'simulated': simulated, 'output': deepcopy(output),
                    'references': deepcopy(references)}, **identity)


    @classmethod
    def from_operation(cls, task_id, objective, call, result, **identity):
        # Importing coordination here avoids making the core coordination
        # protocols depend on the optional observation envelope.
        from .coordination import OperationResult
        if not isinstance(result, OperationResult):
            raise TypeError('Expected OperationResult')
        result.__post_init__()
        return cls(task_id, objective, 'operation', asdict(call) if call is not None else {},
                   result.payload(), feedback_delivery=result.feedback_delivery, **identity)

    def payload(self):
        """Canonical Lab/operation observation journal entry."""
        value = {'operation': deepcopy(self.request), 'result': deepcopy(self.result),
                 'task': {'task_id': self.task_id, 'objective': self.objective,
                          'decision_id': self.decision_id, 'read_only': self.read_only,
                          'issuing_model': asdict(self.issuing_model) if self.issuing_model else None,
                          'job': asdict(self.job) if self.job else None}}
        if self.source == 'lab':
            value['task']['source'] = 'lab'
        if self.measured_evaluation:
            value['task']['measured_evaluation'] = True
        if any(item is not None for item in (self.lab_id, self.run_id, self.agent_id)):
            value['identity'] = {name: getattr(self, name) for name in
                                 ('lab_id', 'run_id', 'agent_id', 'decision_id')}
            value['identity']['issuing_model'] = asdict(self.issuing_model) if self.issuing_model else None
        if self.feedback_delivery is not None:
            value['feedback_delivery'] = asdict(self.feedback_delivery)
        return value


def receipt_task(receipt):
    """Task metadata for Lab and operation receipts."""
    return {**receipt['task'], 'mode': receipt['task'].get('source', 'operation'), 'directive_id': None,
            'simulated': receipt['result']['simulated']}


def receipt_facts(receipt):
    """Measured aggregates only; arbitrary tool data stays in the raw result.

    Operation metrics use the documented data.<group> layout. They do not imply
    any candidate/protocol binding or turn a query into a new evaluation.
    """
    result, task = receipt['result'], receipt['task']
    data = result.get('data', {})
    return {'task_id': task['task_id'], 'decision_id': task.get('decision_id'),
            'source': task.get('source', 'operation'), 'status': result['status'], 'summary': result['summary'],
            'simulated': result['simulated'], 'error': result.get('error'),
            'quality_metrics': deepcopy(data.get('quality_metrics', {})),
            'training_metrics': deepcopy(data.get('training_metrics', {})),
            'execution_metrics': deepcopy(data.get('execution_metrics', {})),
            'artifact_refs': deepcopy(result.get('artifacts', ())),
            'evidence_refs': deepcopy(data.get('evidence_refs', ())), 'candidates': [],
            'adopted_candidate_ids': deepcopy(result.get('adopted_candidate_ids', ()))}
