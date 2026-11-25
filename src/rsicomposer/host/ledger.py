"""Canonical action metadata, optional research sidecars and execution receipts."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from uuid import uuid4

from rsicomposer.harnesses.decision._json import _json_details
from rsicomposer.harnesses.decision.attempts import AttemptContract, known_attempts, requested_attempt, validate_attempt_fields
from rsicomposer.harnesses.memory import HistoryEntry
from rsicomposer.loops.contracts.coordination import LoopStepResult, OperationResult
from rsicomposer.loops.contracts.decision import Decision, OperationCall
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.evidence.records import ExperienceRecord


CORE_METADATA_FIELDS = frozenset(Decision.__dataclass_fields__) | frozenset({
    'status', 'operations', 'operation_outcomes', 'adopted_candidate_ids', 'adopted_decision_ids',
    'unresolved_adopted_candidate_ids', 'confirmed_edges', 'error', 'validation_error',
    'proposal', 'objective', 'operation_read_only', 'decision_model', 'controller_version',
    'requested_metadata', 'metadata_projection', 'context_metadata', 'authorization_rejected',
    'operation_elapsed_seconds', 'operation', 'result', 'event', 'elapsed_seconds',
    'research_cancel_requested',
    'call_id', 'task_id', 'issuing_controller_version', 'issuing_model', 'history_ref', 'history_owner',
})


def metadata_snapshot(value):
    """Detach a JSON-shaped mapping without loading provider objects."""
    if not isinstance(value, dict):
        raise TypeError('Decision metadata snapshot must be a JSON object')
    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError('Decision metadata keys must be strings')
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)
    check_keys(value)
    json.dumps(value, allow_nan=False)
    return deepcopy(value)


def same_metadata(left, right):
    """Compare JSON values, including the distinction between booleans and numbers."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate_metadata_projection(projection, context):
    if set(projection) & CORE_METADATA_FIELDS or set(context) & CORE_METADATA_FIELDS:
        raise ValueError('Metadata admission cannot overwrite core decision fields')
    if any(key not in projection or not same_metadata(projection[key], value) for key, value in context.items()):
        raise ValueError('Metadata context must project admitted record values')


def validate_metadata_record(record):
    """Verify saved opaque projections without interpreting Harness semantics."""
    try:
        metadata_snapshot(record['requested_metadata'])
        projection = metadata_snapshot(record['metadata_projection'])
        context = metadata_snapshot(record['context_metadata'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid saved decision metadata snapshot') from error
    validate_metadata_projection(projection, context)
    if set(record) - CORE_METADATA_FIELDS - set(projection):
        raise ValueError('Saved ledger metadata must belong to its admitted projection')
    if any(key not in record or not same_metadata(record[key], value) for key, value in projection.items()):
        raise ValueError('Saved metadata projection differs from its ledger record')


@dataclass(frozen=True)
class DecisionResearchMetadata:
    """The immutable request attached to exactly one sampled LLM decision."""

    research_attempt_id: str | None = None
    research_attempt_title: str = ''
    feedback_proposal_ids: tuple[str, ...] = ()

    def __post_init__(self):
        validate_attempt_fields(self.research_attempt_id, self.research_attempt_title)
        refs = self.feedback_proposal_ids
        if (not isinstance(refs, tuple) or len(refs) > 16
                or any(not isinstance(ref, str) or not ref.strip() or len(ref) > 512 for ref in refs)
                or len(set(refs)) != len(refs)):
            raise ValueError('feedback_proposal_ids must contain at most 16 distinct proposal IDs')

    @classmethod
    def from_record(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('Invalid requested research metadata snapshot')
        return cls(**value)


def admit_decision_metadata(decision, requested, records, *, attempt_contract, proposal_resolver=None):
    """Resolve a fresh request once; callers persist this detached snapshot.

    Replaying an already registered action must compare its requested_metadata
    rather than call this function and allocate a second attempt identity.
    """
    if not isinstance(requested, DecisionResearchMetadata):
        raise TypeError('Research metadata must be DecisionResearchMetadata')
    if not isinstance(attempt_contract, AttemptContract):
        raise TypeError('attempt_contract must be an AttemptContract')
    requested.__post_init__()
    snapshot = {
        'requested_metadata': asdict(requested),
        'research_attempt_id': None,
        'research_attempt_title': '',
        'feedback_proposal_ids': requested.feedback_proposal_ids,
    }
    if requested.feedback_proposal_ids:
        if not callable(proposal_resolver):
            raise ValueError('No feedback proposal provider is installed')
        snapshot['feedback_proposals'] = deepcopy(proposal_resolver(requested.feedback_proposal_ids))
    for operation in decision.operations or (None,):
        attempt_contract.validate(decision.kind, operation.name if operation else None,
                                  requested.research_attempt_id, requested.research_attempt_title)
    title = requested_attempt(requested.research_attempt_id, requested.research_attempt_title, records)
    if title:
        snapshot.update(research_attempt_id=requested.research_attempt_id or str(uuid4()),
                        research_attempt_title=title)
    return snapshot


def validate_decision_metadata(records):
    """Check saved research snapshots without resolving or changing identities."""
    rows = tuple(records)
    origins = set()
    for row in rows:
        request = row.get('requested_metadata')
        if (isinstance(request, dict) and not set(request).intersection(DecisionResearchMetadata.__dataclass_fields__)
                and not any(name in row for name in DecisionResearchMetadata.__dataclass_fields__)):
            continue
        requested = DecisionResearchMetadata.from_record(row.get('requested_metadata'))
        if (('feedback_proposal_ids' in row or not row.get('validation_error'))
                and row.get('feedback_proposal_ids', ()) != requested.feedback_proposal_ids):
            raise ValueError('Feedback proposal references differ from their original request')
        attempt_id, title = row.get('research_attempt_id'), row.get('research_attempt_title', '')
        validate_attempt_fields(attempt_id, title)
        if (attempt_id is None) != (title == '') or (row.get('validation_error') and attempt_id is not None):
            raise ValueError('Resolved research attempt membership requires a validated ID/title pair')
        if (not row.get('validation_error') and attempt_id is None
                and (requested.research_attempt_id is not None or requested.research_attempt_title)):
            raise ValueError('Validated research request is missing its resolved membership')
        if attempt_id is not None:
            if ((requested.research_attempt_id is not None and requested.research_attempt_id != attempt_id)
                    or (requested.research_attempt_title and requested.research_attempt_title != title)):
                raise ValueError('Research attempt membership differs from its original request')
            if requested.research_attempt_id is None:
                if requested.research_attempt_title != title or attempt_id in origins:
                    raise ValueError('Research attempt requires one explicit title-only origin')
                origins.add(attempt_id)
    if set(known_attempts(rows)) != origins:
        raise ValueError('Research attempt must have a validated ledger origin')


@dataclass(frozen=True)
class _PendingOperationRecord:
    record_id: str
    call: OperationCall | None
    result: OperationResult
    elapsed_seconds: float | None
    decision_id: str | None = None



class OperationLedger:
    def __init__(self, execution):
        self.host = execution

    def _record_operation(self, call: OperationCall | None, result: OperationResult,
                          *, elapsed_seconds: float | None = None) -> None:
        if self.host.state.pending_operation_record is not None:
            raise RuntimeError('Record the pending operation before starting another')
        ident = self.host.state.pending_operation_decision_id
        suffix = 'authorization' if self.host.state.decision_records.get(ident, {}).get('authorization_rejected') else 'operation'
        plural = len(self.host.state.decision_records.get(ident, {}).get('proposal', {}).get('operations', ())) > 1
        record_id = (f'{ident}:{suffix}:' + call.call_id) if plural and call else f'{ident}:{suffix}'
        self.host.state.pending_operation_record = _PendingOperationRecord(
            record_id if ident else str(uuid4()), deepcopy(call), deepcopy(result), elapsed_seconds, ident,
        )
        self.host.state.pending_decision = None
        self._flush_operation_record()


    def _decision_graph(self):
        graph = []
        for row in self.host.state.decision_records.values():
            item = {key: row[key] for key in ('decision_id', 'status')}
            item.update(row['context_metadata'])
            for key in ('operations', 'depends_on_decision_ids', 'source_decision_ids', 'adopted_candidate_ids',
                        'adopted_decision_ids', 'unresolved_adopted_candidate_ids'):
                if row.get(key):
                    item[key] = row[key]
            graph.append(item)
        return deepcopy(tuple(graph))


    def _begin_decision(self, decision, context, *, record_history=True, requested=None):
        if not isinstance(decision, Decision):
            raise TypeError('Policy must return a Decision')
        decision.__post_init__()
        if requested is not None:
            requested = metadata_snapshot(requested)
        ident = decision.decision_id
        row = self.host.state.decision_records.get(ident)
        if row is not None:
            if row['proposal'] != asdict(decision):
                raise ValueError('Decision ID has different content')
            saved_request = metadata_snapshot(row['requested_metadata'])
            if requested is not None and not same_metadata(requested, saved_request):
                raise ValueError('Decision ID has different metadata')
            if row['status'] == 'executing':
                raise RuntimeError('Decision execution outcome is unknown; refusing to execute it again')
            if (row['status'] in ('blocked', 'failed') and not row.get('authorization_rejected') and ident not in self.host.state.decision_results
                    and self.host.state.pending_decision is None):
                raise ValueError(row['error'])
        else:
            requested = {} if requested is None else requested
            row = dict(decision_id=ident, kind=decision.kind, status='pending', reason=decision.reason,

                operations=tuple(call.name for call in decision.operations), operation_outcomes={},
                depends_on_decision_ids=decision.depends_on_decision_ids,
                source_decision_ids=decision.source_decision_ids,
                requested_metadata=requested, metadata_projection={}, context_metadata={},
                adopted_candidate_ids=(), adopted_decision_ids=(), unresolved_adopted_candidate_ids=(),
                confirmed_edges=(), error=None, validation_error=None,
                proposal=deepcopy(asdict(decision)),
                objective=context.objective,
                operation_read_only={call.call_id: next((spec.read_only for spec in context.operations
                    if spec.name == call.name), False) for call in decision.operations},
                decision_model=asdict(context.decision_model) if context.decision_model else None)
            row['controller_version'] = self.host.controller_version
            for name in ('depends_on_decision_ids', 'source_decision_ids'):
                for ref in getattr(decision, name):
                    known = self.host.state.decision_records.get(ref)
                    if known is None or (name == 'depends_on_decision_ids' and known['status'] != 'completed'):
                        message = f'Unknown, forward or incomplete decision dependency/reference: {ref}'
                        row.update(status='failed', error=message, validation_error=message)
                        break
                if row['validation_error']:
                    break
            if not row['validation_error']:
                try:
                    admit = optional_method(self.host.harness, 'admit_decision_metadata')
                    if admit is not None:
                        result = admit(deepcopy(decision), deepcopy(requested), deepcopy(tuple(self.host.state.decision_records.values())))
                        if not isinstance(result, dict) or set(result) != {'record', 'context'}:
                            raise TypeError('Metadata admission must return record and context projections')
                        projection, context_projection = (metadata_snapshot(result[key]) for key in ('record', 'context'))
                        if 'requested_metadata' in projection:
                            if not same_metadata(projection.pop('requested_metadata'), requested):
                                raise ValueError('Metadata admission changed the requested snapshot')
                        validate_metadata_projection(projection, context_projection)
                        row.update(projection, metadata_projection=projection, context_metadata=context_projection)
                except ValueError as error:
                    row.update(status='failed', error=str(error), validation_error=str(error))
            if not row['validation_error']:
                row['confirmed_edges'] = tuple({'from': ref, 'to': ident, 'kind': kind, 'confirmed': True}
                    for kind, refs in (('depends_on', decision.depends_on_decision_ids), ('source', decision.source_decision_ids))
                    for ref in refs)
            self.host.state.decision_records[ident] = row
        self.host.state.pending_decision = deepcopy(decision)
        if record_history or row['validation_error']:
            self.host.memory.append(HistoryEntry(ident, 'decision', None, decision.kind, decision.simulated,
                details=_json_details({**deepcopy(row['proposal']), 'decision_model': row['decision_model'],
                         'requested_metadata': deepcopy(row['requested_metadata']),
                         **deepcopy(row['context_metadata']),
                         'status': 'failed' if row['validation_error'] else 'pending',
                         'validation_error': row['validation_error']})))
        if row['validation_error']:
            self.host.state.pending_decision = None
            raise ValueError(row['error'])
        return row


    def _adoption(self, ident, receipt):
        receipt.__post_init__()
        adopted = receipt.adopted_candidate_ids
        row = self.host.state.decision_records.get(ident, {})
        edges = tuple(edge for edge in row.get('confirmed_edges', ()) if edge['kind'] != 'adopted')
        return dict(decision_id=ident, **deepcopy(row.get('context_metadata', {})), adopted_candidate_ids=adopted,
                    adopted_decision_ids=(), unresolved_adopted_candidate_ids=adopted, confirmed_edges=edges)


    def _write_operation_record(self, pending):
        metadata = self._adoption(pending.decision_id, pending.result)
        result, call = pending.result, pending.call
        details = {'operation': asdict(call) if call else None, 'result': result.payload(), **metadata}
        self.host.memory.append(HistoryEntry(pending.record_id, 'operation', None, result.summary,
            result.simulated, source_refs=tuple(ref.artifact_id for ref in result.artifacts), details=_json_details(details)))
        from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
        from rsicomposer.runtime.storage.codec import encode
        row = self.host.state.decision_records.get(pending.decision_id, {})
        identity = {'call_id': call.call_id if call else result.call_id,
                    'controller_version': self.host.controller_version,
                    'issuing_controller_version': row.get('controller_version'),
                    'issuing_model': row.get('decision_model')}
        event = EvidenceEvent(pending.record_id, 'lab' if call is not None and '.' in call.name
            and not call.name.startswith('host.') else 'operation', decision_id=pending.decision_id,
            task_id=call.call_id if call else pending.record_id, status=result.status.value,
            request=asdict(call) if call else None, output=result.payload(),
            references=tuple(asdict(ref) for ref in result.artifacts), identity=identity)
        self.host.experience.record(ExperienceRecord(pending.record_id, 'runtime_feedback', 'decision_maker',
            None, f'Preparation operation {result.status.value}', result.simulated,
            source_refs=(f'history:decision_maker:{pending.record_id}',),
            details=_json_details({'operation': call.name if call else None, 'status': result.status.value,
                     'error': result.error, 'elapsed_seconds': pending.elapsed_seconds,
                     'history_ref': pending.record_id, 'history_owner': 'decision_maker',
                     'event': encode(asdict(event)), **metadata})))
        self.host.harness.observe(deepcopy(event))
        return metadata


    def _complete_operation(self, pending, metadata):
        if pending.decision_id is not None:
            row = self.host.state.decision_records[pending.decision_id]
            if pending.call is not None:
                row['operation_outcomes'][pending.call.call_id] = pending.result.status.value
            calls = row['proposal'].get('operations', ())
            complete = all(call['call_id'] in row['operation_outcomes'] for call in calls)
            status = ('failed' if 'failed' in row['operation_outcomes'].values() else
                      'cancelled' if 'cancelled' in row['operation_outcomes'].values() else 'completed')
            row.update(metadata, status=status if complete else 'executing',
                       error=pending.result.error, operation_elapsed_seconds=pending.elapsed_seconds)
            if complete and not row.get('authorization_rejected'):
                self.host.state.decision_results[pending.decision_id] = LoopStepResult('progressed',
                    operation_result=deepcopy(pending.result), decision_id=pending.decision_id)
            listeners = tuple(callback for owner, callback in self.host._receipt_listeners
                              if owner is self.host.harness)
            if listeners:
                from rsicomposer.loops.contracts.receipts import ExecutionReceipt
                row = self.host.state.decision_records[pending.decision_id]
                objective = row.get('objective')
                if objective is not None:
                    receipt = ExecutionReceipt.from_operation(pending.record_id, objective,
                        pending.call, pending.result, decision_id=pending.decision_id,
                        read_only=row.get('operation_read_only', {}).get(pending.call.call_id, False) if pending.call else False,
                        issuing_model=ModelRef(**row['decision_model']) if row.get('decision_model') else None)
                    # If observation persistence fails, the existing pending
                    # record retries the same receipt, without executing again.
                    for listener in listeners:
                        listener(receipt)


    def _flush_operation_record(self) -> OperationResult:
        pending = self.host.state.pending_operation_record
        if pending is None:
            raise RuntimeError('No operation recording is pending')
        result = pending.result
        metadata = self._write_operation_record(pending)
        self._complete_operation(pending, metadata)
        self.host.state.operation_results.append(deepcopy(result))
        self.host.state.pending_operation_decision_id = None
        self.host.state.pending_operation_record = None
        return deepcopy(result)
