"""Research-oriented working context, derived without another model call.

Interpretations never authorize dropping evidence. Old receipts may use exact
factual cards with authorized links to originals; oversized live payloads use
bounded previews without changing the execution ledger.
"""
from copy import deepcopy
from dataclasses import asdict, replace

from rsicomposer.harnesses.decision.context import encoded, result_id


def archive_available(context, collection):
    return any(spec.name == 'research_archive' and spec.read_only
        and collection in spec.input_schema.get('properties', {}).get('collection', {}).get('enum', ())
        for spec in context.operations)


def receipt_needs_protection(receipt):
    """Recognize structured facts, including nested evaluation/training receipts."""
    if isinstance(receipt, (tuple, list)):
        return any(receipt_needs_protection(value) for value in receipt)
    if not isinstance(receipt, dict):
        return False
    for key, value in receipt.items():
        # A view descriptor is navigation, not a measurement. Inspect its facts
        # using the original field names (cards store JSON Pointer keys).
        if key == 'evidence_card':
            if not (isinstance(value, dict) and value.get('schema_version') == 1
                    and isinstance(value.get('read_path'), dict)
                    and isinstance(receipt.get('facts'), dict)):
                return True  # Unknown producer cards remain protected.
            continue
        field = key.rsplit('/', 1)[-1].replace('~1', '/').replace('~0', '~')
        if ((field in ('error', 'validation_error', 'metrics') or field.endswith('_metrics')) and value
                or field == 'status' and value in ('failed', 'cancelled', 'rejected')
                or receipt_needs_protection(value)):
            return True
    return False


def evidence_needs_protection(entry):
    """Conservative mechanical rule; never ask prose to classify contradictions."""
    return receipt_needs_protection(entry.details)


def archived_operation_result_ids(context):
    """Recognize recorded receipts after their history rows become cards too.

    A history-card digest identifies the entire history row, not its nested
    result. Keep that nested identity explicitly so a projected history cannot
    make an already archived result appear unrecorded and permanently resident.
    Unknown producer cards without this identity remain protected.
    """
    recorded = set()
    for row in context.history:
        if row.kind != 'operation':
            continue
        result = row.details.get('result')
        if isinstance(result, dict):
            recorded.add(result_id(result))
        card = row.details.get('evidence_card', {})
        if (isinstance(card, dict) and card.get('schema_version') == 1
                and isinstance(card.get('read_path'), dict)
                and card['read_path'].get('collection') == 'history'
                and card['read_path'].get('record_id') == row.record_id):
            digest = card.get('operation_result_sha256')
            if isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest):
                recorded.add(digest)
    return recorded


def resident_evidence_indices(context, collection, *, recent=2):
    """Keep recent measurements and failures, not every past receipt forever.

    This selects working-context residency only. Raw stores are unchanged. Without
    a read-only archive for this collection all factual receipts stay resident.
    Latest/live dependencies are protected separately, regardless of this window.
    """
    rows = getattr(context, collection)
    values = [row.payload() if collection == 'operation_results' else row.details for row in rows]
    measured = [i for i, value in enumerate(values) if receipt_needs_protection(value)]
    if not archive_available(context, collection):
        return set(measured)
    def failed(value):
        if isinstance(value, (list, tuple)):
            return any(failed(v) for v in value)
        if not isinstance(value, dict):
            return False
        for key, val in value.items():
            field = key.rsplit('/', 1)[-1]
            if (field in ('error', 'validation_error') and val
                    or field == 'status' and val in ('failed', 'cancelled', 'rejected')
                    or failed(val)):
                return True
        return False
    keep = set(measured[-recent:]) | set([i for i in measured if failed(values[i])][-recent:])
    if collection == 'operation_results':
        # The tool archive reconstructs receipts from history. Merely advertising
        # an archive does not make an unrecorded result recoverable.
        recorded = archived_operation_result_ids(context)
        for i in measured:
            row = rows[i]
            card = row.data.get('evidence_card', {})
            linked = (isinstance(card, dict) and card.get('schema_version') == 1
                      and isinstance(card.get('read_path'), dict)
                      and card['read_path'].get('collection') == collection)
            if result_id(row.payload()) not in recorded and not linked:
                keep.add(i)
    return keep


def live_evidence_ids(context):
    """Follow all active dependency/source/adoption edges before any projection."""
    graph = {r['decision_id']: r for r in context.decision_graph}
    decisions = {ident for ident, row in graph.items()
                 if row['status'] not in ('completed', 'failed', 'cancelled')}
    directives = set()
    pending = list(decisions)
    while pending:
        row = graph.get(pending.pop(), {})
        for key in ('depends_on_decision_ids', 'source_decision_ids', 'adopted_decision_ids'):
            for ident in row.get(key, ()):
                if ident not in decisions:
                    decisions.add(ident)
                    pending.append(ident)
    return decisions, directives


def full_history_ids(context):
    decisions, directives = live_evidence_ids(context)
    latest_result = result_id(context.operation_results[-1].payload()) if context.operation_results else None
    # Identical reads can have the same receipt digest. Only the newest matching
    # occurrence is the latest observation; older occurrences are not live work.
    latest_result_history = next((h.record_id for h in reversed(context.history)
        if latest_result is not None and isinstance(h.details.get('result'), dict)
        and result_id(h.details['result']) == latest_result), None)
    return {h.record_id for h in context.history[-2:]} | {
        h.record_id for h in context.history
        if h.record_id in decisions or h.details.get('decision_id') in decisions
        or h.task_id in directives
        or h.record_id == latest_result_history}


def factual_fields(value, path=''):
    """Copy structured fields at exact JSON pointers; never interpret prose.

    Unknown containers are traversed, not copied wholesale. Metric maps, IDs,
    reference fields and model/artifact identities retain their complete values.
    Lists retain their original indices in the pointers. Prose and duplicate
    directives are recoverable through the card's digest-qualified archive link.
    """
    facts = {}
    if isinstance(value, dict):
        if isinstance(value.get('sha256'), str) and 'files' in value:
            # A manifest's inventory stays in the raw archive; its aggregate
            # identity remains comparable after compaction.
            return {path + '/sha256': value['sha256']}
        if ({'backend_id', 'model_id', 'revision'} <= value.keys()
                or {'artifact_id', 'kind', 'uri'} <= value.keys()):
            return {path: deepcopy(value)}
        for key, item in value.items():
            pointer = path + '/' + key.replace('~', '~0').replace('/', '~1')
            if (key in ('metrics', 'status', 'simulated', 'error', 'validation_error',
                        'mode', 'base_version', 'parent_system', 'digest', 'sha256', 'error_type', 'exception_type',
                        'exit_code', 'failure_origin', 'verified', 'adopted', 'execution_budget', 'output_diagnostics')
                    or key.endswith(('_id', '_ids', '_ref', '_refs', '_metrics', '_sha256'))):
                facts[pointer] = deepcopy(item)
            else:
                facts.update(factual_fields(item, pointer))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            facts.update(factual_fields(item, path + '/' + str(index)))
    return facts



MAX_EVIDENCE_BYTES = 8192


def previous_feedback_view(context, *, include_memory_route=False, include_plugin_observations=False):
    """Deliver the previous native operation before archive/card projection.

    One replacing window, never a memory log. Large originals use a mechanical
    UTF-8 page with an exact archive address, not a model-written interpretation.
    """
    if not context.operation_results:
        return None
    card = context.operation_results[-1].data.get('evidence_card')
    projected = isinstance(card, dict) and card.get('schema_version') == 1
    original = context.operation_results[-1].payload()
    ident = card.get('sha256') if projected else result_id(original)
    source = next((row for row in reversed(context.history)
        if row.kind == 'operation' and isinstance(row.details.get('result'), dict)
        and result_id(row.details['result']) == ident), None)
    if projected:
        # A refreshed context can contain an already projected result alongside
        # its original history. Never relabel the evidence card as native data.
        if source is not None:
            original = source.details['result']
        else:
            previous = context.research_agenda.get('previous_feedback')
            return deepcopy(previous) if previous and previous.get('source_sha256') == ident else None
    operation = deepcopy(source.details.get('operation')) if source else None
    from rsicomposer.harnesses.decision.observations import (
        native_observation,
        task_observation,
    )
    if source is None or not native_observation(source):
        return None
    if (not include_plugin_observations and not operation.get('name', '').startswith('lab_')
            and not task_observation(source)):
        # Legacy harnesses can have their own smaller wire budgets and need no
        # second copy of query responses. AgentMemory explicitly opts into the
        # generic observation adapter; the original operation result is always kept.
        return None
    if (not include_memory_route and operation
            and (operation.get('arguments') or {}).get('feedback_route') == 'memory'):
        return None
    value = {'source_sha256': ident, 'operation': operation,
        'delivery': 'previous_operation_only; raw_observation_not_memory', 'complete': True}
    raw = encoded(original).encode('utf-8')
    can_read = (source is not None and archive_available(context, 'operation_results')
                and ident in archived_operation_result_ids(context))
    if len(raw) <= 16384 or not can_read:
        value['result'] = deepcopy(original)
    else:
        content = raw[:16384].decode('utf-8', errors='ignore')
        value.update(complete=False, raw_json_page={'content': content, 'offset': 0,
            'end_offset': len(content.encode('utf-8')), 'total_bytes': len(raw), 'eof': False},
            read_path={'operation': 'research_archive', 'collection': 'operation_results',
                       'record_id': ident, 'evidence_path': '/result'})
    return value


def _preview(text, limit=512):
    if not isinstance(text, str) or len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + '\n[omitted; read original evidence]\n' + text[-half:]


def _bounded_facts(value):
    """Keep aggregate metrics/identities exact; explicitly omit large row tables.

    Per-item metrics in an omitted table are not aggregate measurements. Nothing
    computes, guesses or replaces them with an average. Their exact source and
    count are disclosed for a subsequent archive read.
    """
    omitted = []
    def essential(key):
        return (key in ('metrics', 'status', 'simulated', 'mode', 'base_version', 'parent_system',
                        'digest', 'error', 'error_type', 'exception_type', 'exit_code', 'model', 'provenance')
                or key.endswith(('_id', '_ids', '_ref', '_refs', '_metrics')))
    def visit(item, path='', key=''):
        if key in ('error', 'validation_error') and isinstance(item, str):
            preview = _preview(item)
            if preview != item:
                omitted.append({'path': path, 'kind': 'text_excerpt', 'characters': len(item),
                                'sha256': result_id(item)})
            return preview
        if key in ('stdout', 'stderr', 'text', 'content', 'log') and isinstance(item, str) and len(item) > 512:
            omitted.append({'path': path, 'kind': 'text_excerpt', 'characters': len(item),
                            'sha256': result_id(item), 'preview': _preview(item, 256)})
            return ''
        exact = (key in ('metrics', 'status', 'simulated', 'mode', 'base_version', 'parent_system', 'digest', 'error_type', 'exception_type', 'exit_code')
                 or key.endswith(('_id', '_ids', '_ref', '_refs', '_metrics')))
        if exact:
            return deepcopy(item)
        if isinstance(item, (list, tuple)) and len(item) > 16:
            omitted.append({'path': path, 'kind': 'collection_omitted', 'count': len(item),
                            'sha256': result_id(item)})
            return []
        if isinstance(item, dict):
            if len(item) > 64:
                kept = {k: v for k, v in item.items() if essential(k)}
                omitted.append({'path': path, 'kind': 'fields_omitted', 'count': len(item) - len(kept),
                                'retained_fields': list(kept), 'sha256': result_id(item)})
                return {k: visit(v, path + '/' + k.replace('~', '~0').replace('/', '~1'), k)
                        for k, v in kept.items()}
            return {k: visit(v, path + '/' + k.replace('~', '~0').replace('/', '~1'), k)
                    for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(v, path + '/' + str(i)) for i, v in enumerate(item)]
        return deepcopy(item)
    return factual_fields(visit(value)), omitted


def bounded_evidence_view(context):
    """Bound oversized individual observations before either native or local compact.

    Latest/live observations remain present as factual records; only their bulk
    payload is replaced. Without the corresponding read capability no projection
    is permitted. This is an input view, never a mutation of state or archives.
    """
    return evidence_working_view(context, latest=True)

def evidence_working_view(context, *, latest=False):
    """Bound old payloads, preserving every structured fact and exact originals.

    Capability checks are per collection. Cards retain exact facts while visible;
    the token compactor may archive old cards outside its recent/live window.
    Required live facts that still cannot fit cause an atomic budget failure.
    """
    full = full_history_ids(context)
    decisions, directives = live_evidence_ids(context)
    live_results = {result_id(h.details['result']) for h in context.history
                    if h.record_id in full and isinstance(h.details.get('result'), dict)}
    archived_results = archived_operation_result_ids(context)
    projected = {}
    for collection in ('history', 'experience', 'operation_results'):
        rows = getattr(context, collection)
        if not archive_available(context, collection):
            continue
        output = []
        for index, row in enumerate(rows):
            original = row.payload() if collection == 'operation_results' else asdict(row)
            if collection == 'operation_results':
                ident = result_id(original)
                existing = row.data.get('evidence_card')
                known_view = (isinstance(existing, dict) and existing.get('schema_version') == 1
                              and isinstance(existing.get('read_path'), dict))
                if latest and ident not in archived_results and not known_view:
                    output.append(row)
                    continue
                required = index == len(rows) - 1 or ident in live_results
                existing = row.data.get('evidence_card')
            else:
                ident = row.record_id
                required = (ident in full or ident in decisions
                    or row.details.get('decision_id') in decisions
                    or getattr(row, 'directive_id', None) in directives)
                existing = row.details.get('evidence_card')
            already_card = (isinstance(existing, dict) and existing.get('schema_version') == 1
                            and isinstance(existing.get('read_path'), dict))
            if already_card and not latest:
                field = 'data' if collection == 'operation_results' else 'details'
                values = deepcopy(getattr(row, field))
                descriptor = values['evidence_card']
                descriptor.pop('summary_preview', None)
                for omission in descriptor.get('omissions', ()):
                    if isinstance(omission, dict):
                        omission.pop('preview', None)
                output.append(replace(row, **{field: values}))
                continue
            oversized = len(encoded(original).encode('utf-8')) > MAX_EVIDENCE_BYTES
            if already_card or (latest and not oversized) or (not latest and (required or not receipt_needs_protection(original))):
                output.append(row)
                continue
            facts, omissions = _bounded_facts(original) if latest else (factual_fields(original), [])
            card = {'evidence_card': {'schema_version': 1, 'authority': 'exact_structured_fields_with_excerpts' if omissions else 'exact_structured_fields',
                        'sha256': result_id(original),
                        'read_path': {'operation': 'research_archive', 'collection': collection,
                                      'record_id': ident}},
                    'facts': facts}
            if collection == 'history' and row.kind == 'operation' and isinstance(row.details.get('result'), dict):
                card['evidence_card']['operation_result_sha256'] = result_id(row.details['result'])
                operation = row.details.get('operation') or {}
                route = (operation.get('arguments') or {}).get('feedback_route')
                if route in ('context', 'memory'):
                    card['evidence_card']['feedback_route'] = route
            if latest:
                card['evidence_card'].update(omitted_content_read=False, bulk_payload_omitted=True,
                    omissions=omissions)
                if required:
                    card['evidence_card']['summary_preview'] = _preview(row.summary, 128)
            if collection == 'operation_results':
                card['evidence_card']['read_path']['evidence_path'] = '/result'
            updates = {'summary': 'Archived evidence; read original for prose and qualifications.'}
            updates['data' if collection == 'operation_results' else 'details'] = card
            if latest and collection == 'operation_results':
                updates['error'] = _preview(row.error)
            candidate = replace(row, **updates)
            # Small receipts stay verbatim; cards must actually reduce the view.
            output.append(candidate if len(encoded(asdict(candidate))) < len(encoded(original)) else row)
        projected[collection] = tuple(output)
    return replace(context, **projected)


def research_working_view(context, jobs, *, recent=8):
    """Adapt neutral research knowledge to this harness's archive and receipts.

    Attempt organization and feedback target selection belong to feedback. This
    adapter only protects execution dependencies and projects duplicate records.
    """
    graph = {r['decision_id']: r for r in context.decision_graph}
    live_dependencies, outstanding = live_evidence_ids(context)
    resident_history = resident_evidence_indices(context, 'history')
    state = {}
    state.update(
        live_state_fields=['pending', 'progresses', 'run_control'],
        protected_operation_result_ids=[result_id(context.operation_results[i].payload())
            for i in sorted(resident_evidence_indices(context, 'operation_results'))],
        protected_history_ids=[h.record_id for i, h in enumerate(context.history)
            if i in resident_history
            or h.record_id in live_dependencies or h.details.get('decision_id') in live_dependencies
            or h.task_id in outstanding])
    # This digest changes only when semantic learning state changes, not on waits.
    state['revision'] = result_id(state)
    recent_ids = {h.record_id for h in context.history[-recent:]}
    can_archive = archive_available(context, 'history') and archive_available(context, 'decision_graph')
    removed_history = [h for h in context.history if can_archive and jobs
        and h.kind == 'decision' and h.record_id not in recent_ids
        and h.record_id not in state['protected_history_ids']
        and graph.get(h.record_id, {}).get('status') == 'completed'
        and not h.details.get('error') and not h.details.get('validation_error')]
    history = tuple(h for h in context.history if h not in removed_history)
    projection = {
        'history': {'omitted_count': len(removed_history), 'sha256': result_id([asdict(h) for h in removed_history]),
                    'read_path': {'operation': 'research_archive', 'collection': 'history'}},
        'evidence_policy': 'Old evidence may use exact factual cards linked to authorized archives. '
                           'Latest/live receipts and a bounded window of measured/failed facts stay resident; '
                           'older receipts remain in the raw archive and may fold into a source-qualified summary. '
                           'Absence from the working view never means untested, successful or irrelevant.'}
    return replace(context, research_state=state, history=history,
                   research_agenda={**context.research_agenda, 'working_projection': projection})
