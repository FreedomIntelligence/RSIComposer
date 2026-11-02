"""Research-oriented working context, derived without another model call.

Interpretations never authorize dropping evidence. Old receipts may use exact
factual cards with authorized links to originals; oversized live payloads use
bounded previews without changing the execution ledger.
"""
from copy import deepcopy
from dataclasses import asdict, replace

from .context import encoded, result_id


def archive_available(context, collection):
    return any(spec.name == 'research_archive' and spec.read_only
        and collection in spec.input_schema.get('properties', {}).get('collection', {}).get('enum', ())
        for spec in context.operations)


def ordered_jobs(context, jobs):
    # New journals persist capture order. Legacy journals use the raw history,
    # never lexical UUID/hash ordering as a substitute for recency.
    positions = {h.task_id: i for i, h in enumerate(context.history) if h.task_id}
    return sorted(jobs, key=lambda j: (j.get('capture_index', positions.get(
        j['input']['receipt']['directive']['directive_id'], -1)), j['job_id']))


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
                return True  # Unknown/legacy cards remain conservative.
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


def live_evidence_ids(context):
    """Follow all active dependency/source/adoption edges before any projection."""
    graph = {r['decision_id']: r for r in context.decision_graph}
    decisions = {ident for ident, row in graph.items()
                 if row['status'] not in ('completed', 'failed', 'cancelled')}
    directives = {d.directive_id for d in context.outstanding}
    for execution in (context.execution, *context.executions.values()):
        if execution.progress is not None:
            directives.add(execution.progress.directive_id)
    for h in context.history:
        if h.task_id in directives and h.details.get('decision_id'):
            decisions.add(h.details['decision_id'])
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
    latest_feedback = context.last_feedback.directive_id if context.last_feedback else None
    latest_result = result_id(asdict(context.operation_results[-1])) if context.operation_results else None
    # Identical reads can have the same receipt digest. Only the newest matching
    # occurrence is the latest observation; older occurrences are not live work.
    latest_result_history = next((h.record_id for h in reversed(context.history)
        if latest_result is not None and isinstance(h.details.get('result'), dict)
        and result_id(h.details['result']) == latest_result), None)
    return {h.record_id for h in context.history[-2:]} | {
        h.record_id for h in context.history
        if h.record_id in decisions or h.details.get('decision_id') in decisions
        or h.task_id in directives
        or latest_feedback is not None and h.kind == 'feedback' and h.task_id == latest_feedback
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
        if ({'backend_id', 'model_id', 'revision'} <= value.keys()
                or {'artifact_id', 'kind', 'uri'} <= value.keys()):
            return {path: deepcopy(value)}
        for key, item in value.items():
            pointer = path + '/' + key.replace('~', '~0').replace('/', '~1')
            if (key in ('metrics', 'status', 'simulated', 'error', 'validation_error',
                        'mode', 'base_version', 'parent_system', 'digest', 'error_type', 'exception_type', 'exit_code')
                    or key.endswith(('_id', '_ids', '_ref', '_refs', '_metrics'))):
                facts[pointer] = deepcopy(item)
            else:
                facts.update(factual_fields(item, pointer))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            facts.update(factual_fields(item, path + '/' + str(index)))
    return facts



MAX_EVIDENCE_BYTES = 8192


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

    Capability checks are per collection. Measured/failed facts remain protected;
    ordinary query cards may later fold into a source-qualified summary. If facts plus required live state
    are still too large, the token compactor must fail without committing.
    """
    full = full_history_ids(context)
    decisions, directives = live_evidence_ids(context)
    latest_feedback = context.last_feedback.directive_id if context.last_feedback else None
    live_results = {result_id(h.details['result']) for h in context.history
                    if h.record_id in full and isinstance(h.details.get('result'), dict)}
    archived_results = {result_id(h.details['result']) for h in context.history
                        if h.kind == 'operation' and isinstance(h.details.get('result'), dict)}
    projected = {}
    feedback = context.last_feedback
    if (latest and feedback is not None and archive_available(context, 'history')
            and len(encoded(asdict(feedback)).encode('utf-8')) > MAX_EVIDENCE_BYTES):
        original_feedback = asdict(feedback)
        source = next((h for h in reversed(context.history)
                       if h.kind == 'feedback' and isinstance(h.details.get('feedback'), dict)
                       and result_id(h.details['feedback']) == result_id(original_feedback)), None)
        # Require an exact recorded source for last_feedback, not merely a guessed ID.
        if source is not None:
            descriptor = {'authority': 'excerpt_not_summary', 'sha256': result_id(original_feedback),
                'read_path': {'operation': 'research_archive', 'collection': 'history',
                              'record_id': source.record_id, 'evidence_path': '/details/feedback'},
                'omitted_content_read': False}
            summary = encoded({'evidence_view': descriptor, 'summary_preview': _preview(feedback.summary)})
            projected['last_feedback'] = replace(feedback, summary=summary, error=_preview(feedback.error))
    for collection in ('history', 'experience', 'operation_results'):
        rows = getattr(context, collection)
        if not archive_available(context, collection):
            continue
        output = []
        for index, row in enumerate(rows):
            original = asdict(row)
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
                    or getattr(row, 'directive_id', None) in directives
                    or latest_feedback is not None
                    and getattr(row, 'directive_id', None) == latest_feedback)
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
            if latest:
                card['evidence_card'].update(omitted_content_read=False, bulk_payload_omitted=True,
                    omissions=omissions)
                if required:
                    card['evidence_card']['summary_preview'] = _preview(row.summary, 128)
            updates = {'summary': 'Archived evidence; read original for prose and qualifications.'}
            updates['data' if collection == 'operation_results' else 'details'] = card
            if latest and collection == 'operation_results':
                updates['error'] = _preview(row.error)
            candidate = replace(row, **updates)
            # Small receipts stay verbatim; cards must actually reduce the view.
            output.append(candidate if len(encoded(asdict(candidate))) < len(encoded(original)) else row)
        projected[collection] = tuple(output)
    return replace(context, **projected)


def research_working_view(context, jobs, attempts, *, recent=8):
    jobs = ordered_jobs(context, jobs)
    rank = {j['job_id']: i for i, j in enumerate(jobs)}
    active = {r['decision_id'] for r in context.decision_graph
              if r['status'] not in ('completed', 'failed', 'cancelled')}
    graph = {r['decision_id']: r for r in context.decision_graph}
    live_dependencies, outstanding = live_evidence_ids(context)
    ranked_attempts = sorted(attempts, key=lambda a: (
        bool(active.intersection(a['member_decision_ids'])), rank.get(a['provenance']['job_id'], -1), a['attempt_id']))
    selected = [a for a in ranked_attempts if active.intersection(a['member_decision_ids'])
                or a in ranked_attempts[-recent:]]
    # The full membership/revision history remains in feedback_learning_read.
    attempt_summaries = [{k: deepcopy(a[k]) for k in ('attempt_id', 'title', 'summary', 'revision',
                         'provenance', 'simulated')} | {
                             'member_count': len(a['member_decision_ids']),
                             'source_refs': deepcopy(a['source_refs']),
                             'authority': 'unverified_interpretation'} for a in selected]
    chosen_jobs = {a['provenance']['job_id'] for a in selected} | {j['job_id'] for j in jobs[-recent:]}
    findings = [{'job_id': j['job_id'], 'learned_by': deepcopy(j['model']),
                 'simulated': j['simulated'], 'authority': 'unverified_interpretation',
                 'research_context': deepcopy(j['input'].get('research_context', {})),
                 'source_refs': deepcopy(j['input']['source_refs']),
                 'proposals': [deepcopy(p) for p in j['proposals'] if p['kind'] == 'memory']}
                for j in jobs if j['job_id'] in chosen_jobs and j['status'] == 'completed'
                and any(p['kind'] == 'memory' for p in j['proposals'])]
    unsettled = [j for j in jobs if j['status'] != 'completed' or j.get('grouping_error')]
    pending = [j for j in unsettled if j['status'] in ('queued', 'running') or j in unsettled[-recent:]]
    state = {'schema_version': 1, 'authority': 'host_assembled_working_view',
        'interpretation_authority': 'unverified',
        'attempts': attempt_summaries, 'findings': findings,
        'learning_attention': [{'job_id': j['job_id'], 'status': j['status'], 'error': j.get('error'),
                              'grouping_error': j.get('grouping_error')} for j in pending],
        'totals': {'attempts': len(attempts), 'learning_jobs': len(jobs), 'learning_attention': len(unsettled)},
        'read_path': {'operation': 'feedback_learning_read', 'attempt_filter': 'research_attempt_id'},
        'live_state_fields': ['execution', 'executions', 'outstanding', 'candidate_evaluations',
                             'candidates', 'run_control', 'last_feedback'],
        'protected_operation_result_ids': [result_id(asdict(r)) for r in context.operation_results
            if receipt_needs_protection(asdict(r))],
        'protected_history_ids': [h.record_id for h in context.history if evidence_needs_protection(h)
            or h.record_id in live_dependencies or h.details.get('decision_id') in live_dependencies
            or h.task_id in outstanding],
        'note': 'Working view, not a new instruction or proof of success. Missing summaries are not negative '
                'results. Consult original evidence and applicability before reusing a finding.'}
    # This digest changes only when semantic learning state changes, not on waits.
    state['revision'] = result_id(state)
    can_read_learning = (archive_available(context, 'memories') and
                         any(s.name == 'feedback_learning_read' and s.read_only for s in context.operations))
    expected = {'feedback-learning:' + j['job_id']: j.get('memory_digest')
                for j in jobs if j.get('published') and j.get('memory_digest')}
    removed_memories = [m for m in context.memories if can_read_learning
                        and expected.get(m.memory_id) == result_id(asdict(m))]
    memories = tuple(m for m in context.memories if m not in removed_memories)
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
        'memories': {'omitted_count': len(removed_memories),
                     'sha256': result_id([asdict(m) for m in removed_memories]),
                     'read_path': {'operation': 'research_archive', 'collection': 'memories'}},
        'evidence_policy': 'Old evidence may use exact factual cards linked to authorized archives. '
                           'Latest/live receipts and measured or failed facts remain protected; '
                           'ordinary archived queries may fold into a source-qualified summary.'}
    return replace(context, research_state=state, memories=memories, history=history,
                   research_agenda={**context.research_agenda, 'working_projection': projection})
