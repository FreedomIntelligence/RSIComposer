"""Research-oriented working context, derived without another model call.

Interpretations never authorize dropping evidence. Old receipts may use exact
factual cards with authorized links to originals; live receipts remain complete.
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
        if ((key in ('error', 'validation_error', 'metrics') or key.endswith('_metrics')) and value
                or key == 'status' and value in ('failed', 'cancelled', 'rejected')
                or key == 'evidence_card' or receipt_needs_protection(value)):
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
    return {h.record_id for h in context.history[-2:]} | {
        h.record_id for h in context.history
        if h.record_id in decisions or h.details.get('decision_id') in decisions
        or h.task_id in directives
        or latest_feedback is not None and h.kind == 'feedback' and h.task_id == latest_feedback
        or latest_result is not None and isinstance(h.details.get('result'), dict)
        and result_id(h.details['result']) == latest_result}


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
                        'mode', 'base_version', 'parent_system', 'digest')
                    or key.endswith(('_id', '_ids', '_ref', '_refs', '_metrics'))):
                facts[pointer] = deepcopy(item)
            else:
                facts.update(factual_fields(item, pointer))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            facts.update(factual_fields(item, path + '/' + str(index)))
    return facts


def evidence_working_view(context):
    """Bound old payloads, preserving every structured fact and exact originals.

    Capability checks are per collection. A card is retained as protected state,
    never fed through a lossy summary. If factual cards plus required live state
    are still too large, the token compactor must fail without committing.
    """
    full = full_history_ids(context)
    decisions, directives = live_evidence_ids(context)
    latest_feedback = context.last_feedback.directive_id if context.last_feedback else None
    live_results = {result_id(h.details['result']) for h in context.history
                    if h.record_id in full and isinstance(h.details.get('result'), dict)}
    projected = {}
    for collection in ('history', 'experience', 'operation_results'):
        rows = getattr(context, collection)
        if not archive_available(context, collection):
            continue
        output = []
        for index, row in enumerate(rows):
            original = asdict(row)
            if collection == 'operation_results':
                ident = result_id(original)
                required = index == len(rows) - 1 or ident in live_results
                already_card = 'evidence_card' in row.data
            else:
                ident = row.record_id
                required = (ident in full or ident in decisions
                    or row.details.get('decision_id') in decisions
                    or getattr(row, 'directive_id', None) in directives
                    or latest_feedback is not None
                    and getattr(row, 'directive_id', None) == latest_feedback)
                already_card = 'evidence_card' in row.details
            if required or already_card or not receipt_needs_protection(original):
                output.append(row)
                continue
            card = {'evidence_card': {'schema_version': 1, 'authority': 'exact_structured_fields',
                        'sha256': result_id(original),
                        'read_path': {'operation': 'research_archive', 'collection': collection,
                                      'record_id': ident}},
                    'facts': factual_fields(original)}
            updates = {'summary': 'Archived evidence; read original for prose and qualifications.'}
            updates['data' if collection == 'operation_results' else 'details'] = card
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
                           'Latest and live receipts remain complete; factual cards cannot be summarized away.'}
    return replace(context, research_state=state, memories=memories, history=history,
                   research_agenda={**context.research_agenda, 'working_projection': projection})
