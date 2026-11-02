"""Research-oriented working context, derived without another model call.

Interpretations never authorize dropping evidence. Only duplicate, fingerprinted
learning memories and old completed decision scaffolding leave the working view;
receipts remain available to the ordinary token-budget compactor and archive.
"""
from copy import deepcopy
from dataclasses import asdict, replace

from .context import result_id


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


def evidence_needs_protection(entry):
    """Conservative mechanical rule; never ask prose to classify contradictions."""
    details = entry.details
    if details.get('error') or details.get('validation_error'):
        return True
    for key in ('feedback', 'result'):
        receipt = details.get(key)
        if isinstance(receipt, dict) and (receipt.get('error') or receipt.get('quality_metrics')
                or receipt.get('status') in ('failed', 'cancelled', 'rejected')):
            return True
    return bool(details.get('quality_metrics'))


def research_working_view(context, jobs, attempts, *, recent=8):
    jobs = ordered_jobs(context, jobs)
    rank = {j['job_id']: i for i, j in enumerate(jobs)}
    active = {r['decision_id'] for r in context.decision_graph
              if r['status'] not in ('completed', 'failed', 'cancelled')}
    graph = {r['decision_id']: r for r in context.decision_graph}
    live_dependencies = set(active)
    pending_dependencies = list(active)
    while pending_dependencies:
        row = graph.get(pending_dependencies.pop(), {})
        for key in ('depends_on_decision_ids', 'source_decision_ids', 'adopted_decision_ids'):
            for ident in row.get(key, ()):
                if ident not in live_dependencies:
                    live_dependencies.add(ident)
                    pending_dependencies.append(ident)
    outstanding = {d.directive_id for d in context.outstanding}
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
        'evidence_policy': 'No feedback or operation receipt is removed by this projection. '
                           'Token compaction remains separately budgeted and source-linked.'}
    return replace(context, research_state=state, memories=memories, history=history,
                   research_agenda={**context.research_agenda, 'working_projection': projection})
