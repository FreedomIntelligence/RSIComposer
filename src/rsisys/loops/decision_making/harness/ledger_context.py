"""Select archived execution state for a request, never for action validation.

This is harness policy, not feedback interpretation. Exact current/live records
and a recent comparison window remain resident; historical records can be read
by identity. Missing from a view never means untested, failed or unadopted.
"""
from copy import deepcopy
from dataclasses import asdict, replace

from rsisys.runtime.json_tools import result_id
from .research_state import archive_available, live_evidence_ids


LEDGER_COLLECTIONS = {'candidates': 'candidate_id',
    'candidate_evaluations': 'candidate_id', 'candidate_origins': 'candidate_id',
    'directive_provenance': 'directive_id'}


def _candidate_refs(value):
    """Only explicit candidate fields; prose and model versions are not IDs."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'candidate_id' or key.endswith('_candidate_id'):
                if isinstance(item, str):
                    found.add(item)
            elif key == 'candidate_ids' or key.endswith('_candidate_ids'):
                if isinstance(item, (list, tuple)):
                    found.update(i for i in item if isinstance(i, str))
            else:
                found.update(_candidate_refs(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_candidate_refs(item))
    return found


def working_ledger_context(context, *, recent=8):
    """Archive-backed selection before token compaction, including native harnesses.

    The full context is still used by the decision parser and execution checks.
    Collections without an authorized read capability remain verbatim. Live
    dependencies are never trimmed to satisfy this selection window.
    """
    if type(recent) is not int or recent < 1:
        raise ValueError('recent must be positive')
    live_decisions, directives = live_evidence_ids(context)
    if context.last_feedback is not None:
        directives.add(context.last_feedback.directive_id)
    updates, navigation = {}, {}

    def select(name, rows, identity, keep):
        selected = tuple(deepcopy(row) for row in rows if getattr(row, identity) in keep)
        if len(selected) < len(rows):
            updates[name] = selected
            navigation[name] = {'operation': 'research_archive', 'collection': name,
                'identity_field': identity, 'total': len(rows), 'visible': len(selected),
                'sha256': result_id([asdict(row) for row in rows])}
        return selected

    candidate_collections = ('candidates', 'candidate_evaluations', 'candidate_origins')
    if all(archive_available(context, name) for name in candidate_collections):
        keep = {row.candidate_id for row in context.candidates[-recent:]}
        keep.update(list(context.candidate_origins)[-recent:])
        keep.update(row.candidate_id for row in context.candidate_evaluations[-recent:])
        # Keep a recent failed/zero-score counterexample even after many successes.
        negatives = [row for row in context.candidate_evaluations
                     if row.status != 'completed' or any(v <= 0 for v in row.quality_metrics.values())]
        keep.update(row.candidate_id for row in negatives[-2:])
        keep.update(ident for ident, origin in context.candidate_origins.items() if origin in live_decisions)
        for row in context.decision_graph:
            if row['decision_id'] in live_decisions:
                keep.update(_candidate_refs(row))
        # Last confirmed adoption is an exact receipt, not a guessed match to a
        # model alias. The execution model identity remains visible separately.
        adopted = next((row for row in reversed(context.decision_graph)
                        if row.get('adopted_candidate_ids')), {})
        keep.update(adopted.get('adopted_candidate_ids', ()))
        for value in (*context.outstanding, *([context.last_feedback] if context.last_feedback else [])):
            keep.update(_candidate_refs(asdict(value)))
        for execution in (context.execution, *context.executions.values()):
            if execution.progress is not None:
                keep.update(_candidate_refs(asdict(execution.progress.directive)))
        if context.operation_results:
            keep.update(_candidate_refs(asdict(context.operation_results[-1])))
        select('candidates', context.candidates, 'candidate_id', keep)
        select('candidate_evaluations', context.candidate_evaluations, 'candidate_id', keep)
        origins = {key: value for key, value in context.candidate_origins.items() if key in keep}
        if len(origins) < len(context.candidate_origins):
            updates['candidate_origins'] = origins
            navigation['candidate_origins'] = {'operation': 'research_archive',
                'collection': 'candidate_origins', 'identity_field': 'candidate_id',
                'total': len(context.candidate_origins), 'visible': len(origins),
                'sha256': result_id(context.candidate_origins)}
    if archive_available(context, 'directive_provenance'):
        directives.update(row.directive_id for row in updates.get(
            'candidate_evaluations', context.candidate_evaluations))
        directives.update(row.directive_id for row in context.directive_provenance[-recent:])
        select('directive_provenance', context.directive_provenance, 'directive_id', directives)
    if not navigation:
        return context
    prior = context.research_agenda.get('ledger_archives', {})
    return replace(context, **updates, research_agenda={**context.research_agenda,
        'ledger_archives': {**prior, **navigation},
        'ledger_view_note': 'Selected exact execution records, not complete history. '
            'Read older candidates, evaluations, origins and model provenance by ID. '
            'Absence is not a result; archive edges may point outside the visible decision graph.'})
