"""Rolling token budgets; raw histories and receipts remain in their stores."""
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Protocol


from rsisys.runtime.storage.json_tools import encoded, result_id


class ContextBudgetError(ValueError):
    """Fixed state and protected feedback cannot fit; nothing was discarded."""


class ContextCompactor(Protocol):
    def compact(self, context, *, recent_limit=32): ...


class IdentityContextCompactor:
    """A replacement harness owns its native conversation and compaction."""
    def compact(self, context, *, recent_limit=32):
        return deepcopy(context)


def working_decision_graph(graph, candidate_origins, *, recent=32):
    """Bound the working ledger view; original identities/relations are archived.

    Keep every active decision and its transitive dependencies, selected candidate
    producers, recent decisions and recent Attempt representatives. Historical
    completed edges can point into the archive instead of pinning entire chains.
    Large live graphs still fail explicitly; no active dependency is truncated.
    """
    if len(graph) <= recent * 2:
        return tuple(deepcopy(graph))
    indexed = {row['decision_id']: row for row in graph}
    keep = {row['decision_id'] for row in graph[-recent:]} | set(candidate_origins.values())
    attempts = {}
    active = set()
    for row in graph:
        if row['status'] not in ('completed', 'failed', 'cancelled'):
            active.add(row['decision_id'])
        if row.get('research_attempt_id'):
            attempts.pop(row['research_attempt_id'], None)
            attempts[row['research_attempt_id']] = row['decision_id']
    keep.update(list(attempts.values())[-8:])
    keep.update(active)
    pending = list(active)
    visited = set()
    while pending:
        ident = pending.pop()
        if ident in visited:
            continue
        visited.add(ident)
        row = indexed.get(ident, {})
        for name in ('depends_on_decision_ids', 'source_decision_ids', 'adopted_decision_ids'):
            for ident in row.get(name, ()):
                keep.add(ident)
                pending.append(ident)
    return tuple(deepcopy(row) for row in graph if row['decision_id'] in keep)


def with_working_decision_graph(context):
    # Only project when the advertised read capability can recover exact IDs.
    archives = [spec for spec in context.operations if spec.name == 'research_archive' and spec.read_only]
    if not any('decision_graph' in spec.input_schema.get('properties', {}).get(
            'collection', {}).get('enum', ()) for spec in archives):
        return context
    graph = working_decision_graph(context.decision_graph, context.candidate_origins)
    if len(graph) == len(context.decision_graph):
        return context
    navigation = {'collection': 'decision_graph', 'operation': 'research_archive',
        'total': len(context.decision_graph), 'visible': len(graph),
        'sha256': result_id(context.decision_graph),
        'note': 'Working view only. Read archived decision_id or paginated ledger for older decisions. '
                'Historical edges may reference archived nodes. Missing from this view does not mean nonexistent or failed.'}
    return replace(context, decision_graph=graph,
                   research_agenda={**context.research_agenda, 'decision_graph_archive': navigation})


class BasicContextCompactor:
    """Keep full context below 80k, then fold old records toward a 24k request.

    measure(context) must price the complete model request, including its prompt
    and capabilities. Without a tokenizer the UTF-8 byte count is an explicitly
    conservative bound. The last tool result, its two newest history entries and
    all live control/feedback remain present with their required facts. Structured
    evidence may use factual cards only with authorized archive links. Ordinary
    query cards and old archived measurements may fold; recent measurements/failures
    and live observations remain protected. Oversized protected
    facts fail clearly rather than truncating an observation. Folded fingerprints and a
    source-qualified rolling summary are portable snapshot state, not an archive.
    """
    streams = ('history', 'experience', 'memories', 'operation_results')

    def __init__(self, *, trigger_tokens=80000, target_tokens=24000, summary_tokens=4096,
                 max_target_tokens=None, target_increment_tokens=12000, input_budget_tokens=None):
        for name, value in (('trigger_tokens', trigger_tokens), ('target_tokens', target_tokens),
                            ('summary_tokens', summary_tokens)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not summary_tokens < target_tokens < trigger_tokens:
            raise ValueError('Require summary_tokens < target_tokens < trigger_tokens')
        maximum = target_tokens if max_target_tokens is None else max_target_tokens
        if (type(maximum) is not int or not target_tokens <= maximum < trigger_tokens
                or type(target_increment_tokens) is not int or target_increment_tokens < 1
                or (maximum - target_tokens) % target_increment_tokens):
            raise ValueError('Adaptive target must use positive increments up to a ceiling below trigger_tokens')
        self.trigger_tokens, self.target_tokens, self.summary_tokens = trigger_tokens, target_tokens, summary_tokens
        self.max_target_tokens, self.target_increment_tokens = maximum, target_increment_tokens
        if input_budget_tokens is not None and (type(input_budget_tokens) is not int
                or input_budget_tokens < trigger_tokens):
            raise ValueError('input_budget_tokens must cover trigger_tokens and reserve model output separately')
        self.input_budget_tokens = input_budget_tokens
        self._folded = {name: set() for name in self.streams}
        self._summary = {}
        self._project_evidence = False
        self.compactions = 0
        self.last_measurement = {}

    def export_state(self):
        return deepcopy({'folded': {k: sorted(v) for k, v in self._folded.items()},
                         'summary': self._summary, 'compactions': self.compactions,
                         'project_evidence': self._project_evidence})

    def restore_state(self, state):
        if (not isinstance(state, dict) or set(state) - {'folded', 'summary', 'compactions', 'project_evidence'}
                or not {'folded', 'summary', 'compactions'} <= state.keys()
                or not isinstance(state['folded'], dict) or set(state['folded']) != set(self.streams)
                or not isinstance(state['summary'], dict)
                or type(state['compactions']) is not int or state['compactions'] < 0
                or type(state.get('project_evidence', False)) is not bool):
            raise ValueError('Invalid context compactor state')
        for values in state['folded'].values():
            if not isinstance(values, list) or any(not isinstance(v, str) or len(v) != 64 for v in values):
                raise ValueError('Invalid folded context fingerprints')
        self._folded = {k: set(v) for k, v in state['folded'].items()}
        self._summary, self.compactions = deepcopy(state['summary']), state['compactions']
        self._project_evidence = state.get('project_evidence', False)

    def compact(self, context, *, recent_limit=32, measure=None, summarize=None):
        """Try the base target first; only budget failures may use bounded headroom.

        Each attempt commits atomically. Real summarizer calls, if necessary,
        still use the caller's backend/budget. Other errors are never retried.
        The configured target stays unchanged; future pressure starts at it again.
        """
        attempts = []
        targets = list(range(self.target_tokens, self.max_target_tokens + 1, self.target_increment_tokens))
        for target in targets:
            try:
                result = self._compact(context, recent_limit=recent_limit, measure=measure,
                                       summarize=summarize, target_tokens=target)
            except ContextBudgetError as exc:
                attempts.append({'target_tokens': target, 'status': 'budget_exceeded',
                    'error': str(exc), **{k: self.last_measurement[k] for k in
                        ('minimum_input_tokens', 'summary_token_limit') if k in self.last_measurement}})
                self.last_measurement.update(configured_target_tokens=self.target_tokens,
                    max_target_tokens=self.max_target_tokens, target_attempts=deepcopy(attempts))
                if target == targets[-1]:
                    # A compression goal is not the model's context capacity.
                    # Only an explicitly declared input budget authorizes this
                    # bounded fallback; reserve output/headroom at the caller.
                    # Try to leave room for provenance and the rolling summary,
                    # then at most once use the complete hard input allowance.
                    hard = self.input_budget_tokens
                    minimum = self.last_measurement.get('minimum_input_tokens', hard)
                    if hard is not None and target < hard and minimum <= hard:
                        proposed = min(hard, max(target + 1,
                            minimum + self.summary_tokens + 1024))
                        targets.append(proposed)
                        if proposed < hard:
                            targets.append(hard)
                    else:
                        self.last_measurement.update(input_budget_tokens=hard,
                            soft_target_exceeded=target > self.max_target_tokens)
                        raise
            else:
                attempts.append({'target_tokens': target, 'status': 'accepted'})
                self.last_measurement.update(configured_target_tokens=self.target_tokens,
                    max_target_tokens=self.max_target_tokens, target_attempts=deepcopy(attempts),
                    input_budget_tokens=self.input_budget_tokens,
                    soft_target_exceeded=target > self.max_target_tokens)
                return result

    def _compact(self, context, *, recent_limit, measure, summarize, target_tokens):
        if type(recent_limit) is not int or recent_limit < 1:
            raise ValueError('recent_limit must be positive')
        from .research_state import bounded_evidence_view
        from .ledger_context import working_ledger_context
        context = bounded_evidence_view(context)
        context = working_ledger_context(context)
        context = with_working_decision_graph(context)
        meter = measure or (lambda c: len(encoded(asdict(c)).encode('utf-8')))
        def count(view):
            value = meter(view)
            if type(value) is not int or value < 0:
                raise ValueError('Token counter must return a nonnegative integer')
            return value
        from .research_state import (evidence_working_view,
                                     full_history_ids, live_evidence_ids, resident_evidence_indices)
        project_evidence = self._project_evidence
        raw = {name: tuple(getattr(context, name)) for name in self.streams}
        fingerprints = {name: [result_id(row.payload() if name == 'operation_results' else asdict(row))
                               for row in rows] for name, rows in raw.items()}
        protected = {name: set() for name in self.streams}
        research_protected = full_history_ids(context) | set(context.research_state.get('protected_history_ids', ()))
        protected['history'].update(i for i, row in enumerate(raw['history'])
                                    if row.record_id in research_protected)
        protected['history'].update(resident_evidence_indices(context, 'history'))
        live_decisions, live_directives = live_evidence_ids(context)
        protected['experience'].update(i for i, row in enumerate(raw['experience'])
            if row.record_id in research_protected
            or row.details.get('decision_id') in live_decisions or row.directive_id in live_directives
            or context.last_feedback is not None and row.directive_id == context.last_feedback.directive_id)
        protected['experience'].update(resident_evidence_indices(context, 'experience'))
        protected_results = set(context.research_state.get('protected_operation_result_ids', ()))
        protected_results.update(result_id(row.details['result']) for row in raw['history']
            if row.record_id in research_protected and isinstance(row.details.get('result'), dict))
        latest_result_indices = {key: i for i, key in enumerate(fingerprints['operation_results'])}
        protected['operation_results'].update(i for i, key in enumerate(fingerprints['operation_results'])
            if key in protected_results and i == latest_result_indices[key])
        protected['operation_results'].update(resident_evidence_indices(context, 'operation_results'))
        if raw['operation_results']:
            protected['operation_results'].add(len(raw['operation_results'])-1)
        kept = {name: {i for i, key in enumerate(fingerprints[name])
                       if key not in self._folded[name] or i in protected[name]} for name in self.streams}
        previous_summary = self._summary or context.context_summary

        def view(indices, summary):
            projected = {name: tuple(deepcopy(row) for i, row in enumerate(rows) if i in indices[name])
                         for name, rows in raw.items()}
            # Compact payloads, not the lightweight references needed by input_ids.
            from ..action_space import OperationResult
            by_simulation = {False: {}, True: {}}
            retained = {ref for row in projected['operation_results'] for ref in row.artifacts}
            for i, row in enumerate(raw['operation_results']):
                if i not in indices['operation_results']:
                    for ref in row.artifacts:
                        if ref not in retained:
                            by_simulation[row.simulated][ref] = None
            refs = tuple(OperationResult('Archived operation artifact references', tuple(group), simulated)
                         for simulated, group in by_simulation.items() if group)
            projected['operation_results'] = (*refs, *projected['operation_results'])
            working = replace(context, **projected, context_summary=deepcopy(summary))
            return evidence_working_view(working) if project_evidence else working

        current = view(kept, previous_summary)
        before = count(current)
        self.last_measurement = {'input_tokens': before, 'trigger_tokens': self.trigger_tokens,
                                 'target_tokens': target_tokens, 'compacted': False}
        if before < self.trigger_tokens:
            return current
        project_evidence = True
        current = view(kept, previous_summary)
        after = count(current)
        if after <= target_tokens:
            self._project_evidence = True
            self.compactions += 1
            self.last_measurement.update(compacted=True, output_tokens=after, compaction=self.compactions)
            return current
        # Fold identical receipts together: persisted state is content-addressed,
        # so retaining half a duplicate group would change the view on restore.
        groups = {}
        for name in self.streams:
            order = reversed(range(len(raw[name]))) if name == 'memories' else range(len(raw[name]))
            for i in order:
                if i in kept[name] and i not in protected[name]:
                    groups.setdefault((name, fingerprints[name][i]), []).append(i)
        removable = list(groups.items())
        minimal = {name: indices & protected[name] for name, indices in kept.items()}
        # The rolling summary is an input to its replacement, not an
        # irreducible fact. Price resident records without the old summary;
        # the new summary and its provenance are priced before committing.
        minimal_view = view(minimal, {})
        minimum_tokens = count(minimal_view)
        self.last_measurement.update(minimum_input_tokens=minimum_tokens,
            protected_records={name: len(indices) for name, indices in minimal.items()})
        if minimum_tokens > target_tokens:
            self.last_measurement['minimum_field_bytes'] = {
                name: len(encoded(value).encode('utf-8'))
                for name, value in asdict(minimal_view).items()}
            raise ContextBudgetError('Irreducible factual evidence, live state and latest complete feedback '
                                     'exceed target_tokens; no compaction state was committed')
        # Reserve space for the rolling summary and compact provenance. Search by
        # whole records, never split a tool call/result to satisfy the budget.
        reserve = self.summary_tokens + min(512, target_tokens // 8)
        desired = target_tokens - reserve
        def after_removing(n):
            indices = {name: set(values) for name, values in kept.items()}
            for (name, _), members in removable[:n]:
                indices[name].difference_update(members)
            return indices
        lo, hi = 0, len(removable)
        while lo < hi:
            mid = (lo+hi)//2
            if count(view(after_removing(mid), {})) <= desired:
                hi = mid
            else:
                lo = mid+1
        def label(name, row):
            ident = result_id(row) if name == 'operation_results' else row.get('record_id', row.get('memory_id'))
            return f'{name}:{ident}'
        def draft(indices):
            removed = {name: [raw[name][i].payload() if name == 'operation_results' else asdict(raw[name][i])
                              for i in sorted(kept[name]-indices[name])]
                       for name in self.streams}
            sources = {name: {'omitted_count': len(rows), 'sha256': result_id(rows),
                              'first_omitted': label(name, rows[0]), 'last_omitted': label(name, rows[-1]),
                              'read_path': {'operation': 'research_archive', 'collection': name,
                                            'record_id': 'source label after collection prefix'}}
                       for name, rows in removed.items() if rows}
            return removed, {'kind': 'rolling_context_summary', 'authority': 'unverified_summary',
                'note': 'Older measurements and failures may be archived. Absence here is not evidence of '
                        'success or of an unevaluated candidate. Consult raw research_archive sources '
                        'for original receipts and qualifications.',
                'text': '', 'sources': sources, 'compaction': self.compactions+1}
        # The initial reserve is a heuristic, not the size of the actual source
        # manifest. Fold additional eligible records if provenance itself needs
        # more room. A model summarizer needs its configured output allowance;
        # a tiny positive remainder is not a usable structured summary budget.
        # Reject insufficient headroom before inference so only this budget
        # failure can use the bounded adaptive target. Increasing
        # steps bound the number of full-request meter calls on long histories.
        required_summary_tokens = self.summary_tokens if summarize is not None else 1
        extra = 1
        while True:
            chosen = after_removing(lo)
            removed, summary = draft(chosen)
            if not summary['sources']:
                raise ContextBudgetError('No old records can be compacted without losing latest feedback')
            available = target_tokens - count(view(chosen, summary))
            if available >= required_summary_tokens + 64 or lo == len(removable):
                break
            lo = min(len(removable), lo + extra)
            extra *= 2
        summary_limit = min(self.summary_tokens, available - 64)
        self.last_measurement['summary_token_limit'] = max(0, summary_limit)
        if summary_limit < required_summary_tokens:
            raise ContextBudgetError('Required facts and summary provenance leave insufficient summary budget; '
                                     'no compaction state was committed')
        if summarize is not None:
            text = summarize(deepcopy(removed), deepcopy(previous_summary), summary_limit)
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Context summarizer must return nonempty text')
        else:
            # Portable scripted-policy fallback. Model policies inject their own
            # summarizer; sources are navigation, never verified scientific truth.
            excerpts = [{'stream': name, 'source': result_id(row), 'record': row}
                        for name, rows in removed.items() for row in rows[-2:]]
            text = encoded({'previous': previous_summary.get('text', ''), 'recent_sources': excerpts})
            text = text.encode('utf-8')[:summary_limit].decode('utf-8', errors='ignore')
        summary['text'] = text
        projected = view(chosen, summary)
        after = count(projected)
        if after > target_tokens:
            raise ContextBudgetError('Summary plus fixed state and required facts and latest evidence references exceed target_tokens; '
                                     'no compaction state was committed')
        # Commit only after the summary and complete final request fit.
        for name in self.streams:
            self._folded[name].update(fingerprints[name][i] for i in kept[name]-chosen[name])
        self._summary = deepcopy(summary)
        self._project_evidence = True
        self.compactions += 1
        self.last_measurement.update(compacted=True, output_tokens=after, compaction=self.compactions)
        return projected
