"""Rolling token budgets; raw histories and receipts remain in their stores."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from typing import Protocol


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def result_id(value):
    return sha256(encoded(value).encode('utf-8')).hexdigest()


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

    Keep every active decision, candidate producer, one representative per attempt,
    and the transitive dependencies/sources/adoptions of retained nodes. Large live
    graphs still fail the budget explicitly; no active dependency is truncated.
    """
    if len(graph) <= recent * 2:
        return tuple(deepcopy(graph))
    indexed = {row['decision_id']: row for row in graph}
    keep = {row['decision_id'] for row in graph[-recent:]} | set(candidate_origins.values())
    attempts = {}
    for row in graph:
        if row['status'] not in ('completed', 'failed', 'cancelled'):
            keep.add(row['decision_id'])
        if row.get('research_attempt_id'):
            attempts[row['research_attempt_id']] = row['decision_id']
    keep.update(attempts.values())
    pending = list(keep)
    while pending:
        row = indexed.get(pending.pop(), {})
        for name in ('depends_on_decision_ids', 'source_decision_ids', 'adopted_decision_ids'):
            for ident in row.get(name, ()):
                if ident not in keep:
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
                'Missing from this view does not mean nonexistent or failed.'}
    return replace(context, decision_graph=graph,
                   research_agenda={**context.research_agenda, 'decision_graph_archive': navigation})


class BasicContextCompactor:
    """Keep full context below 80k, then fold old records toward a 24k request.

    measure(context) must price the complete model request, including its prompt
    and capabilities. Without a tokenizer the UTF-8 byte count is an explicitly
    conservative bound. The last tool result, its two newest history entries and
    all live control/feedback remain verbatim. Oversized protected state fails
    clearly rather than truncating an observation. Folded fingerprints and a
    source-qualified rolling summary are portable snapshot state, not an archive.
    """
    streams = ('history', 'experience', 'memories', 'operation_results')

    def __init__(self, *, trigger_tokens=80000, target_tokens=24000, summary_tokens=4096):
        for name, value in (('trigger_tokens', trigger_tokens), ('target_tokens', target_tokens),
                            ('summary_tokens', summary_tokens)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not summary_tokens < target_tokens < trigger_tokens:
            raise ValueError('Require summary_tokens < target_tokens < trigger_tokens')
        self.trigger_tokens, self.target_tokens, self.summary_tokens = trigger_tokens, target_tokens, summary_tokens
        self._folded = {name: set() for name in self.streams}
        self._summary = {}
        self.compactions = 0
        self.last_measurement = {}

    def export_state(self):
        return deepcopy({'folded': {k: sorted(v) for k, v in self._folded.items()},
                         'summary': self._summary, 'compactions': self.compactions})

    def restore_state(self, state):
        if (not isinstance(state, dict) or set(state) != {'folded', 'summary', 'compactions'}
                or not isinstance(state['folded'], dict) or set(state['folded']) != set(self.streams)
                or not isinstance(state['summary'], dict)
                or type(state['compactions']) is not int or state['compactions'] < 0):
            raise ValueError('Invalid context compactor state')
        for values in state['folded'].values():
            if not isinstance(values, list) or any(not isinstance(v, str) or len(v) != 64 for v in values):
                raise ValueError('Invalid folded context fingerprints')
        self._folded = {k: set(v) for k, v in state['folded'].items()}
        self._summary, self.compactions = deepcopy(state['summary']), state['compactions']

    def compact(self, context, *, recent_limit=32, measure=None, summarize=None):
        if type(recent_limit) is not int or recent_limit < 1:
            raise ValueError('recent_limit must be positive')
        context = with_working_decision_graph(context)
        meter = measure or (lambda c: len(encoded(asdict(c)).encode('utf-8')))
        def count(view):
            value = meter(view)
            if type(value) is not int or value < 0:
                raise ValueError('Token counter must return a nonnegative integer')
            return value
        raw = {name: tuple(getattr(context, name)) for name in self.streams}
        fingerprints = {name: [result_id(asdict(row)) for row in rows] for name, rows in raw.items()}
        protected = {name: set() for name in self.streams}
        protected['history'] = set(range(max(0, len(raw['history'])-2), len(raw['history'])))
        research_protected = set(context.research_state.get('protected_history_ids', ()))
        protected['history'].update(i for i, row in enumerate(raw['history'])
                                    if row.record_id in research_protected)
        protected_results = set(context.research_state.get('protected_operation_result_ids', ()))
        protected['operation_results'].update(i for i, key in enumerate(fingerprints['operation_results'])
                                               if key in protected_results)
        if raw['operation_results']:
            protected['operation_results'].add(len(raw['operation_results'])-1)
        kept = {name: {i for i, key in enumerate(fingerprints[name])
                       if key not in self._folded[name] or i in protected[name]} for name in self.streams}

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
            return replace(context, **projected, context_summary=deepcopy(summary or context.context_summary))

        current = view(kept, self._summary)
        before = count(current)
        self.last_measurement = {'input_tokens': before, 'trigger_tokens': self.trigger_tokens,
                                 'target_tokens': self.target_tokens, 'compacted': False}
        if before < self.trigger_tokens:
            return current
        removable = [(name, i) for name in self.streams
                     for i in (reversed(range(len(raw[name]))) if name == 'memories' else range(len(raw[name])))
                     if i in kept[name] and i not in protected[name]]
        minimal = {name: indices & protected[name] for name, indices in kept.items()}
        if count(view(minimal, self._summary)) >= self.target_tokens:
            raise ContextBudgetError('Fixed state and latest complete feedback exceed target_tokens; '
                                     'paginate the tool output or explicitly raise the budget')
        # Reserve space for the rolling summary and compact provenance. Search by
        # whole records, never split a tool call/result to satisfy the budget.
        reserve = self.summary_tokens + min(512, self.target_tokens // 8)
        desired = self.target_tokens - reserve
        def after_removing(n):
            indices = {name: set(values) for name, values in kept.items()}
            for name, i in removable[:n]:
                indices[name].remove(i)
            return indices
        lo, hi = 0, len(removable)
        while lo < hi:
            mid = (lo+hi)//2
            if count(view(after_removing(mid), self._summary)) <= desired:
                hi = mid
            else:
                lo = mid+1
        chosen = after_removing(lo)
        removed = {name: [asdict(raw[name][i]) for i in sorted(kept[name]-chosen[name])]
                   for name in self.streams}
        def label(name, row):
            ident = result_id(row) if name == 'operation_results' else row.get('record_id', row.get('memory_id'))
            return f'{name}:{ident}'
        sources = {name: {'omitted_count': len(rows), 'sha256': result_id(rows),
                          'first_omitted': label(name, rows[0]), 'last_omitted': label(name, rows[-1]),
                          'read_path': {'operation': 'research_archive', 'collection': name,
                                        'record_id': 'source label after collection prefix'}}
                   for name, rows in removed.items() if rows}
        if not sources:
            raise ContextBudgetError('No old records can be compacted without losing latest feedback')
        if summarize is not None:
            text = summarize(deepcopy(removed), deepcopy(self._summary), self.summary_tokens)
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Context summarizer must return nonempty text')
        else:
            # Portable scripted-policy fallback. Model policies inject their own
            # summarizer; sources are navigation, never verified scientific truth.
            excerpts = [{'stream': name, 'source': result_id(row), 'record': row}
                        for name, rows in removed.items() for row in rows[-2:]]
            text = encoded({'previous': self._summary.get('text', ''), 'recent_sources': excerpts})
            text = text.encode('utf-8')[:self.summary_tokens].decode('utf-8', errors='ignore')
        summary = {'kind': 'rolling_context_summary', 'authority': 'unverified_summary',
                   'note': 'Consult raw research_archive sources for original receipts and qualifications.',
                   'text': text, 'sources': sources, 'compaction': self.compactions+1}
        projected = view(chosen, summary)
        after = count(projected)
        if after > self.target_tokens:
            raise ContextBudgetError('Summary plus fixed state and complete latest feedback exceed target_tokens; '
                                     'no compaction state was committed')
        # Commit only after the summary and complete final request fit.
        for name in self.streams:
            self._folded[name].update(fingerprints[name][i] for i in kept[name]-chosen[name])
        self._summary = deepcopy(summary)
        self.compactions += 1
        self.last_measurement.update(compacted=True, output_tokens=after, compaction=self.compactions)
        return projected
