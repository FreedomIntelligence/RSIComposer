"""Read-only projections owned by the default exploration harness.

No backend polls, URI reads, raw request/config dumps or private-attribute discovery.
Pagination is newest-first over a live bounded window; concurrent writes may move offsets.
"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import islice
import json
import math

from rsisys.loops.contracts.queries import (
    QuerySpec, QueryRequest, QueryResult, MAX_QUERY_LIMIT, MAX_QUERY_OFFSET,
)


DEFAULT_QUERIES = (
    QuerySpec('status', 'Read current or identified execution status without polling', ('directive_id',)),
    QuerySpec('metrics', 'Read observed numeric execution metrics without polling', ('directive_id',)),
    QuerySpec('tasks', 'List in-process task receipts, newest first', ('directive_id', 'mode')),
    QuerySpec('logs', 'Read available local history summaries; remote backend logs are not exposed', ('directive_id',)),
    QuerySpec('candidates', 'List known training candidates and application state', ('candidate_id',)),
    QuerySpec('model', 'Read the currently bound Actor model'),
    QuerySpec('artifacts', 'List known input/output/model artifact references; never read URI contents', ('directive_id',)),
    QuerySpec('experience', 'Read system experience summaries, newest first', ('kind', 'text')),
)


def _metrics(values):
    result = {}
    for key, value in islice(values.items(), 100):
        if isinstance(key, str) and len(key) <= 100 and type(value) in (int, float):
            try:
                if math.isfinite(value):
                    result[key] = value
            except OverflowError:
                pass
    return result


def _progress(progress):
    feedback = progress.feedback
    row = {
        'directive_id': progress.directive_id, 'mode': progress.directive.mode,
        'phase': progress.phase, 'status': feedback.status.value if feedback else 'running',
        'job': asdict(progress.job) if progress.job else None,
        'elapsed_seconds': progress.elapsed_seconds, 'metrics': _metrics(progress.metrics),
        'simulated': progress.directive.simulated or bool(feedback and feedback.simulated),
    }
    sample = progress.job_progress
    row['job_progress'] = ({'step': sample.step, 'total_steps': sample.total_steps,
                            'metrics': _metrics(sample.metrics), 'elapsed_seconds': sample.elapsed_seconds,
                            'observed_at': sample.observed_at} if sample is not None else None)
    if feedback:
        row.update(summary=feedback.summary[:1000], quality_metrics=_metrics(feedback.quality_metrics),
                   training_metrics=_metrics(feedback.training_metrics), execution_metrics=_metrics(feedback.execution_metrics))
    return row


class ExplorationQueries:
    """The owner may project its public local state; callers only use QueryProvider."""

    def __init__(self, harness):
        self.harness = harness

    def describe_queries(self):
        return DEFAULT_QUERIES

    def _receipts(self, request):
        # Bound scans as well as response pages. Receipts are local, not durable backend logs.
        for ident in islice(reversed(self.harness.commands.receipts), MAX_QUERY_OFFSET + MAX_QUERY_LIMIT + 1):
            state = self.harness.commands.receipts[ident]
            if request.filters.get('directive_id', ident) != ident:
                continue
            if request.filters.get('mode', state.directive.mode) != state.directive.mode:
                continue
            yield state

    def _rows(self, request):
        harness = self.harness
        if request.name in ('status', 'metrics'):
            ident = request.filters.get('directive_id')
            if ident is not None:
                if ident not in harness.commands.receipts:
                    return
                progress = harness.inspect(ident)
            else:
                progress = harness.pending()
            if progress is not None:
                yield _progress(progress), tuple(progress.feedback.artifact_refs) if progress.feedback else ()
        elif request.name == 'model':
            model = harness.execution_view().model
            if model is not None:
                yield {'model': asdict(model)}, ()
        elif request.name == 'tasks':
            for state in self._receipts(request):
                yield _progress(harness.inspect(state.directive.directive_id)), ()
        elif request.name == 'logs':
            entries = harness.memory.history(task_id=request.filters.get('directive_id'),
                                             limit=request.offset + request.limit + 1)
            for entry in reversed(entries):
                yield {'record_id': entry.record_id, 'directive_id': entry.task_id, 'kind': entry.kind,
                       'summary': entry.summary[:1000], 'simulated': entry.simulated,
                       'source_refs': list(entry.source_refs[:100])}, ()
        elif request.name == 'experience':
            kind = request.filters.get('kind')
            entries = harness.experience.retrieve(limit=request.offset + request.limit + 1,
                kinds=(kind,) if kind else (), text=request.filters.get('text'))
            for entry in reversed(entries):
                yield {'record_id': entry.record_id, 'directive_id': entry.directive_id, 'kind': entry.kind,
                       'scope': entry.scope, 'summary': entry.summary[:1000], 'simulated': entry.simulated,
                       'source_refs': list(entry.source_refs[:100]), 'run_id': entry.run_id}, ()
        elif request.name == 'candidates':
            for ident in islice(reversed(harness.state.candidate_results), MAX_QUERY_OFFSET + MAX_QUERY_LIMIT + 1):
                if request.filters.get('candidate_id', ident) != ident:
                    continue
                result = harness.state.candidate_results[ident]
                yield {'candidate_id': ident, 'base': asdict(result.base), 'model': asdict(result.model),
                       'applied': ident in harness.state.applied_candidates, 'simulated': result.simulated,
                       'affected_roles': list(result.candidate.affected_roles), 'metrics': _metrics(result.metrics)}, \
                    tuple(islice(result.candidate.changes.values(), MAX_QUERY_LIMIT))
        elif request.name == 'artifacts':
            seen = set()
            for state in self._receipts(request):
                feedback = state.feedback
                refs = state.directive.input_refs + (feedback.artifact_refs if feedback else ())
                for ref in refs:
                    if ref.artifact_id not in seen:
                        seen.add(ref.artifact_id)
                        yield {'artifact_id': ref.artifact_id, 'kind': ref.kind,
                               'directive_id': state.directive.directive_id,
                               'simulated': state.directive.simulated or bool(feedback and feedback.simulated)}, (ref,)
            if 'directive_id' not in request.filters:
                for result in harness.state.candidate_results.values():
                    for ref in result.candidate.changes.values():
                        if ref.artifact_id not in seen:
                            seen.add(ref.artifact_id)
                            yield {'artifact_id': ref.artifact_id, 'kind': ref.kind,
                                   'candidate_id': result.candidate.candidate_id, 'simulated': result.simulated}, (ref,)

    def query(self, request: QueryRequest) -> QueryResult:
        if not isinstance(request, QueryRequest):
            raise TypeError('Expected a QueryRequest')
        specs = {spec.name: spec for spec in self.describe_queries()}
        if request.name not in specs:
            raise ValueError(f'Unsupported exploration query: {request.name}')
        request.validate(specs[request.name])
        page = list(islice(self._rows(request), request.offset, request.offset + request.limit + 1))
        more = len(page) > request.limit
        selected = page[:request.limit]
        refs = {}
        truncated = False
        for _, values in selected:
            for ref in values:
                if ref.artifact_id not in refs and len(refs) >= MAX_QUERY_LIMIT:
                    truncated = True
                else:
                    refs[ref.artifact_id] = ref
        data = {'items': [row for row, _ in selected], 'available': bool(selected), 'refs_truncated': truncated}
        if not selected:
            data['unavailable_reason'] = ('No local log summaries in this page; remote backend logs are not exposed'
                                          if request.name == 'logs' else 'No matching observations in this page')
        offset = request.offset + request.limit
        next_cursor = str(offset) if more and offset <= MAX_QUERY_OFFSET else None
        data['window_exhausted'] = more and next_cursor is None
        values = tuple(refs.values())
        version = sha256(json.dumps({'data': data, 'refs': [asdict(ref) for ref in values]},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()
        return deepcopy(QueryResult(request.name, 'exploration', datetime.now(timezone.utc).isoformat(),
                                    version, data, values, next_cursor,
                                    simulated=any(row.get('simulated', False) for row, _ in selected)))
