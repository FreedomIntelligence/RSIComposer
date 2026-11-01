"""Replaceable, deterministic projections; archives remain owned by their stores."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from typing import Protocol


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def result_id(value):
    """Stable lookup key shared by the projection and archived tool receipts."""
    return sha256(encoded(value).encode('utf-8')).hexdigest()


class ContextCompactor(Protocol):
    def compact(self, context, *, recent_limit=32): ...


class IdentityContextCompactor:
    """Explicit opt-out for policies that own their context management."""

    def compact(self, context, *, recent_limit=32):
        return deepcopy(context)


class BasicContextCompactor:
    """Bound archival context by serialized characters, without an extra model call.

    Whole recent records are retained when they fit. Oversized records are
    represented by labeled excerpts and archive provenance, never silently
    truncated into apparently complete observations. Live control, candidates,
    objective and capability schemas are not archives and remain intact.
    """

    streams = ('history', 'memories', 'experience', 'operation_results')

    def __init__(self, *, recent_tail=8, stream_chars=16000, summary_chars=6000):
        for name, value, minimum in (('recent_tail', recent_tail, 1),
                                     ('stream_chars', stream_chars, 256),
                                     ('summary_chars', summary_chars, 4096)):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        self.recent_tail, self.stream_chars, self.summary_chars = recent_tail, stream_chars, summary_chars

    def compact(self, context, *, recent_limit=32):
        if type(recent_limit) is not int or recent_limit < 1:
            raise ValueError('recent_limit must be a positive integer')
        previous = context.context_summary
        projected = {}
        sources = deepcopy({key: value for key, value in previous.get('sources', {}).items()
                            if key != 'history'})
        excerpts = deepcopy([item for item in previous.get('excerpts', [])
                             if not item['source'].startswith('history:')])
        for name in self.streams:
            rows = getattr(context, name)
            tail = min(self.recent_tail, recent_limit) if name in ('history', 'memories') else self.recent_tail
            # recall() is newest-first; other streams are oldest-first.
            order = range(len(rows)) if name == 'memories' else reversed(range(len(rows)))
            kept, used = [], 2  # JSON array delimiters
            for index in list(order)[:tail]:
                size = len(encoded(asdict(rows[index]))) + (1 if kept else 0)
                if used + size > self.stream_chars:
                    # Keep a contiguous recent tail, not an arbitrary selection.
                    break
                kept.append(index)
                used += size
            kept.sort()
            projected[name] = tuple(deepcopy(rows[index]) for index in kept)
            kept_set = set(kept)
            omitted = [index for index in range(len(rows)) if index not in kept_set]
            if not omitted:
                continue
            digest = sha256()
            for row in rows:
                digest.update(encoded(asdict(row)).encode('utf-8'))
                digest.update(b'\n')
            def ref(index):
                row = rows[index]
                ident = (result_id(asdict(row)) if name == 'operation_results'
                         else getattr(row, 'record_id', getattr(row, 'memory_id', str(index))))
                return f'{name}:{ident}'
            sources[name] = {'input_count': len(rows), 'retained_count': len(kept),
                'omitted_count': len(omitted), 'sha256': digest.hexdigest(),
                'first_omitted': ref(omitted[0])[:256], 'last_omitted': ref(omitted[-1])[:256],
                'reference_labels_truncated': any(len(ref(index)) > 256 for index in (omitted[0], omitted[-1])),
                'read_path': {'operation': 'research_archive', 'collection': name,
                              'record_id': 'source label after collection prefix'},
                'scope': 'full_history' if name == 'history' else 'selected_context_records'}
            newest_omitted = omitted[:3] if name == 'memories' else omitted[-3:]
            for index in newest_omitted:
                row = rows[index]
                excerpts.append({'source': ref(index)[:256],
                    'source_label_truncated': len(ref(index)) > 256,
                    'source_refs': list(getattr(row, 'source_refs', ()))[:2],
                    'excerpt': getattr(row, 'summary', getattr(row, 'content', ''))[:240],
                    'simulated': getattr(row, 'simulated', None), 'incomplete': True})
        summary = {}
        if sources:
            summary = {'kind': 'extractive_navigation_summary', 'authority': 'not_scientific_truth',
                'archive_operation': 'research_archive',
                'note': 'Excerpts may omit evidence and qualifications. Read raw sources before concluding.',
                'sources': sources, 'excerpts': []}
            for excerpt in excerpts:
                candidate = {**summary, 'excerpts': [*summary['excerpts'], excerpt]}
                if len(encoded(candidate)) <= self.summary_chars:
                    summary = candidate
            # Extremely long externally supplied IDs can inflate provenance; hashes/counts still identify the view.
            if len(encoded(summary)) > self.summary_chars:
                for source in sources.values():
                    source.pop('first_omitted', None)
                    source.pop('last_omitted', None)
        return replace(context, **projected, context_summary=summary)
