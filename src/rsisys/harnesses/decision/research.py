"""Small model-owned research notebook; no scheduler or scientific authority."""

import json
from copy import deepcopy
from pathlib import Path

from rsisys.runtime.storage.files import atomic_write, json_bytes

KINDS = ('agenda', 'todo', 'stage_summary')
MAX_PATCH_CHANGES = 32


class ResearchInputRejected(ValueError):
    """Expected notebook state/input conflict, safe to return to the model."""


def _text(value, name, maximum=16000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{name} must be nonempty text of at most {maximum} characters')
    return value


def _refs(value):
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError('evidence_refs must be an explicit list of at most 64 references')
    return [_text(ref, 'evidence reference', 512) for ref in value]


def page_offset(cursor, revision):
    if cursor is None:
        return 0
    if not isinstance(cursor, str):
        raise ValueError('Invalid cursor')
    prefix, separator, offset = cursor.rpartition(':')
    if not separator or prefix != str(revision) or not offset.isascii() or not offset.isdecimal():
        raise ValueError('Stale or invalid cursor; restart pagination')
    return int(offset)


def page_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('limit must be an integer from 1 to 32')


class ResearchAgenda:
    """Single-writer copy-on-write store with a complete, replayable revision log.

    IDs are supplied by the model. Only explicitly patched IDs/fields change.
    Empty evidence means no evidence supplied, never independent verification.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._items = {}
        self._revisions = []
        if self.path is not None and self.path.exists():
            self.restore(json.loads(self.path.read_text(encoding='utf-8')))

    @property
    def revision(self):
        return len(self._revisions)

    @staticmethod
    def validate_patch(arguments):
        required = {'expected_revision', 'upserts', 'remove_ids', 'reason', 'evidence_refs'}
        if not isinstance(arguments, dict) or set(arguments) != required:
            raise ValueError('Patch requires expected_revision, upserts, remove_ids, reason and evidence_refs')
        revision = arguments['expected_revision']
        if type(revision) is not int or revision < 0:
            raise ValueError('expected_revision must be a nonnegative integer')
        _text(arguments['reason'], 'reason', 4000)
        _refs(arguments['evidence_refs'])
        upserts, removals = arguments['upserts'], arguments['remove_ids']
        if (not isinstance(upserts, list) or not isinstance(removals, list)
                or not 1 <= len(upserts) + len(removals) <= MAX_PATCH_CHANGES):
            raise ValueError('Patch requires between 1 and 32 explicit changes')
        ids = []
        for item in upserts:
            if not isinstance(item, dict) or 'id' not in item or set(item) - {
                    'id', 'kind', 'text', 'status', 'evidence_refs'}:
                raise ValueError('Invalid research item fields')
            ids.append(_text(item['id'], 'id', 128))
            if 'kind' in item and item['kind'] not in KINDS:
                raise ValueError('Unknown research item kind')
            if 'text' in item:
                _text(item['text'], 'text')
            if 'status' in item:
                _text(item['status'], 'status', 128)
            if 'evidence_refs' in item:
                _refs(item['evidence_refs'])
        ids.extend(_text(ident, 'id', 128) for ident in removals)
        if len(set(ids)) != len(ids):
            raise ValueError('An ID may occur only once per patch')
        return deepcopy(arguments)

    def patch(self, *, expected_revision, upserts, remove_ids, reason, evidence_refs):
        arguments = self.validate_patch(dict(expected_revision=expected_revision, upserts=upserts,
            remove_ids=remove_ids, reason=reason, evidence_refs=evidence_refs))
        if expected_revision != self.revision:
            raise ResearchInputRejected(f'Stale revision: expected {expected_revision}, current {self.revision}')
        items = deepcopy(self._items)
        revision = self.revision + 1
        for patch in arguments['upserts']:
            ident = patch['id']
            if ident not in items and not {'kind', 'text'} <= set(patch):
                raise ResearchInputRejected('New items require kind and text')
            item = items.get(ident, {'status': 'open', 'evidence_refs': []})
            item.update(patch)
            item['revision'] = revision
            items[ident] = item
        for ident in arguments['remove_ids']:
            if ident not in items:
                raise ResearchInputRejected(f'Cannot remove unknown ID: {ident}')
            del items[ident]
        revisions = [*self._revisions, dict(revision=revision, **arguments)]
        payload = {'version': 1, 'items': list(items.values()), 'revisions': revisions}
        if self.path is not None:
            atomic_write(self.path, json_bytes(payload))
        self._items, self._revisions = items, revisions
        return {'revision': revision, 'changed_ids': [item['id'] for item in upserts],
                'removed_ids': list(remove_ids), 'evidence_refs': deepcopy(evidence_refs)}

    def read(self, *, view='items', kind=None, limit=20, cursor=None):
        page_limit(limit)
        if view not in ('items', 'revisions') or (kind is not None and kind not in KINDS):
            raise ValueError('Invalid research view or kind')
        if view == 'revisions' and kind is not None:
            raise ValueError('kind is only supported for items')
        # Bind cursor to both the revision and selected view/filter.
        scope = f'{self.revision}:{view}:{kind or "all"}'
        try:
            offset = page_offset(cursor, scope)
        except ValueError as exc:
            raise ResearchInputRejected(str(exc)) from exc
        rows = list(self._items.values()) if view == 'items' else self._revisions
        rows = [row for row in rows if kind is None or row['kind'] == kind]
        end = offset + limit
        return deepcopy({'revision': self.revision, 'view': view, 'items': rows[offset:end],
                         'total': len(rows), 'next_cursor': f'{scope}:{end}' if end < len(rows) else None,
                         'authority': 'model_notes_not_scientific_truth'})

    def context(self, *, limit=6, text_chars=400):
        """Small navigation view; full text and all revisions remain in read()."""
        page_limit(limit)
        if type(text_chars) is not int or not 1 <= text_chars <= 16000:
            raise ValueError('text_chars must be an integer from 1 to 16000')
        rows = list(self._items.values())
        recent = sorted(rows, key=lambda item: item['revision'], reverse=True)[:limit]
        return {'revision': self.revision, 'total': len(rows),
                'authority': 'model_notes_not_scientific_truth',
                'items': [{'id': item['id'], 'kind': item['kind'], 'revision': item['revision'],
                           'status': item['status'], 'text': item['text'][:text_chars],
                           'text_truncated': len(item['text']) > text_chars,
                           'evidence_refs': item['evidence_refs'][:2],
                           'evidence_refs_truncated': len(item['evidence_refs']) > 2} for item in recent],
                'read_operation': 'research_read'}

    def export(self):
        return deepcopy({'version': 1, 'items': list(self._items.values()), 'revisions': self._revisions})

    def restore(self, payload):
        """Validate a complete archive by replay before swapping live state; no I/O."""
        if (not isinstance(payload, dict) or set(payload) != {'version', 'items', 'revisions'}
                or type(payload['version']) is not int or payload['version'] != 1):
            raise ValueError('Invalid research archive')
        if not isinstance(payload['items'], list) or not isinstance(payload['revisions'], list):
            raise ValueError('Invalid research archive lists')
        candidate = ResearchAgenda()
        for revision in payload['revisions']:
            if not isinstance(revision, dict):
                raise ValueError('Invalid research revision')
            args = dict(revision)
            number = args.pop('revision', None)
            if type(number) is not int or number != candidate.revision + 1:
                raise ValueError('Invalid research revision order')
            candidate.patch(**args)
        if candidate.export() != payload:
            raise ValueError('Research archive does not match its revision log')
        self._items, self._revisions = candidate._items, candidate._revisions
