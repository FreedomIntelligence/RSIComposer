"""Research notebook capabilities using the existing ActionSpace/Tools boundary."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from functools import partial

from ...action_space import OperationCall, OperationResult, OperationSpec
from ...action_space.capabilities import Capabilities, Capability
from ..context import encoded, result_id
from ..research import KINDS, MAX_PATCH_CHANGES, ResearchAgenda, page_limit, page_offset


class _Definition:
    def __init__(self, spec, validator):
        self.spec, self.validator = spec, validator

    def validate(self, arguments):
        return OperationCall(self.spec.name, self.validator(arguments))


def _read_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) - {'view', 'kind', 'limit', 'cursor'}:
        raise ValueError('Read accepts only view, kind, limit and cursor')
    args = dict(view='items', kind=None, limit=4, cursor=None)
    args.update(arguments)
    page_limit(args['limit'])
    if (args['view'] not in ('items', 'revisions') or args['kind'] not in (*KINDS, None)
            or (args['view'] == 'revisions' and args['kind'] is not None)
            or (args['cursor'] is not None and not isinstance(args['cursor'], str))):
        raise ValueError('Invalid research read arguments')
    return deepcopy(args)


def _archive_arguments(arguments, *, experience=True):
    if not isinstance(arguments, dict) or set(arguments) - {'collection', 'record_id', 'limit', 'cursor'}:
        raise ValueError('Archive accepts only collection, record_id, limit and cursor')
    args = {'collection': 'history', 'record_id': None, 'limit': 8, 'cursor': None}
    args.update(arguments)
    page_limit(args['limit'])
    if args['collection'] == 'experience' and not experience:
        raise ValueError('Experience archive provider is not bound')
    if args['collection'] not in ('history', 'memories', 'operation_results',
                                  *(['experience'] if experience else []), 'research_items', 'research_revisions'):
        raise ValueError('Invalid archive collection')
    if args['record_id'] is not None and (not isinstance(args['record_id'], str) or not args['record_id']):
        raise ValueError('record_id must be nonempty text or null')
    if args['cursor'] is not None and not isinstance(args['cursor'], str):
        raise ValueError('cursor must be a string or null')
    return deepcopy(args)


class ResearchTools:
    """Inject the same agenda and memory instances as the decision harness.

    Read/archive are read-only; patch remains subject to existing permissions.
    No automatic installation or external discovery occurs here.
    """

    def __init__(self, agenda: ResearchAgenda, memory, *, experience=None):
        self.agenda, self.memory = agenda, memory
        self.experience = experience
        text = {'type': 'string', 'minLength': 1, 'maxLength': 128, 'pattern': r'\S'}
        refs = {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 512,
                                         'pattern': r'\S'}, 'maxItems': 64}
        schema = lambda props, required=(): {'type': 'object', 'properties': props,
                                           'required': list(required), 'additionalProperties': False}
        read_schema = schema({'view': {'enum': ['items', 'revisions']}, 'kind': {'enum': [*KINDS, None]},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 32}, 'cursor': {'type': ['string', 'null']}})
        read_schema['oneOf'] = [
            {'properties': {'view': {'enum': ['items']}}},
            {'properties': {'view': {'enum': ['revisions']}, 'kind': {'type': 'null'}}, 'required': ['view']},
        ]
        read = _Definition(OperationSpec('research_read',
            'Read model-owned agenda, todos, stage summaries or revision evidence. Notes are not scientific truth. '
            'Follow next_cursor; restart pagination after a revision changes.',
            read_schema, read_only=True), _read_arguments)
        patch_props = {'expected_revision': {'type': 'integer', 'minimum': 0},
            'upserts': {'type': 'array', 'maxItems': MAX_PATCH_CHANGES, 'items': schema({
                'id': text, 'kind': {'enum': list(KINDS)},
                'text': {'type': 'string', 'minLength': 1, 'maxLength': 16000, 'pattern': r'\S'},
                'status': text, 'evidence_refs': refs}, ('id',))},
            'remove_ids': {'type': 'array', 'items': text, 'maxItems': MAX_PATCH_CHANGES},
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 4000, 'pattern': r'\S'}, 'evidence_refs': refs}
        patch_schema = schema(patch_props, patch_props)
        patch_schema['anyOf'] = [{'properties': {name: {'minItems': 1}}}
                                for name in ('upserts', 'remove_ids')]
        patch = _Definition(OperationSpec('research_patch',
            'Atomically patch stable IDs at expected_revision. Omitted IDs and fields survive. New IDs require kind/text. '
            'Choose your own statuses and research process. Supply reason and evidence_refs (empty means no evidence). '
            f'Upserts plus removals must total 1..{MAX_PATCH_CHANGES}; IDs must be distinct across both lists. '
            'Read revision before retrying conflicts; summaries are model notes, not validated findings.',
            patch_schema), agenda.validate_patch)
        archive = _Definition(OperationSpec('research_archive',
            'Read raw history, memories, operation_results, research_items or research_revisions '
            + ('and experience ' if experience is not None else '') + 'as paginated 1000-character JSON chunks. '
            'Concatenate chunks for each record_id before parsing; history includes full tool receipts. '
            'Use exact record_id to retrieve a source; operation_results IDs are content SHA-256 hashes. '
            'Derived experience may require following its source_refs. Cursor is pinned to the archive digest.',
            schema({'collection': {'enum': ['history', 'memories', 'operation_results',
                                            'research_items', 'research_revisions',
                                            *(['experience'] if experience is not None else [])]},
                    'record_id': {'type': ['string', 'null'], 'minLength': 1}, 'cursor': {'type': ['string', 'null']},
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 32}}), read_only=True),
            partial(_archive_arguments, experience=experience is not None))
        self.capabilities = Capabilities((Capability(read, self._read), Capability(patch, self._patch),
                                          Capability(archive, self._archive)))
        self.definitions, self.bindings = self.capabilities.definitions, self.capabilities.bindings

    def describe(self):
        return tuple(self.capabilities)

    def _read(self, arguments):
        return OperationResult('Model-owned research notes', (), False,
                               data=self.agenda.read(**_read_arguments(arguments)))

    def _patch(self, arguments):
        return OperationResult('Research patch recorded', (), False, data=self.agenda.patch(**arguments))

    def _archive(self, arguments):
        args = _archive_arguments(arguments, experience=self.experience is not None)
        collection = args['collection']
        if collection == 'experience':
            if self.experience is None:
                raise ValueError('Experience archive provider is not bound')
            rows = [asdict(row) for row in self.experience.retrieve(limit=None)]
            identity = 'record_id'
        elif collection == 'operation_results':
            receipts = {}
            for entry in self.memory.history(kind='operation'):
                result = entry.details.get('result')
                if isinstance(result, dict):
                    ident = result_id(result)
                    row = receipts.setdefault(ident, {'record_id': ident, 'result': deepcopy(result),
                                                      'history_refs': []})
                    row['history_refs'].append(f'history:decision_maker:{entry.record_id}')
            rows, identity = list(receipts.values()), 'record_id'
        elif collection.startswith('research_'):
            rows = self.agenda.export()['items' if collection == 'research_items' else 'revisions']
            identity = 'id' if collection == 'research_items' else 'revision'
        else:
            rows = (self.memory.history() if collection == 'history'
                    else self.memory.recall(limit=2**63 - 1))
            rows = [asdict(row) for row in rows]
            identity = 'record_id' if collection == 'history' else 'memory_id'
        if args['record_id'] is not None:
            rows = [row for row in rows if str(row[identity]) == args['record_id']]
        # Pin the original prefix so receipts appended by this read do not
        # invalidate history pagination. Changed records still invalidate it.
        count = len(rows)
        if args['cursor'] is not None:
            parts = args['cursor'].rsplit(':', 3)
            if len(parts) != 4 or not parts[1].isascii() or not parts[1].isdecimal():
                raise ValueError('Invalid archive cursor')
            count = int(parts[1])
            if count > len(rows):
                raise ValueError('Stale archive cursor; restart pagination')
            rows = rows[:count]
        # Fingerprint only the selected records so an exact-ID read remains stable
        # while the harness appends receipts for the read operations themselves.
        digest = sha256()
        for row in rows:
            digest.update(encoded(row).encode('utf-8'))
            digest.update(b'\n')
        revision = digest.hexdigest()
        scope = f'{collection}:{args["record_id"] or "all"}:{count}:{revision}'
        start = page_offset(args['cursor'], scope)
        chunks, offset, total = [], 0, 0
        for row in rows:
            content = encoded(row)
            record_chunks = max(1, (len(content) + 999) // 1000)
            for chunk in range(max(0, start - offset), min(record_chunks, start + args['limit'] - offset)):
                chunks.append({'record_id': str(row[identity]), 'chunk': chunk, 'chunks': record_chunks,
                               'text': content[chunk * 1000:(chunk + 1) * 1000]})
            offset += record_chunks
            total += record_chunks
        end = start + len(chunks)
        return OperationResult('Raw research archive page', (), False, data={
            'revision': revision, 'collection': args['collection'], 'items': chunks,
            'found': bool(rows),
            'total_chunks': total, 'next_cursor': f'{scope}:{end}' if end < total else None})
