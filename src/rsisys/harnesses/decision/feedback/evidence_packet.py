"""Bounded views of captured feedback; raw evidence remains immutable."""
from copy import deepcopy

from .evidence import candidate_evidence
from rsisys.loops.contracts.receipts import receipt_task
from rsisys.runtime.storage.json_tools import encoded, result_id

# Only prose is excerpted. Numbers, identities, references, receipt facts and
# provenance are never rounded, rewritten, or inferred from narrative text.
_PROSE = frozenset(('objective', 'strategy', 'reason', 'summary', 'title',
                    'research_attempt_title', 'content', 'text', 'prompt',
                    'description', 'stdout', 'stderr'))


def feedback_text_preview(value, *, job_id, evidence_path, maximum=512):
    """Bound receipt prose without changing its journal original or factual fields.

    Keep both ends (including ordinary exception type prefixes) and an explicit
    gap. The descriptor addresses the original string, not its JSON-text page
    length. Even the smallest profile retains a short head and tail; if that and
    required facts do not fit, admission must fail rather than erase the receipt.
    """
    maximum = max(64, maximum)
    if not isinstance(value, str) or len(value) <= maximum:
        return deepcopy(value), None
    marker = '\n[... omitted ...]\n'
    supplied = maximum - len(marker)
    head = (supplied + 1) // 2
    tail = supplied - head
    return value[:head] + marker + value[-tail:], {
        'path': evidence_path, 'preview': 'head_tail',
        'supplied_characters': supplied, 'total_characters': len(value),
        'omitted_characters': len(value) - supplied, 'sha256': result_id(value),
        'read': {'operation': 'feedback_learning_read', 'job_id': job_id,
                 'evidence_path': evidence_path}}


def _excerpt(value, maximum, path='', omissions=None):
    omissions = [] if omissions is None else omissions
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            pointer = path + '/' + key.replace('~', '~0').replace('/', '~1')
            if key in _PROSE and isinstance(item, str) and len(item) > maximum:
                result[key] = item[:maximum]
                omissions.append({'path': pointer, 'supplied_characters': maximum,
                                  'total_characters': len(item), 'sha256': result_id(item)})
            elif ('metric' in key or key in ('provenance', 'decision_model',
                    'issuing_decision_model', 'learning_decision_model')
                    or key.endswith(('_id', '_ids', '_ref', '_refs', '_digest'))):
                result[key] = deepcopy(item)
            else:
                result[key] = _excerpt(item, maximum, pointer, omissions)
        return result
    if isinstance(value, (list, tuple)):
        return [_excerpt(item, maximum, path + '/' + str(index), omissions)
                for index, item in enumerate(value)]
    return deepcopy(value)


def evidence_packet(row, count, text_limit, *, preview_feedback=True):
    """Deterministic view of immutable raw evidence, with explicit coverage.

    Stream digests/references cover omitted records without making their IDs
    eligible citations. Record digests always identify the original, not its
    excerpt. Current summary/error prose may use head/tail previews; all current
    metrics, identities, status, adoption, candidates and jobs remain exact.
    Oversized indivisible facts still fail admission honestly.
    """
    raw = row['raw_input']
    manifest = {'version': 1, 'authority': 'untrusted_evidence',
                'raw_input_ref': 'journal:' + row['job_id'] + '/raw_input',
                'raw_input_digest': result_id(raw), 'streams': {}, 'records': [],
                'digests_identify': 'original_evidence_not_excerpts',
                'read': {'job_id': row['job_id'], 'path_parameter': 'evidence_path',
                         'offset_unit': 'characters', 'characters_per_limit': 128},
                'omitted_content_read': False}
    if 'dispatch_attempts' in row:
        manifest['dispatch_attempts_ref'] = 'journal:' + row['job_id'] + '/dispatch_attempts'
        manifest['dispatch_attempts_digest'] = result_id(row['dispatch_attempts'])
    source = 'feedback:' + row['receipt_digest']
    refs, coverage = [source], {}

    def project(record, path, ref=None, *, receipt=False, interpretation=False, excluded=()):
        omissions = []
        if receipt and record.get('task', {}).get('source') == 'lab':
            # Native producer keys carry no host semantics. Bound the complete
            # payload representation rather than protecting arbitrary *_id or
            # *metric fields, while retaining exact originals behind read paths.
            def native(value, pointer):
                text = encoded(value)
                preview, omission = feedback_text_preview(text, job_id=row['job_id'],
                    evidence_path=pointer, maximum=max(64, text_limit * 4))
                if omission is None:
                    return deepcopy(value)
                omission.update(sha256=result_id(value), representation='json_text')
                omissions.append(omission)
                return {'representation': 'json_preview', 'text': preview,
                        'complete': False, 'read': omission['read']}

            base = {**record, 'operation': {}, 'result': {k: v for k, v in record['result'].items()
                    if k not in ('output', 'references')}}
            projected = _excerpt(base, text_limit, '/receipt', omissions)
            projected['task'] = deepcopy(record['task'])
            if preview_feedback:
                for key in ('summary', 'error'):
                    projected['result'][key], omission = feedback_text_preview(
                        record['result'].get(key), job_id=row['job_id'],
                        evidence_path='/receipt/result/' + key, maximum=text_limit)
                    if omission:
                        omissions.append(omission)
            projected['operation'] = native(record['operation'], '/receipt/operation')
            for key in ('output', 'references'):
                projected['result'][key] = native(record['result'].get(key), '/receipt/result/' + key)
        elif receipt and 'directive' not in record:
            projected = _excerpt(record, text_limit, '/receipt', omissions)
            # Task identities and terminal facts cannot be clipped even when
            # the tool result includes large prose or nested tables.
            projected['task'] = deepcopy(record['task'])
            if preview_feedback:
                for key in ('summary', 'error'):
                    if key not in record['result']:
                        continue
                    projected['result'][key], omission = feedback_text_preview(
                        record['result'][key], job_id=row['job_id'],
                        evidence_path='/receipt/result/' + key, maximum=text_limit)
                    if omission:
                        omissions.append(omission)
        elif receipt:
            projected = {'directive': _excerpt(record['directive'], text_limit,
                                               '/receipt/directive', omissions),
                         'feedback': deepcopy(record['feedback'])}
            if 'identity' in record:
                projected['identity'] = deepcopy(record['identity'])
            if preview_feedback:
                for key in ('summary', 'error'):
                    if key not in projected['feedback']:
                        continue
                    projected['feedback'][key], omission = feedback_text_preview(
                        record['feedback'][key], job_id=row['job_id'],
                        evidence_path='/receipt/feedback/' + key, maximum=text_limit)
                    if omission:
                        omissions.append(omission)
        else:
            projected = _excerpt({k: v for k, v in record.items() if k not in excluded},
                                 text_limit, omissions=omissions)
        descriptor = {'raw_ref': path, 'sha256': result_id(record),
                      'coverage': 'excerpt' if omissions else 'complete',
                      'authority': 'model_interpretation' if interpretation else 'untrusted_evidence'}
        if ref:
            descriptor['source_ref'] = ref
            refs.append(ref)
        if omissions:
            descriptor['omitted_text'] = omissions
        if excluded:
            descriptor['coverage'] = 'fields_only'
            descriptor['separately_packaged_fields'] = list(excluded)
        # Complete stream records are located by supplied_indices and their
        # original identity. Avoid repeating a per-record digest unless a view
        # actually differs from the raw record (or is the current receipt).
        if omissions or receipt:
            manifest['records'].append(descriptor)
        return projected

    current = raw.get('research_context', {}).get('decision_id')
    current = current or raw.get('grouping_context', {}).get('receipt_decision_id')

    def stream(records, path, maximum, identity=None, preferred=None, duplicates=()):
        # Prefer the current receipt and newest records; retain original order in
        # the view. Deduplicate only byte-equivalent records, never an ID alone.
        duplicate_keys = {(r['record_id'], result_id(r)) for r in duplicates}
        eligible = [i for i, r in enumerate(records)
                    if not (identity == 'record_id' and (r['record_id'], result_id(r)) in duplicate_keys)]
        important = [i for i in eligible if preferred and preferred(records[i])]
        chosen = list(dict.fromkeys([*reversed(important), *reversed(eligible)]))[:maximum]
        chosen.sort()
        manifest['streams'][path] = {'raw_ref': path, 'sha256': result_id(records),
            'total': len(records), 'supplied': len(chosen),
            'supplied_indices': chosen,
            'authority': 'model_interpretation' if path in ('/grouping_context/attempts', '/dispatch_attempts') else 'untrusted_evidence',
            'deduplicated': len(records) - len(eligible), 'omitted': len(eligible) - len(chosen)}
        supplied = []
        for index in chosen:
            record = records[index]
            ref = (('history:' if identity == 'record_id' else 'decision:') + record[identity]
                   if identity else None)
            supplied.append(project(record, path + '/' + str(index), ref,
                                    interpretation=path in ('/grouping_context/attempts', '/dispatch_attempts')))
            if identity == 'record_id' and result_id(supplied[-1]) == result_id(record):
                coverage[record['record_id']] = result_id(record)
        return supplied

    packet = {'receipt': project(raw['receipt'], '/receipt', source, receipt=True)}
    task = receipt_task(raw['receipt'])
    task_id = task.get('task_id') or task.get('directive_id')
    scope = raw.get('research_context', {})
    members = scope.get('member_decisions', [])
    packet['research_context'] = project(scope, '/research_context', excluded=('member_decisions',))
    if 'member_decisions' in scope:
        packet['research_context']['member_decisions'] = stream(members,
            '/research_context/member_decisions', count, 'decision_id',
            lambda r: r['decision_id'] == current)
    packet['evaluations'] = stream(raw['evaluations'], '/evaluations', min(count, 16),
                                  preferred=lambda r: r.get('directive_id') == task_id)
    packet['history'] = stream(raw['history'], '/history', min(count, 16), 'record_id',
                              lambda r: r.get('task_id') == task_id)
    if 'grouping_context' in raw:
        group = raw['grouping_context']
        packet['grouping_context'] = project(group, '/grouping_context',
            excluded=('decisions', 'observations', 'attempts', 'candidate_origins'))
        view = packet['grouping_context']
        view['decisions'] = stream(group.get('decisions', []), '/grouping_context/decisions',
                                  count, 'decision_id', lambda r: r['decision_id'] == current)
        # Compare with the original history, since the supplied view may contain
        # excerpts. The duplicate's content is supplied once at that history ref.
        supplied_history = {r['record_id'] for r in packet['history']}
        view['observations'] = stream(group.get('observations', []), '/grouping_context/observations',
            min(count, 16), 'record_id',
            lambda r: r.get('task_id') == task_id,
            [r for r in raw['history'] if r['record_id'] in supplied_history])
        attempt_path = '/dispatch_attempts' if 'dispatch_attempts' in row else '/grouping_context/attempts'
        view['attempts'] = stream(row.get('dispatch_attempts', group.get('attempts', [])),
                                  attempt_path, min(count, 8))
        # Origins are execution links, never a claim that artifact content was read.
        if 'candidate_origins' in group:
            ids = {r['decision_id'] for r in view['decisions']}
            origins = group['candidate_origins']
            view['candidate_origins'] = {k: v for k, v in origins.items() if v in ids}
            manifest['streams']['/grouping_context/candidate_origins'] = {
                'raw_ref': '/grouping_context/candidate_origins', 'sha256': result_id(origins),
                'total': len(origins), 'supplied': len(view['candidate_origins']),
                'omitted': len(origins) - len(view['candidate_origins'])}
    evidence = candidate_evidence(raw, manifest['streams']['/evaluations']['supplied_indices'])
    if evidence is not None:
        packet['candidate_evidence'] = evidence
        refs.extend(s['source_ref'] for s in evidence['evaluations'])
    packet['source_refs'] = list(dict.fromkeys(refs))
    packet['evidence_packaging'] = manifest
    return packet, coverage


