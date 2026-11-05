"""Optional, bounded feedback interpretation by the decision model.

Workers return proposals only. The owning thread journals receipts and publishes
memory; model output never authorizes training, edits, or candidate activation.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict, replace
from inspect import getattr_static
from pathlib import Path
from threading import RLock, get_ident
import json

from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.loops.contracts.receipts import ExecutionReceipt, receipt_task, receipt_facts
from rsisys.loops.contracts.feedback_context import ContextQuery
from rsisys.loops.contracts.feedback_modules import PendingFeedback, FeedbackFailurePolicy
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.json_tools import encoded, result_id, EvidencePathError, evidence_value
from rsisys.runtime.json_tools import load_object as _load_object
from .attempts import attempt_schema, validate_attempt_updates, apply_attempt_updates, attempt_view
from .evidence import candidate_evidence, claim_schema, validate_claims, admitted_candidate_evidence
from .knowledge import proposal_records, interpretation_page, SUBJECTS
from .components import JournalContextMemory, JournalTrajectoryArchive
from .context_memory import ContextMemoryBinding, feedback_context_event, validate_context_memory
from .archive import ArchiveBinding, captured_archive_event, validate_archive
from .scheduling import FifoFeedbackScheduler, validate_scheduler
from .stages import FeedbackStageBinding, component_configuration
from .legacy import LegacyFeedbackInterpreter
from .delivery import producer_delivery


LEGACY_KINDS = ('memory', 'training_data', 'harness_change', 'system_change')
KINDS = (*LEGACY_KINDS, 'policy_training')
FIELDS = ('kind', 'content', 'applicability', 'source_refs', 'validation_plan')
_EVIDENCE_PAGE_BYTES = 6000
_DEFAULT_COMPONENT = object()
_CANDIDATE_INSTRUCTIONS = (
    'candidate_evidence binds evaluations to full candidate IDs, separately from the current receipt. '
    'Every proposal/attempt update requires evaluation_claims: [] for hypotheses or observations without metric claims; '
    'otherwise enumerate every factual candidate metric claim with candidate_id, evaluation_ref, protocol_ref, '
    'metrics (exact names/values), simulated, and cite evaluation_ref in source_refs. '
    'Use completed supplied evaluations only. Never transfer metrics between candidates or protocols. '
    'Training completion, parents, adjacency and prior interpretations prove no candidate quality. '
    'not_observed_in_snapshot is snapshot-scoped absence; not_supplied is omitted evidence. '
    'Keep prose consistent with claims; structural validation does not verify prose or claim completeness. ')

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


def _evidence_packet(row, count, text_limit, *, preview_feedback=True):
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


class SerializedLearningBackend:
    """Share one backend, including its admission/logging, across both callers.

    Generation is serialized because arbitrary backend clients and their counters
    are not thread safe. Jobs are asynchronous to the experiment, not a promise of
    parallel model inference. The wrapped transport must have a finite timeout
    and permit calls from a worker thread (serialization cannot remove affinity).
    """
    def __init__(self, backend):
        self.backend, self._lock = backend, RLock()

    def __getattr__(self, name):
        # Domain policies may inspect the same admission counters/capabilities.
        # All generate/tokenize calls still pass through the serialized methods.
        return getattr(self.backend, name)

    @property
    def supports_response_schema(self):
        return getattr(self.backend, 'supports_response_schema', False) is True

    @property
    def token_count_mode(self):
        return (getattr(self.backend, 'token_count_mode', 'provider_tokenizer')
                if getattr_static(self.backend, 'count_tokens', None) is not None
                else 'utf8_conservative_estimate')

    def count_tokens(self, request):
        with self._lock:
            counter = (getattr(self.backend, 'count_tokens')
                       if getattr_static(self.backend, 'count_tokens', None) is not None else None)
            if callable(counter):
                tokens = counter(request)
            else:
                tokens = len(json_bytes(asdict(request)))
            if type(tokens) is not int or tokens < 0:
                raise ValueError('Feedback token counter must return a nonnegative integer')
            return tokens

    def generate(self, request):
        with self._lock:
            return self.backend.generate(request)


def _schema(refs, grouping_context=None, candidate_evidence=None, *, kinds=LEGACY_KINDS):
    properties = {
        'kind': {'enum': list(KINDS)},
        'subject': {'enum': list(SUBJECTS)},
        'content': {'type': 'string', 'minLength': 1, 'maxLength': 2400},
        'applicability': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'source_refs': {'type': 'array', 'minItems': 1, 'maxItems': 16,
                        'items': {'enum': list(refs)}},
        'validation_plan': {'type': 'string', 'minLength': 1, 'maxLength': 1200},
    }
    if candidate_evidence is not None:
        properties['evaluation_claims'] = claim_schema(candidate_evidence)
    branches = []
    for kind in kinds:
        fields = {**properties, 'kind': {'const': kind}}
        if kind == 'training_data':
            fields['training_example'] = {'type': 'object', 'properties': {
                'input': {'type': 'string', 'minLength': 1, 'maxLength': 4000},
                'target': {'type': 'string', 'minLength': 1, 'maxLength': 4000},
                'target_status': {'const': 'unverified'}},
                'required': ['input', 'target', 'target_status'], 'additionalProperties': False}
        branches.append({'type': 'object', 'properties': fields, 'required': list(fields),
                         'additionalProperties': False})
    fields = {'proposals': {'type': 'array', 'maxItems': 6, 'items': {'oneOf': branches}}}
    if grouping_context is not None:
        fields['attempt_updates'] = attempt_schema(grouping_context, refs, candidate_evidence)
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def validate_proposals(text, refs, *, candidate_evidence=None, maximum=6):
    payload = _load_object(text)
    if (set(payload) != {'proposals'} or not isinstance(payload['proposals'], list)
            or (maximum is not None and len(payload['proposals']) > maximum)):
        raise ValueError('Expected at most six learning proposals')
    for row in payload['proposals']:
        if not isinstance(row, dict) or row.get('kind') not in KINDS:
            raise ValueError('Invalid learning proposal fields or kind')
        fields = set(FIELDS) | ({'training_example'} if row['kind'] == 'training_data' else set())
        if 'subject' in row:
            fields.add('subject')
            if row['subject'] not in SUBJECTS:
                raise ValueError('Invalid feedback proposal subject')
        if candidate_evidence is not None:
            fields.add('evaluation_claims')
        if set(row) != fields:
            raise ValueError('Invalid learning proposal fields')
        if row['kind'] == 'training_data':
            sample = row['training_example']
            if (not isinstance(sample, dict) or set(sample) != {'input', 'target', 'target_status'}
                    or sample['target_status'] != 'unverified'
                    or any(not isinstance(sample[k], str) or not sample[k].strip()
                           or len(sample[k]) > 4000 for k in ('input', 'target'))):
                raise ValueError('Training example must be a bounded, unverified input/target pair')
        for key, maximum in (('content', 2400), ('applicability', 1000), ('validation_plan', 1200)):
            if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > maximum:
                raise ValueError(f'Invalid proposal {key}')
        sources = row['source_refs']
        if (not isinstance(sources, list) or not 1 <= len(sources) <= 16
                or any(not isinstance(ref, str) or ref not in refs for ref in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError('Proposal cites unknown or duplicate evidence')
        if candidate_evidence is not None:
            validate_claims(row, candidate_evidence)
    return payload['proposals']


def validate_publication(row):
    """Recheck saved output against admitted raw evidence before exposing it.

    Structured attribution is checked; free text is still an unverified model
    interpretation. Legacy candidate-related outputs require the new contract.
    """
    evidence = admitted_candidate_evidence(row)
    validate_proposals(encoded({'proposals': row['proposals']}), row['input']['source_refs'],
                       candidate_evidence=evidence, maximum=None)
    validate_attempt_updates(row.get('attempt_updates', []), row['input'].get('grouping_context', {}),
                             row['input']['source_refs'], evidence)
    if evidence is not None:
        for revision in row.get('attempt_revisions', ()):
            validate_claims(revision, evidence)
    return evidence


def checked_processor_result(result, reserved):
    """Validate product/accounting metadata without granting journal ownership."""
    fields = {'status', 'error', 'charged_calls', 'proposals', 'attempt_updates',
              'response', 'simulated', 'grouping_error', 'products', 'output', 'outputs'}
    error = None
    if not isinstance(result, dict):
        error = 'Feedback processor must return a result object'
    elif type(result.get('charged_calls')) is not int or not 0 <= result['charged_calls'] <= reserved:
        error = 'Invalid native workflow accounting; full reservation retained'
    elif result.keys() - fields:
        error = 'Feedback processor cannot overwrite host journal fields'
    elif result.get('status') not in ('completed', 'failed', 'input_too_large',
                                      'budget_exhausted', 'cancelled', 'interrupted'):
        error = 'Feedback processor must return a terminal status'
    elif 'simulated' in result and type(result['simulated']) is not bool:
        error = 'Feedback processor simulated flag must be a boolean'
    try:
        json_bytes(result)
    except (ValueError, TypeError, OverflowError):
        error = 'Feedback processor result must be finite JSON data'
    if error:
        return {'status': 'failed', 'error': error, 'charged_calls': reserved,
                'proposals': [], 'attempt_updates': []}
    value = deepcopy(result)
    value.setdefault('proposals', [])
    value.setdefault('attempt_updates', [])
    if value['status'] != 'completed':
        value.update(proposals=[], attempt_updates=[])
    return value


class FeedbackLearning:
    """Single-writer journal plus one background inference worker.

    Reload preserves completed outputs and marks uncertain in-flight work
    interrupted; it never repeats that request automatically. Queued jobs have not
    called the model and may be dispatched after reopening. This journal is not a
    replacement for the Runner's own snapshot/recovery contract.
    """
    def __init__(self, backend: SerializedLearningBackend | None, *, path=None, max_calls=64,
                 max_pending=16, max_input_tokens=16000, max_output_tokens=4096,
                 max_input_bytes=64000, executor=None,
                 context_memory=_DEFAULT_COMPONENT, trajectory_archive=_DEFAULT_COMPONENT,
                 processor=None, assembly_configuration=None, scheduler=None,
                 max_concurrent_jobs=1, failure_policy=None, legacy_interpreter=None, admission=None):
        binding = FeedbackStageBinding(processor)
        if backend is None and (processor is None or binding.max_calls > 0):
            raise ValueError('Feedback inference requires a model backend')
        if backend is not None and not isinstance(backend, SerializedLearningBackend):
            raise TypeError('Use a shared SerializedLearningBackend for decision and learning calls')
        if context_memory is _DEFAULT_COMPONENT:
            context_memory = JournalContextMemory()
        if trajectory_archive is _DEFAULT_COMPONENT:
            trajectory_archive = JournalTrajectoryArchive()
        validate_context_memory(context_memory)
        validate_archive(trajectory_archive)
        self._context_memory, self._trajectory_archive = context_memory, trajectory_archive
        self._processor_binding = binding
        self.processor = processor
        self.legacy_interpreter = LegacyFeedbackInterpreter() if legacy_interpreter is None else legacy_interpreter
        if not callable(getattr(self.legacy_interpreter, 'request', None)):
            raise TypeError('Legacy feedback interpreter requires request(row, backend, limits)')
        processor_config = self._processor_binding.configuration() if processor is not None else None
        self.scheduler = FifoFeedbackScheduler() if scheduler is None else scheduler
        validate_scheduler(self.scheduler)
        if admission is not None and not callable(getattr(admission, 'admit', None)):
            raise TypeError('Feedback admission requires admit(event, pending)')
        self.admission = admission
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError('Host concurrency must be a positive integer')
        self.max_concurrent_jobs = max_concurrent_jobs
        self.failure_policy = FeedbackFailurePolicy() if failure_policy is None else failure_policy
        if not isinstance(self.failure_policy, FeedbackFailurePolicy):
            raise TypeError('failure_policy must be a FeedbackFailurePolicy')
        runtime_config = {'scheduler': component_configuration(self.scheduler),
                          'max_concurrent_jobs': max_concurrent_jobs,
                          'failure_policy': asdict(self.failure_policy),
                          'legacy_interpreter': component_configuration(self.legacy_interpreter)
                                                if processor is None else None}
        if admission is not None:
            runtime_config['admission'] = component_configuration(admission)
        self.backend, self.path = backend, Path(path) if path is not None else None
        self.limits = dict(max_calls=max_calls, max_pending=max_pending, max_input_tokens=max_input_tokens,
                           max_output_tokens=max_output_tokens, max_input_bytes=max_input_bytes)
        if any(type(v) is not int or v < 1 for v in self.limits.values()):
            raise ValueError('Learning limits must be positive integers')
        self._owner, self._closed = get_ident(), False
        self._executor = executor
        self._owns_executor = executor is None
        self._futures = {}
        self.state = {'version': 1, 'limits': self.limits, 'attempted': 0, 'jobs': {}}
        self.state['runtime'] = runtime_config
        if assembly_configuration is not None:
            json_bytes(assembly_configuration)
            self.state['assembly'] = deepcopy(assembly_configuration)
        if processor_config is not None:
            self.state['processor'] = processor_config
        if self.path is not None and self.path.exists():
            loaded = json.loads(self.path.read_text())
            if (loaded.get('version') != 1 or loaded.get('limits') != self.limits
                    or loaded.get('processor') != processor_config
                    or loaded.get('runtime') != runtime_config
                    or loaded.get('assembly') != assembly_configuration
                    or not isinstance(loaded.get('jobs'), dict)
                    or type(loaded.get('attempted')) is not int or loaded['attempted'] < 0):
                raise ValueError('Invalid learning journal or changed limits')
            self.state = loaded
            recovered = deepcopy(loaded)
            for row in sorted(recovered['jobs'].values(), key=lambda r: r.get('capture_index', -1)):
                for name in ('context_delivery', 'archive_delivery', 'publication_delivery'):
                    delivery = row.get(name, {})
                    if delivery.get('status') == 'running':
                        row[name] = {**delivery, 'status': 'interrupted',
                            'error': 'Delivery outcome unknown; no automatic replay'}
                if row['status'] == 'running':
                    try:
                        result = self._processor_binding.recover(row) if processor is not None else None
                    except Exception as exc:
                        row.update(status='interrupted', error=f'Recovery failed: {type(exc).__name__}: {exc}; '
                                   'reservation retained; no automatic retry')
                        continue
                    if result is None:
                        row.update(status='interrupted', error='Model request outcome unknown; no automatic retry')
                    else:
                        self._settle_processor(recovered, row, result)
                        self._finalize_job(recovered, row)
            if recovered != loaded:
                self._commit(recovered)

    def _assert_owner(self):
        if get_ident() != self._owner:
            raise RuntimeError('Only the owning thread may mutate the learning journal')

    def _commit(self, state):
        self._assert_owner()
        if self.path is not None:
            atomic_write(self.path, json_bytes(state))
        self.state = state

    def capture(self, directive, feedback, model, history=(), *, evaluations=(), research_context=None,
                grouping_context=None):
        receipt = ExecutionReceipt.from_feedback(directive, feedback)
        return self.capture_receipt(receipt, model, history, evaluations=evaluations,
            research_context=research_context, grouping_context=grouping_context)

    def capture_receipt(self, observation, model, history=(), *, evaluations=(), research_context=None,
                        grouping_context=None, archive_only=False):
        self._assert_owner()
        if self._closed:
            raise RuntimeError('Feedback learning is closed')
        if not isinstance(model, ModelRef) or not isinstance(observation, ExecutionReceipt):
            raise ValueError('Learning requires a decision model and execution receipt')
        if type(archive_only) is not bool:
            raise TypeError('archive_only must be a bool')
        observation.__post_init__()
        receipt = observation.payload()
        digest = result_id(receipt)
        prefix = {'operation': 'learning-operation-', 'lab': 'learning-lab-',
                  'exploration': 'learning-'}[observation.source]
        # Explicit Lab/run identities prevent different environments from
        # sharing a journal key. Legacy captures retain their historical key.
        identity = ([observation.lab_id, observation.run_id, observation.task_id]
                    if observation.lab_id or observation.run_id else observation.task_id)
        ident = prefix + result_id(identity)
        previous = self.state['jobs'].get(ident)
        if previous is not None:
            if previous['receipt_digest'] != digest:
                raise ValueError('Learning receipt changed for an existing task')
            if previous.get('capture_archive_only', previous.get('archive_only', False)) != archive_only:
                raise ValueError('Existing receipt processing mode cannot change')
            return ident
        producer_event, duplicate_of = producer_delivery(observation, self.state['jobs'].values(),
                                                         archive_only=archive_only)
        # Preserve raw evidence independently of the bounded model projection.
        scope = deepcopy(research_context or {})
        if observation.decision_id is not None:
            if scope.get('decision_id', observation.decision_id) != observation.decision_id:
                raise ValueError('Receipt decision identity differs from supplied research context')
            scope['decision_id'] = observation.decision_id
        if observation.issuing_model is not None:
            issuer = asdict(observation.issuing_model)
            if scope.get('issuing_decision_model') not in (None, issuer):
                raise ValueError('Receipt issuing model differs from supplied research context')
            scope['issuing_decision_model'] = issuer
        if observation.decision_id is not None or observation.issuing_model is not None:
            scope['learning_decision_model'] = asdict(model)
        members = scope.get('member_decisions', [])
        decisions = {item['decision_id'] for item in members}
        directives = {item['directive_id'] for item in members if item.get('directive_id')}
        directives.add(observation.task_id)
        if observation.decision_id:
            decisions.add(observation.decision_id)
        selected = [asdict(row) for row in history if row.task_id in directives
                    or row.record_id in decisions or row.details.get('decision_id') in decisions
                    or (isinstance(row.details.get('directive'), dict)
                        and row.details['directive'].get('directive_id') in directives)]
        packet = {'receipt': receipt, 'evaluations': [asdict(row) for row in evaluations],
                  'history': selected, 'research_context': scope}
        if grouping_context is not None:
            packet['grouping_context'] = deepcopy(grouping_context)
        row = {'job_id': ident, 'capture_index': len(self.state['jobs']),
               'receipt_digest': digest, 'model': asdict(model), 'raw_input': packet,
               'status': 'queued', 'proposals': [], 'error': None,
               'simulated': bool(observation.request.get('simulated', False) or observation.result['simulated']
                   or any(item.get('simulated', False) for item in packet['history'])
                   or any(item.get('simulated', False) for item in packet['evaluations'])
                   or any(item.get('simulated', False) for item in (grouping_context or {}).get('observations', []))
                   or any(item.get('proposal', {}).get('simulated', False)
                          for item in (grouping_context or {}).get('decisions', []))), 'response': None}
        if producer_event is not None:
            row.update(producer_event=producer_event, capture_archive_only=archive_only)
        if duplicate_of is not None:
            row['duplicate_of'] = duplicate_of
        admitted = self._bound_input(row)
        state = deepcopy(self.state)
        admission = self.admission
        if admission is not None:
            if not archive_only and duplicate_of is None:
                from rsisys.loops.contracts.feedback_modules import FeedbackAdmission
                pending = tuple(PendingFeedback(feedback_context_event(item), item['capture_index'])
                    for item in state['jobs'].values() if item['status'] == 'queued')
                choice = admission.admit(feedback_context_event(row), pending)
                eligible = {item.event.event_id for item in pending}
                if (not isinstance(choice, FeedbackAdmission) or type(choice.organize) is not bool
                        or not isinstance(choice.coalesce, tuple) or not isinstance(choice.reason, str)
                        or len(set(choice.coalesce)) != len(choice.coalesce)
                        or not set(choice.coalesce) <= eligible or (choice.coalesce and not choice.organize)):
                    raise ValueError('Invalid feedback admission or nonqueued coalescing target')
                row['admission'] = asdict(choice)
                if not choice.organize:
                    row.update(status='archived', organization_skipped=True, published=True)
                elif admitted:
                    for prior in choice.coalesce:
                        state['jobs'][prior].update(status='archived', organization_skipped=True,
                            coalesced_into=ident, published=True)
        if archive_only or duplicate_of is not None:
            row.update(status='archived', archive_only=True, published=True)
        elif row.get('organization_skipped'):
            pass
        elif not admitted:
            row.update(status='input_too_large', error='Indivisible learning evidence exceeds byte budget')
        elif sum(item['status'] in ('queued', 'running') for item in state['jobs'].values()) >= self.limits['max_pending']:
            row.update(status='queue_full', error='Learning queue limit reached; raw feedback remains available')
        state['jobs'][ident] = row
        self._commit(state)
        return ident

    def _bound_input(self, row, *, tokens=False):
        # Token admission includes the system instructions AND generated schema.
        # Profiles are deterministic, contain no generated summaries, and never
        # enlarge a budget. Small evidence remains verbatim.
        for count, text_limit, preview_feedback in (
                (32, 512, False), (32, 512, True), (16, 256, True), (8, 192, True),
                (4, 128, True), (2, 96, True), (1, 64, True), (0, 0, True)):
            row['input'], row['coverage'] = _evidence_packet(
                row, count, text_limit, preview_feedback=preview_feedback)
            if len(json_bytes(row['input'])) > self.limits['max_input_bytes']:
                continue
            if (not tokens or self.processor is not None
                    or self.backend.count_tokens(self._request(row)) <= self.limits['max_input_tokens']):
                return True
        return False

    def _request(self, row):
        if row.get('request') is not None:
            request = deepcopy(row['request'])
            request['model'] = ModelRef(**request['model'])
            return GenerationRequest(**request)
        return self.legacy_interpreter.request(row, self.backend, self.limits)

    def _work(self, row):
        if self.processor is not None:
            directory = (self.path.parent / (self.path.stem + '-exchanges') / row['job_id']
                         if self.path is not None else None)
            return self._processor_binding.process(row, self.backend,
                {**self.limits, 'max_calls': row['reserved_calls'],
                 'exchange_directory': str(directory) if directory is not None else None},
                path=directory / 'product.json' if directory is not None else None)
        # No stores, policy mutation, or memory writes in this thread.
        request = self._request(row)
        if self.backend.count_tokens(request) > self.limits['max_input_tokens']:
            return {'status': 'input_too_large', 'error': 'Complete request exceeds token budget', 'response': None}
        try:
            response = self.backend.generate(request)
        except Exception as exc:
            return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'response': None}
        receipt = asdict(response)
        try:
            if response.model != request.model:
                raise ValueError('Learning model identity mismatch')
            if response.finish_reason == 'length':
                raise ValueError('Feedback interpretation output was truncated')
            payload = _load_object(response.text)
            grouping = row['input'].get('grouping_context')
            updates = payload.pop('attempt_updates', []) if grouping is not None else []
            evidence = admitted_candidate_evidence(row)
            proposals = validate_proposals(json.dumps(payload), row['input']['source_refs'],
                                           candidate_evidence=evidence)
            # Grouping failure is isolated from valid feedback and execution.
            grouping_error = None
            try:
                updates = validate_attempt_updates(updates, grouping or {}, row['input']['source_refs'], evidence)
            except (ValueError, TypeError, KeyError) as exc:
                updates, grouping_error = [], str(exc)
        except (ValueError, TypeError, KeyError) as exc:
            return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'response': receipt}
        return {'status': 'completed', 'error': None, 'proposals': proposals,
                'attempt_updates': updates, 'grouping_error': grouping_error,
                'response': receipt, 'simulated': row['simulated'] or response.simulated}

    @staticmethod
    def _settle_processor(state, row, result):
        reserved = row['reserved_calls']
        result = checked_processor_result(result, reserved)
        state['attempted'] -= reserved - result['charged_calls']
        simulated = bool(row['simulated'] or result.get('simulated', False))
        row.update(result)
        row['simulated'] = simulated

    @staticmethod
    def _finalize_job(state, row):
        """Identical validation/revision transition for live completion and recovery."""
        if row['status'] != 'completed':
            return
        try:
            validate_publication(row)
        except (ValueError, TypeError, KeyError) as exc:
            row.update(status='failed', proposals=[], attempt_updates=[], attempt_revisions=[],
                       error='Evidence validation failed: ' + str(exc))
        else:
            try:
                current = attempt_view(r for ident, r in state['jobs'].items() if ident != row['job_id'])
                row['attempt_revisions'] = apply_attempt_updates(row, current['items'])
            except ValueError as exc:
                row.update(attempt_revisions=[], grouping_error=str(exc))

    def ordered_jobs(self):
        """Use persisted capture order; legacy rows retain their stored order.

        Pre-index journals cannot reconstruct capture chronology after JSON key
        sorting. Do not infer it from hashes or replay an uncertain model call.
        """
        rows = list(self.state['jobs'].values())
        return [row for _, row in sorted(enumerate(rows),
            key=lambda pair: (pair[1].get('capture_index', -1), pair[0]))]

    def poll(self, *, dispatch=True):
        """Commit ready results without waiting, then admit at most one request."""
        self._assert_owner()
        for ident, future in list(self._futures.items()):
            if not future.done():
                continue
            try:
                result = future.result()
            except Exception as exc:
                result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            state = deepcopy(self.state)
            row = state['jobs'][ident]
            if self.processor is not None:
                self._settle_processor(state, row, result)
            else:
                row.update(result)
            self._finalize_job(state, row)
            self._commit(state)
            del self._futures[ident]
        if not dispatch or self._closed or len(self._futures) >= self.max_concurrent_jobs:
            return
        pending = [r for r in self.ordered_jobs() if r['status'] == 'queued']
        if not pending:
            return
        try:
            selected = self.scheduler.select(tuple(PendingFeedback(feedback_context_event(row),
                row.get('capture_index', -1)) for row in pending), tuple(self._futures))
            if selected is None:
                return
            if not isinstance(selected, str) or selected not in {r['job_id'] for r in pending}:
                raise ValueError('Scheduler selected an event that is not pending')
        except Exception as exc:
            state = deepcopy(self.state)
            state['scheduler_error'] = f'{type(exc).__name__}: {exc}'
            self._commit(state)
            return
        queued = next(row for row in pending if row['job_id'] == selected)
        state = deepcopy(self.state)
        state.pop('scheduler_error', None)
        row = state['jobs'][queued['job_id']]
        # Refresh only the interpretation catalog at dispatch. Execution evidence
        # remains the immutable capture, and model provenance hashes the actual input.
        # Legacy queued journals contain only input; adopt it as the raw snapshot
        # before projecting. Completed historical requests are never rewritten.
        row.setdefault('raw_input', deepcopy(row['input']))
        grouping = row['raw_input'].get('grouping_context')
        if grouping is not None:
            supplied = {r['decision_id'] for r in grouping['decisions']}
            catalog = self.attempts()['items']
            row['dispatch_attempts'] = [r for r in catalog
                                   if set(r['member_decision_ids']).intersection(supplied) or r in catalog[-8:]]
        if (state['attempted'] >= self.limits['max_calls']
                and (self.processor is None or self._processor_binding.max_calls > 0)):
            row.update(status='budget_exhausted', error='Learning request budget exhausted')
            self._commit(state)
            return
        try:
            admitted = self._bound_input(row, tokens=True)
        except Exception as exc:
            # Tokenizer/transport failures are learning failures, not exceptions
            # in the execution owner's feedback delivery path.
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            self._commit(state)
            return
        if not admitted:
            row.update(status='input_too_large', error='Indivisible learning evidence exceeds request budget')
            self._commit(state)
            return
        # Persist the exact feedback prompt/schema before inference. Recovery
        # reads this exchange rather than reconstructing it from newer code.
        if self.processor is None:
            row['request'] = asdict(self._request(row))
            state['attempted'] += 1  # Conservative reservation includes rejected requests.
        else:
            row['reserved_calls'] = min(self._processor_binding.max_calls,
                self.limits['max_calls'] - state['attempted'])
            row['request'] = {'processor': deepcopy(state['processor']),
                              'input': deepcopy(row['input']),
                              'actual_exchanges_read_path': getattr(self.processor,
                                  'actual_exchanges_read_path', '/context_memory/model_exchanges')}
            state['attempted'] += row['reserved_calls']
        row['status'] = 'running'
        self._commit(state)
        try:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rsi-feedback')
            self._futures[row['job_id']] = self._executor.submit(self._work, deepcopy(row))
        except Exception as exc:
            state = deepcopy(self.state)
            state['jobs'][row['job_id']].update(status='failed', error=f'Submission failed: {exc}')
            self._commit(state)

    def attempts(self, decision_ids=()):
        return attempt_view(self.state['jobs'].values(), decision_ids)

    @property
    def context_memory(self):
        return self._context_memory

    @property
    def trajectory_archive(self):
        return self._trajectory_archive

    @property
    def context_binding(self):
        return ContextMemoryBinding(self.context_memory)

    @property
    def archive_binding(self):
        return ArchiveBinding(self.trajectory_archive)

    def notify_archive(self):
        """Persist intent before external writes; no replay of unknown effects."""
        self._assert_owner()
        binding = self.archive_binding
        if self._closed or not binding.requires_delivery:
            return
        for row in self.ordered_jobs():
            if row.get('archive_delivery'):
                continue
            ident = row['job_id']
            event = captured_archive_event(row)
            delivery = {'status': 'running', 'event_digest': result_id(asdict(event))}
            state = deepcopy(self.state)
            state['jobs'][ident]['archive_delivery'] = delivery
            self._commit(state)
            try:
                reference = binding.write(event)
            except Exception as exc:
                delivery = {**delivery, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                delivery = {**delivery, 'status': 'delivered', 'reference': reference}
            state = deepcopy(self.state)
            state['jobs'][ident]['archive_delivery'] = delivery
            self._commit(state)

    def notify_context(self):
        """Offer raw feedback on a later tick, independently of interpretation.

        Notification does not acknowledge/re-execute a Lab task. Persist intent
        before a provider hook; successful, failed or uncertain side effects are
        not automatically repeated. Providers can deduplicate by event_id.
        """
        self._assert_owner()
        if self._closed:
            return
        observer = self.context_binding.observer
        if observer is None:
            return
        for captured in self.ordered_jobs():
            if captured.get('archive_only') or captured.get('context_delivery'):
                continue
            ident = captured['job_id']
            event = feedback_context_event(captured)
            state = deepcopy(self.state)
            state['jobs'][ident]['context_delivery'] = {'status': 'running'}
            self._commit(state)
            try:
                observer(event)
            except Exception as exc:
                delivery = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                delivery = {'status': 'delivered'}
            state = deepcopy(self.state)
            state['jobs'][ident]['context_delivery'] = delivery
            self._commit(state)

    def context(self, query=None, *, active_decision_ids=(), policy=None):
        """Return the current text view; no journal types reach a new provider."""
        query = ContextQuery() if query is None else query
        if not isinstance(query, ContextQuery):
            raise TypeError('Context query must be a ContextQuery')
        binding = self.context_binding
        jobs = self.ordered_jobs()
        if not binding.legacy and query.latest_feedback is None:
            latest = next((row for row in reversed(jobs) if not row.get('archive_only')), None)
            if latest is not None:
                query = replace(query, latest_feedback=feedback_context_event(latest))
        try:
            value = (binding.context(query) if not binding.legacy else
                binding.context(query, jobs=jobs, attempts=self.attempts()['items'],
                    active_decision_ids=active_decision_ids, policy=policy))
        except Exception as exc:
            previous = self.state.get('context_cache', {})
            reuse = (self.failure_policy.context == 'last_valid' and
                     previous.get('objective') == query.objective and 'text' in previous)
            state = deepcopy(self.state)
            state['context_error'] = {'error': f'{type(exc).__name__}: {exc}', 'used_previous': reuse,
                                      'objective': query.objective}
            self._commit(state)
            if reuse:
                return previous['text']
            raise
        if self.failure_policy.context == 'last_valid' or 'context_error' in self.state:
            state = deepcopy(self.state)
            state.pop('context_error', None)
            if self.failure_policy.context == 'last_valid':
                state['context_cache'] = {'objective': query.objective, 'text': value}
            self._commit(state)
        return value

    def context_record(self, job_id):
        return self.context_binding.record(self.state['jobs'][job_id])

    def working_memory(self, *, active_decision_ids=(), policy=None):
        """Compatibility inspection; harnesses consume context() for new views."""
        binding = self.context_binding
        if binding.legacy:
            return binding.working_memory(self.ordered_jobs(), self.attempts()['items'],
                active_decision_ids=active_decision_ids, policy=policy)
        return {'context': self.context()}

    def trajectories(self, *, research_attempt_id=None):
        if self.trajectory_archive is None:
            return []
        return self._trajectory_records(self.ordered_jobs(), research_attempt_id=research_attempt_id)

    def extract_trajectory(self, job_id):
        if self.trajectory_archive is None:
            raise EvidencePathError('Trajectory archival is disabled')
        return self.archive_binding.extract(self.state['jobs'][job_id])

    def _trajectory_records(self, jobs, *, research_attempt_id=None):
        return self.archive_binding.records(jobs, research_attempt_id=research_attempt_id)

    def trajectory_snapshot(self, job_id):
        if self.trajectory_archive is None:
            raise EvidencePathError('Trajectory archival is disabled')
        return self.archive_binding.snapshot(self.state['jobs'][job_id])

    def read(self, job_id=None, *, offset=0, limit=8, research_attempt_id=None, evidence_path=None,
             view='jobs', target=None, expected_digest=None):
        """Read proposals, or a bounded JSON-text page of original evidence.

        evidence_path is a JSON pointer into raw_input (empty for the root).
        /dispatch_attempts addresses the separately refreshed catalog. Evidence
        offsets count characters, with at most 128 characters per limit unit.
        Pages also fit 6000 JSON-encoded UTF-8 bytes including metadata (except
        irreducible metadata plus one character). Assemble all pages before
        parsing their JSON. Reads never expand model coverage.
        """
        self._assert_owner()
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError('Invalid learning page')
        if view not in ('jobs', 'attempts', 'proposals', 'trajectories'):
            raise ValueError('Unknown research feedback view')
        if expected_digest is not None and (not isinstance(expected_digest, str) or len(expected_digest) != 64):
            raise ValueError('expected_digest must be a SHA-256 digest')
        if view != 'jobs':
            if job_id is not None or evidence_path is not None:
                raise ValueError('Knowledge views cannot be combined with a job evidence read')
            if research_attempt_id is not None and (not isinstance(research_attempt_id, str)
                    or not research_attempt_id.strip() or len(research_attempt_id) > 200):
                raise ValueError('Invalid research attempt ID')
            if view == 'trajectories':
                if target is not None:
                    raise ValueError('target applies only to proposal views')
                value = self.trajectories(research_attempt_id=research_attempt_id)
            elif view == 'attempts':
                if target is not None:
                    raise ValueError('target applies only to proposal views')
                value = self.attempts()['items']
                if research_attempt_id is not None:
                    value = [a for a in value if a['attempt_id'] == research_attempt_id]
            else:
                value = proposal_records(self.ordered_jobs(), target=target,
                                         research_attempt_id=research_attempt_id)
            metadata = {'view': view, 'total': len(value),
                        'target': target, 'research_attempt_id': research_attempt_id}
            if view == 'trajectories':
                metadata['enabled'] = self.trajectory_archive is not None
            return interpretation_page(value, offset=offset, limit=limit,
                expected_digest=expected_digest, metadata=metadata,
                authority='archived_public_trajectory' if view == 'trajectories' else 'unverified_interpretations')
        if target is not None:
            raise ValueError('target applies only to proposal views')
        if research_attempt_id is not None and (not isinstance(research_attempt_id, str)
                or not research_attempt_id.strip() or len(research_attempt_id) > 200 or job_id is not None):
            raise ValueError('An attempt filter requires a bounded ID and a job listing')
        if evidence_path is not None:
            if (job_id is None or not isinstance(evidence_path, str) or len(evidence_path) > 1000
                    or (evidence_path and not evidence_path.startswith('/'))):
                raise ValueError('Evidence reading requires a job ID and a JSON pointer of at most 1000 characters')
            if job_id not in self.state['jobs']:
                raise EvidencePathError('Evidence job ID is not recorded; list feedback jobs before reading evidence')
            row = self.state['jobs'][job_id]
            if evidence_path.split('/')[1:2] in (['context_memory'], ['trajectory'], ['trajectory_segment'], ['trajectory_snapshot']):
                namespace = evidence_path.split('/')[1]
                if namespace == 'context_memory':
                    value = self.context_record(job_id)
                elif namespace == 'trajectory_segment':
                    value = self.extract_trajectory(job_id)
                elif namespace == 'trajectory_snapshot':
                    value = self.trajectory_snapshot(job_id)
                else:
                    if self.trajectory_archive is None:
                        raise EvidencePathError('Trajectory archival is disabled')
                    records = self._trajectory_records([row])
                    if len(records) != 1:
                        raise EvidencePathError('No unique trajectory segment for this job')
                    value = records[0]
                value = evidence_value({namespace: value}, evidence_path)
                return interpretation_page(value, offset=offset, limit=limit,
                    expected_digest=expected_digest, authority=(
                        'organized_context_memory' if namespace == 'context_memory' else 'archived_public_trajectory'),
                    metadata={'job_id': job_id, 'evidence_path': evidence_path, 'simulated': row['simulated']})
            if evidence_path.split('/')[1:2] in (['proposals'], ['attempt_revisions'], ['outputs'], ['output']):
                # Interpretation output has its own namespace. It is never
                # presented as captured execution evidence or admitted coverage.
                value = evidence_value(row, evidence_path)
                return interpretation_page(value, offset=offset, limit=limit,
                    expected_digest=expected_digest, metadata={'job_id': job_id,
                        'evidence_path': evidence_path, 'simulated': row['simulated']})
            value = row.get('raw_input', row['input'])
            root = '/raw_input' if 'raw_input' in row else '/input'
            pointer = evidence_path
            if pointer == '/dispatch_attempts' or pointer.startswith('/dispatch_attempts/'):
                value = row['dispatch_attempts']
                root = '/dispatch_attempts'
                pointer = pointer[len('/dispatch_attempts'):]
            for namespace, key in (('/feedback_request', 'request'), ('/feedback_response', 'response')):
                if pointer == namespace or pointer.startswith(namespace + '/'):
                    if row.get(key) is None:
                        raise EvidencePathError('Feedback exchange was not recorded for this job')
                    value, root, pointer = row[key], '/' + key, pointer[len(namespace):]
                    break
            value = evidence_value(value, pointer)
            if expected_digest is not None and expected_digest != result_id(value):
                raise EvidencePathError('Feedback evidence changed; restart pagination')
            text = encoded(value)
            interpretation = evidence_path.startswith(('/dispatch_attempts', '/grouping_context/attempts'))
            metadata = {'job_id': job_id, 'evidence_path': evidence_path,
                    'raw_ref': 'journal:' + job_id + root + pointer, 'sha256': result_id(value),
                    'authority': 'untrusted_evidence',
                    'content_authority': 'model_interpretation' if interpretation else 'untrusted_evidence',
                    'encoding': 'json', 'offset': offset, 'total_characters': len(text),
                    'simulated': row['simulated'], 'training_ready': False, 'applied': False}
            chunk = text[offset:offset + limit * 128]

            def page(size):
                end = offset + size
                return {**metadata, 'text': chunk[:size],
                        'next_offset': end if end < len(text) else None}

            complete = page(len(chunk))
            if len(json_bytes(complete)) <= _EVIDENCE_PAGE_BYTES:
                return complete
            # Price the outer JSON string too: quotes, backslashes and control
            # escapes can grow again when JSON evidence is nested inside a page.
            # Keep at least one source character, never a byte fragment or a
            # non-advancing cursor, even if metadata alone exhausts the soft cap.
            lo, hi = min(1, len(chunk)), len(chunk)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(json_bytes(page(mid))) <= _EVIDENCE_PAGE_BYTES:
                    lo = mid
                else:
                    hi = mid - 1
            return page(lo)
        if job_id is not None:
            if expected_digest is not None:
                raise ValueError('expected_digest requires an evidence or knowledge page')
            row = self.state['jobs'][job_id]
            return deepcopy({'job_id': job_id, 'status': row['status'], 'model': row['model'],
                'receipt_digest': row['receipt_digest'], 'proposals': row['proposals'],
                'products': row.get('products', {}),
                'research_context': row['input'].get('research_context', {}),
                'attempt_revisions': deepcopy(row.get('attempt_revisions', [])),
                'grouping_error': row.get('grouping_error'),
                'publication_error': row.get('publication_error'),
                'source_refs': row['input']['source_refs'], 'error': row['error'],
                'evidence_packaging': row['input'].get('evidence_packaging'),
                'context_memory': {'enabled': True,
                                  'read_path': {'operation': 'feedback_learning_read', 'job_id': job_id,
                                                'evidence_path': '/context_memory'}},
                'trajectory': ({'enabled': True,
                               'read_path': {'operation': 'feedback_learning_read', 'job_id': job_id,
                                             'evidence_path': '/trajectory'}}
                               if self.trajectory_archive is not None else {'enabled': False}),
                'simulated': row['simulated'], 'authority': 'unverified_proposals',
                'training_ready': False, 'applied': False})
        rows = self.ordered_jobs()
        if expected_digest is not None:
            raise ValueError('expected_digest requires an evidence or knowledge page')
        if research_attempt_id is not None:
            rows = [row for row in rows if row['input'].get('research_context', {}).get('research_attempt_id') == research_attempt_id
                    or any(r['attempt_id'] == research_attempt_id for r in row.get('attempt_revisions', []))]
        return {'attempted': self.state['attempted'], 'limit': self.limits['max_calls'], 'total': len(rows),
                'items': [{'job_id': r['job_id'], 'status': r['status'],
                           'products': deepcopy(r.get('products', {})),
                           **{key: r['input'].get('research_context', {}).get(key)
                              for key in ('decision_id', 'research_attempt_id', 'research_attempt_title')},
                           'interpreted_attempt_ids': [a['attempt_id'] for a in r.get('attempt_revisions', [])],
                           'grouping_error': r.get('grouping_error'),
                           'kinds': [p['kind'] for p in r['proposals']]} for r in rows[offset:offset+limit]],
                'next_offset': offset+limit if offset+limit < len(rows) else None}

    def close(self, *, timeout=30):
        """Stop admissions; preserve uncertain calls, never claim cancellation.

        Python cannot kill a running backend call. A timed-out close reports failure;
        the finite transport timeout must ultimately release the worker.
        """
        self._assert_owner()
        self._closed = True
        self.poll(dispatch=False)
        if self._futures:
            _, pending = wait(tuple(self._futures.values()), timeout=timeout)
            self.poll(dispatch=False)
            if pending:
                raise TimeoutError('Feedback inference still running; transport cancellation not confirmed')
        state = deepcopy(self.state)
        for row in state['jobs'].values():
            if row['status'] == 'queued':
                row.update(status='cancelled', error='Closed before model submission')
        self._commit(state)
        if self._executor is not None and self._owns_executor:
            self._executor.shutdown(wait=True)
