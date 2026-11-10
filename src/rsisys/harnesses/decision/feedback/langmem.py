"""Opt-in LangMem workflow, using upstream extraction, patches and storage.

The adapter owns feedback triggering, evidence validation, host accounting and
transactional delivery. It does not implement a memory consolidation algorithm.
"""
import asyncio
import fcntl
import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from importlib import metadata, resources
from inspect import signature
from pathlib import Path
from threading import RLock

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.store.base import BaseStore, PutOp, SearchOp
from langgraph.store.sqlite import SqliteStore
from langmem import create_memory_store_manager
from langsmith import tracing_context
from pydantic import BaseModel, Field, model_validator

from rsisys.adapters.langchain_model import BackendChatModel
from rsisys.harnesses.decision.feedback.evidence import (
    admitted_candidate_evidence,
    claim_schema,
    validate_claims,
)
from rsisys.harnesses.decision.feedback.knowledge import (
    WorkingMemoryPolicy,
    context_record,
)
from rsisys.harnesses.decision.feedback.model_gateway import (
    FeedbackModelGateway,
    NativeAdmissionAbort,
)
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.runtime.storage.files import json_bytes
from rsisys.runtime.storage.json_tools import result_id


class _AtomicSqliteStore(SqliteStore):
    """Keep upstream SQLite batches atomic on error as well as on success.

    Pinned 3.1.3's _cursor commits in finally even on an exception. Everything
    else, including SQL schema, query behavior and migrations, stays upstream.
    """
    @contextmanager
    def _cursor(self, *, transaction=True):
        if not self.is_setup:
            self.setup()
        with self.lock:
            if transaction:
                self.conn.execute('BEGIN')
            cursor = self.conn.cursor()
            try:
                yield cursor
            except BaseException:
                if transaction:
                    self.conn.execute('ROLLBACK')
                raise
            else:
                if transaction:
                    self.conn.execute('COMMIT')
            finally:
                cursor.close()


class _StagedStore(BaseStore):
    """Buffer upstream mutations, then commit them and the job receipt together.

    LangMem 0.0.30 does not inspect its write futures. Retain their errors here,
    so a failed background write cannot be reported as a completed feedback job.
    Read/search still use the original upstream store.
    """
    def __init__(self, store, namespace, schema, *, max_retrieval_bytes=None):
        self.store, self.namespace, self.schema = store, namespace, schema
        self.puts, self.errors, self.lock = [], [], RLock()
        self.max_retrieval_bytes = max_retrieval_bytes

    def batch(self, ops):
        result = []
        for op in ops:
            if not isinstance(op, PutOp):
                values = self.store.batch([op])
                if isinstance(op, SearchOp) and self.max_retrieval_bytes is not None:
                    selected, used = [], 0
                    for item in values[0]:
                        size = len(json_bytes(item.value))
                        if used + size <= self.max_retrieval_bytes:
                            selected.append(item)
                            used += size
                    values = [selected]
                result.extend(values)
                continue
            try:
                if op.namespace != self.namespace:
                    raise ValueError('Memory mutation outside configured namespace')
                if op.value is not None:
                    if op.value.get('kind') != self.schema.__name__:
                        raise ValueError('Unexpected memory schema')
                    content = self.schema.model_validate(op.value['content']).model_dump()
                    op = op._replace(value={**deepcopy(op.value), 'content': content})
                with self.lock:
                    self.puts.append(op)
                result.append(None)
            except Exception as exc:
                with self.lock:
                    self.errors.append(f'{type(exc).__name__}: {exc}')
                raise
        return result

    async def abatch(self, ops):
        return await asyncio.to_thread(self.batch, list(ops))


class _ExtractionAudit(BaseCallbackHandler):
    """Observe Trustcall's public extraction outputs, without redoing its patches.

    Trustcall 0.0.39 can return only its valid subset after validation exhaustion.
    Do not misreport discarded invalid calls as a successful no-change update.
    """
    raise_error = True

    def __init__(self):
        self.rejected = []

    def on_chain_end(self, outputs, **kwargs):
        if not isinstance(outputs, dict) or not {
                'messages', 'responses', 'response_metadata', 'attempts'}.issubset(outputs):
            return
        accepted = {item['id'] for item in outputs['response_metadata']}
        for message in outputs['messages']:
            for call in getattr(message, 'tool_calls', ()):
                if call['id'] not in accepted:
                    self.rejected.append(call['id'])


def _research_schema(refs, evidence, simulated_refs=frozenset()):
    # The store manager uses "Memory" for unchanged dictionary-valued ephemeral
    # records between native phases. Keep that upstream kind name, adding only
    # our evidence fields instead of introducing an incompatible second kind.
    class Memory(BaseModel):
        """One concise research memory; source evidence remains authoritative."""
        content: str = Field(min_length=1, max_length=3000)
        source_refs: list[str] = Field(min_length=1, max_length=32,
            description='Exact supplied source references supporting this memory.')
        evaluation_claims: list[dict] = Field(default_factory=list, max_length=16,
            description='Factual evaluation claims only: candidate_id, evaluation_ref, '
                        'protocol_ref, metrics (exact names and values), simulated. '
                        'Use [] for non-evaluation observations or hypotheses.',
            json_schema_extra=claim_schema(evidence))
        simulated: bool = Field(default=False,
            description='Derived by the host from cited receipts; not an inference about real experiments.')

        @model_validator(mode='after')
        def validate_evidence(self):
            if not set(self.source_refs).issubset(refs):
                raise ValueError('Memory cites an unsupplied evidence source. Copy exact source_refs '
                    'from the supplied feedback or existing memories; do not construct refs from job IDs. '
                    'Examples of admitted refs: ' + repr(sorted(refs)[:8]))
            validate_claims(self.model_dump(), evidence)
            self.simulated = bool(set(self.source_refs).intersection(simulated_refs))
            return self
    return Memory


class LangMemContextMemory:
    """Native processor and context provider, independent of the decision harness.

    This SQLite configuration provides bounded recency retrieval, not semantic
    search. Embedding providers need their own explicit host admission adapter.
    A namespace and database must belong to one host/journal; no hidden cloud LLM
    or embedding provider is selected. Actual requests are kept per feedback job.
    """
    requires_tool_calling = True

    def __init__(self, path, *, namespace, max_calls_per_job=6, max_steps=1,
                 query_limit=8, consolidate=True, additional_instructions=None,
                 parallel_tool_calls=None):
        self.path = Path(path).resolve()
        self.namespace = tuple(namespace)
        if (not self.namespace or any(not isinstance(x, str) or not x or '.' in x
                                      for x in self.namespace)):
            raise ValueError('Use a nonempty, explicitly scoped memory namespace')
        for value in (max_calls_per_job, max_steps, query_limit):
            if type(value) is not int or value < 1:
                raise ValueError('LangMem limits must be positive integers')
        if max_steps != 1:
            raise ValueError('Pinned LangMem uses single-step phases; multi-step reconciliation is unverified')
        self.max_calls_per_job, self.max_steps = max_calls_per_job, max_steps
        if type(consolidate) is not bool:
            raise TypeError('consolidate must be a boolean')
        self.consolidate = consolidate
        if parallel_tool_calls is not None and type(parallel_tool_calls) is not bool:
            raise TypeError('parallel_tool_calls must be a boolean or None')
        self.parallel_tool_calls = parallel_tool_calls
        self.query_limit = query_limit
        self.additional_instructions = (resources.files(__package__).joinpath(
            'prompts/langmem_research.txt').read_text() if additional_instructions is None
            else additional_instructions)
        self._jobs_ns = (*self.namespace, 'feedback-jobs')
        self._sources_ns = (*self.namespace, 'feedback-sources')
        # MemoryStoreManager searches a namespace prefix. Keep operational
        # receipts and source catalogs outside its memory subtree.
        self._memory_ns = (*self.namespace, 'memories')
        self.audit_dir = self.path.parent / (self.path.name + '.exchanges')

    @contextmanager
    def _store(self):
        # Configuration/preflight must not create a run or database before the
        # host has claimed its workspace. Connections initialize lazily.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _AtomicSqliteStore.from_conn_string(str(self.path)) as store:
            yield store

    def configuration(self):
        return {'provider': 'langmem', 'database': str(self.path),
            'namespace': list(self.namespace), 'max_calls_per_job': self.max_calls_per_job,
            'max_steps': self.max_steps, 'query_limit': self.query_limit,
            'consolidate': self.consolidate,
            **({'parallel_tool_calls': self.parallel_tool_calls} if self.parallel_tool_calls is not None else {}),
            'consolidation_input': 'current_feedback_and_retrieved_memories',
            'additional_instructions_digest': result_id(self.additional_instructions),
            'retrieval': 'recent',
            'native_admission': 'complete_request_with_source_backed_evidence_projection',
            'retrieval_budget': 'whole_memories_up_to_one_input_token_limit_in_bytes',
            'versions': {p: metadata.version(p) for p in (
                'langmem', 'trustcall', 'langgraph', 'langchain-core', 'langgraph-checkpoint-sqlite')}}

    @staticmethod
    def _scan(store, namespace):
        offset = 0
        while True:
            items = store.search(namespace, limit=100, offset=offset)
            yield from items
            if len(items) < 100:
                return
            offset += len(items)

    def recover(self, job):
        with self._store() as store:
            item = store.get(self._jobs_ns, job['job_id'])
        if item is None:
            return None
        if item.value['receipt_digest'] != job['receipt_digest'] or item.value['model'] != job['model']:
            raise ValueError('Committed memory job identity differs from captured feedback')
        if item.value.get('configuration') != self.configuration():
            raise ValueError('Committed memory job configuration differs from the active component')
        if item.value.get('input_digest') != result_id(job['input']):
            raise ValueError('Committed memory job evidence snapshot differs from the captured input')
        return deepcopy(item.value['result'])

    def process(self, job, backend, limits):
        # A process lock prevents two journals from planning concurrent updates
        # against the same memory store. It fails promptly rather than blocking
        # the decision owner while an unknown worker is active.
        gateway = None
        exchange_path = self.audit_dir / (job['job_id'] + '.json')
        try:
            with self.path.with_suffix(self.path.suffix + '.writer-lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if self.recover(job) is not None:
                    raise ValueError('Memory job already committed; use recovery instead of inference replay')
                gateway = FeedbackModelGateway(backend, ModelRef(**job['model']),
                    max_calls=limits['max_calls'], max_input_tokens=limits['max_input_tokens'],
                    path=exchange_path, request_candidates=self._request_candidates(job))
                with self._store() as store, tracing_context(enabled=False):
                    current = admitted_candidate_evidence(job)
                    catalog = [item.value for item in self._scan(store, self._sources_ns)]
                    sources = {'source_refs': job['input']['source_refs'],
                               'evaluations': (current or {}).get('evaluations', []),
                               'simulated': job['simulated']}
                    catalog.append(sources)
                    refs = {ref for c in catalog for ref in c['source_refs']}
                    evidence = {'evaluations': [item for c in catalog for item in c['evaluations']]}
                    simulated_refs = {ref for c in catalog if c.get('simulated', False)
                                      for ref in c['source_refs']}
                    schema = _research_schema(refs, evidence, simulated_refs)
                    staged = _StagedStore(store, self._memory_ns, schema,
                        max_retrieval_bytes=limits['max_input_tokens'])
                    extraction_audit = _ExtractionAudit()
                    chat = BackendChatModel(backend=gateway, model_ref=ModelRef(**job['model']),
                        max_output_tokens=limits['max_output_tokens'], cache=False,
                        parallel_tool_calls=self.parallel_tool_calls, abort_on_capacity=True)
                    upstream = signature(create_memory_store_manager).parameters['instructions'].default
                    manager = create_memory_store_manager(chat, schemas=[schema],
                        instructions=upstream + '\n\n' + self.additional_instructions,
                        enable_deletes=True, namespace=self._memory_ns, store=staged,
                        query_limit=self.query_limit,
                        phases=[{'enable_inserts': False, 'enable_deletes': True,
                            'include_messages': True,
                            'instructions': 'You are a memory manager. Deduplicate, consolidate, '
                                'and enrich these memories.\n\n' + self.additional_instructions}]
                               if self.consolidate else None)
                    changes = manager.invoke({'messages': [{'role': 'user',
                        'content': json_bytes(job['input']).decode()}], 'max_steps': self.max_steps},
                        # A Trustcall validation repair spans several graph nodes
                        # per inference. The gateway owns the finite call limit;
                        # a fixed 16-node graph cut it off before that allowance.
                        config={'recursion_limit': max(32, limits['max_calls'] * 8 + 16), 'max_concurrency': 1,
                                'callbacks': [extraction_audit]})
                    if extraction_audit.rejected:
                        raise ValueError('Native memory extraction discarded invalid tool calls after validation')
                    if staged.errors:
                        raise RuntimeError('Native memory writes failed: ' + '; '.join(staged.errors))
                    result = {'status': 'completed', 'error': None, 'proposals': [],
                        'attempt_updates': [], 'charged_calls': gateway.attempted,
                        'response': {'provider': 'langmem', 'changes': changes,
                            'deleted_keys': [op.key for op in staged.puts if op.value is None],
                            'exchanges_ref': str(gateway.path), 'authority': 'unverified_model_memory'},
                        'simulated': job['simulated'] or any(x.get('response', {}).get('simulated', False)
                                                           for x in gateway.exchanges)}
                    # This atomic upstream batch includes an idempotency receipt.
                    # If the owner journal crashes afterwards, recover the result
                    # without a second model call or a second memory mutation.
                    store.batch([*staged.puts,
                        PutOp(self._sources_ns, job['job_id'], sources),
                        PutOp(self._jobs_ns, job['job_id'], {'receipt_digest': job['receipt_digest'],
                              'model': job['model'], 'configuration': self.configuration(),
                              'input_digest': result_id(job['input']), 'result': result})])
                    return result
        except (Exception, NativeAdmissionAbort) as exc:
            return {'status': 'input_too_large' if isinstance(exc, NativeAdmissionAbort) else 'failed',
                'error': f'{type(exc).__name__}: {exc}',
                'charged_calls': gateway.attempted if gateway is not None else (
                    limits['max_calls'] if exchange_path.exists() else 0),
                'proposals': [], 'attempt_updates': [],
                'response': {'provider': 'langmem', 'exchanges_ref': str(gateway.path)
                              if gateway is not None else None}}

    @staticmethod
    def _request_candidates(job):
        """Shrink only the captured evidence embedded in complete native calls.

        Upstream messages/tools, retrieved memories and patch conversations keep
        their identity. Every candidate is counted after native assembly; excess
        retrieval/repair content that cannot fit ends this job without inference.
        The original capture and read paths remain in the journal.
        """
        from rsisys.harnesses.decision.feedback.evidence_packet import evidence_packet
        original = json_bytes(job['input']).decode()

        def candidates(request):
            yield request
            for count, text_limit in ((16, 256), (8, 192), (4, 128), (2, 96), (1, 64), (0, 0)):
                packet, _ = evidence_packet(job, count, text_limit, preview_feedback=True)
                replacement = json_bytes(packet).decode()
                if len(replacement) >= len(original):
                    continue
                messages = request.chat_messages()
                changed = False
                for message in messages:
                    content = message.get('content')
                    if isinstance(content, str) and original in content:
                        message['content'] = content.replace(original, replacement)
                        changed = True
                if changed:
                    yield replace(request, messages=messages)
        return candidates

    def record(self, job):
        result = context_record(job)
        result.update(provider='langmem', organization=deepcopy(job.get('response')))
        path = self.audit_dir / (job['job_id'] + '.json')
        result['model_exchanges'] = json.loads(path.read_text())['exchanges'] if path.exists() else []
        return result

    def memory(self, job):
        return None  # Use the native retrieved view; do not inject a second per-job memory.

    def working_memory(self, jobs, attempts, *, active_decision_ids=(), policy=None):
        policy = policy or WorkingMemoryPolicy()
        with self._store() as store:
            selected = store.search(self._memory_ns, limit=policy.recent + 1)
        result = {'schema_version': 1, 'provider': 'langmem',
            'authority': 'unverified_model_memory', 'training_ready': False, 'applied': False,
            'retrieval': 'recent',
            'memories': [], 'more_available': len(selected) > policy.recent,
            'omitted_from_selection': 0}
        for item in selected[:policy.recent]:
            value = {'memory_id': item.key, **deepcopy(item.value)}
            result['memories'].append(value)
            if len(json_bytes(result)) > policy.max_bytes:
                result['memories'].pop()
                result['omitted_from_selection'] += 1
        return result


def enable_langmem_feedback(loop, *, memory_path, namespace, memory_options=None,
                           harness_proposals=None, **options):
    """Explicit opt-in; no second legacy interpretation call for the memory task."""
    from rsisys.harnesses.decision.learning_harness import enable_feedback_learning
    organizer = LangMemContextMemory(memory_path, namespace=namespace, **(memory_options or {}))
    if harness_proposals is not None:
        from rsisys.harnesses.decision.feedback.pipeline import enable_feedback_pipeline
        return enable_feedback_pipeline(loop, context_memory=organizer, memory_processor=organizer,
                                        harness_proposals=harness_proposals, **options)
    return enable_feedback_learning(loop, context_memory=organizer, processor=organizer, **options)
