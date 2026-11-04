"""Real upstream LangMem/Trustcall + SQLite, scripted inference; no training."""
from copy import deepcopy
from dataclasses import asdict, replace
from importlib.util import find_spec
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from rsisys.loops.contracts.model_service import GenerationResponse, TokenUsage
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.json_tools import result_id
from tests.test_feedback_learning_integration import ImmediateExecutor
from tests.test_feedback_required_outputs import MODEL, observation

HAS_LANGMEM = find_spec('langmem') is not None
if HAS_LANGMEM:
    from langgraph.store.base import PutOp
    from rsisys.loops.decision_making.feedback.langmem import (
        LangMemContextMemory, _AtomicSqliteStore, _StagedStore, _research_schema)


class NativeFixture:
    """Emit standard native calls; upstream code applies all JSON patches."""
    supports_tool_calling = True

    def __init__(self):
        self.requests = []
        self.value = None
        self.invalid_first = False
        self.wrong_identity = False
        self.huge_tokens = False
        self.emit_second = False

    def count_tokens(self, request):
        # Include native tool schemas and complete messages in fixture admission.
        return 100000 if self.huge_tokens else len(json_bytes(asdict(request))) // 4

    def generate(self, request):
        self.requests.append(request)
        names = [t['function']['name'] for t in request.tools]
        text = json.dumps(request.chat_messages())
        if 'PatchFunctionErrors' in names:
            previous = [m for m in request.chat_messages() if m['role'] == 'assistant'][-1]
            ident = previous['tool_calls'][0]['id']
            name, args = 'PatchFunctionErrors', {'json_doc_id': ident,
                'planned_edits': 'Replace the invalid source reference with the supplied reference.',
                'patches': [{'op': 'replace', 'path': '/source_refs',
                             'value': self.value['source_refs']}]}
        elif 'PatchDoc' in names:
            ident = re.findall(r'<instance id=([^ ]+) schema_type=', text)[0]
            name, args = 'PatchDoc', {'json_doc_id': ident, 'planned_edits': 'Update the existing memory.',
                'patches': [{'op': 'replace', 'path': '/' + k, 'value': v}
                            for k, v in self.value.items()]}
        else:
            name, args = 'Memory', deepcopy(self.value)
            if self.invalid_first:
                self.invalid_first = False
                args['source_refs'] = ['unsupplied']
        model = replace(request.model, revision='wrong') if self.wrong_identity else request.model
        call = {'id': 'fixture-call-' + str(len(self.requests)), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}
        calls = [call]
        if self.emit_second and name == 'Memory':
            calls.append({'id': 'second-' + call['id'], 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps({**args, 'content': 'A different observation'})}})
        return GenerationResponse('', model, True, TokenUsage(100, 20),
            provider_model=request.model.model_id, revision_verified=True,
            tool_calls=calls, finish_reason='tool_calls')


@unittest.skipUnless(HAS_LANGMEM, 'Install the optional langmem extra')
class LangMemFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = NativeFixture()

    def learning(self, **options):
        self.provider = LangMemContextMemory(self.root / 'memory.sqlite', namespace=('lab', 'run'))
        learning = FeedbackLearning(SerializedLearningBackend(self.backend),
            path=self.root / 'feedback.json', processor=self.provider, context_memory=self.provider,
            executor=ImmediateExecutor(), **options)
        self.addCleanup(lambda: None if learning._closed else learning.close())
        return learning

    def capture(self, learning, ident, *, content='One corrected observation', memory_content=None, model=MODEL):
        receipt = observation(ident, summary=content)
        self.backend.value = {'content': content if memory_content is None else memory_content,
            'source_refs': ['feedback:' + result_id(receipt.payload())], 'evaluation_claims': []}
        job_id = learning.capture_receipt(receipt, model)
        learning.poll()
        learning.poll()
        return job_id

    def test_native_feedback_replaces_legacy_prompt_and_reconciles_in_sqlite(self):
        learning = self.learning()
        first = self.capture(learning, 'first', content='Old observation')
        memory_id = learning.working_memory()['memories'][0]['memory_id']
        second = self.capture(learning, 'second', content='Corrected observation')
        self.assertEqual([r['status'] for r in learning.ordered_jobs()], ['completed', 'completed'])
        memories = learning.working_memory()['memories']
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]['memory_id'], memory_id)
        self.assertEqual(memories[0]['content']['content'], 'Corrected observation')
        self.assertEqual(learning.state['attempted'], len(self.backend.requests))
        self.assertEqual(len(self.backend.requests), 4)  # Extraction + native consolidation each time.
        self.assertTrue(all(r.messages is not None and r.tools for r in self.backend.requests))
        self.assertFalse(any('Return {"proposals"' in str(r.messages) for r in self.backend.requests))
        self.assertEqual(learning.state['jobs'][second]['proposals'], [])
        self.assertIsNone(self.provider.memory(learning.state['jobs'][first]))
        exchanges = self.provider.record(learning.state['jobs'][second])['model_exchanges']
        self.assertEqual(len(exchanges), 2)
        self.assertEqual(exchanges[0]['request']['model'], asdict(MODEL))
        before = len(self.backend.requests)
        self.assertEqual(learning.capture_receipt(observation('second', summary='Corrected observation'), MODEL), second)
        learning.poll()
        self.assertEqual(len(self.backend.requests), before)

    def test_upstream_validation_retry_is_audited_and_charged(self):
        learning = self.learning()
        self.backend.invalid_first = True
        ident = self.capture(learning, 'retry')
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed', row['error'])
        exchanges = self.provider.record(row)['model_exchanges']
        self.assertEqual(len(exchanges), 3)
        self.assertEqual(learning.state['attempted'], 3)
        self.assertIn('PatchFunctionErrors', [t['function']['name'] for t in self.backend.requests[1].tools])

    def test_native_phase_retains_an_unchanged_ephemeral_memory_with_evidence_fields(self):
        learning = self.learning()
        self.backend.emit_second = True
        ident = self.capture(learning, 'two-observations')
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed', row['error'])
        memories = learning.working_memory()['memories']
        self.assertEqual(len(memories), 2)
        self.assertEqual({m['kind'] for m in memories}, {'Memory'})
        self.assertTrue(all(m['content']['simulated'] for m in memories))
        self.assertEqual({m['content']['content'] for m in memories},
                         {'One corrected observation', 'A different observation'})

    def test_default_harness_consumes_native_view_and_tracks_active_weights(self):
        from rsisys.config import build_demo
        from rsisys.loops.decision_making.feedback import enable_langmem_feedback
        from tests.test_feedback_required_outputs import CONFIG
        runner, _ = build_demo(CONFIG)
        with runner:
            loop = runner.decision_making
            enable_langmem_feedback(loop, memory_path=self.root / 'memory.sqlite',
                namespace=('default', 'run'), backend=self.backend, executor=ImmediateExecutor())
            service = loop.feedback_service
            for index in range(2):
                model = replace(loop.harness.policy.model, revision='active-' + str(index))
                loop.harness.policy.update_model(model)
                receipt = observation('step-' + str(index))
                self.backend.value = {'content': 'Current observation',
                    'source_refs': ['feedback:' + result_id(receipt.payload())], 'evaluation_claims': []}
                ident = service.observe(receipt)
                service.refresh()
                self.assertEqual(service.learning.state['jobs'][ident]['status'], 'completed')
                self.assertEqual(self.backend.requests[-1].model, model)
            context = loop.harness._context('objective', loop.action_space)
            self.assertEqual(json.loads(context.memory_context)['provider'], 'langmem')
            self.assertEqual(len(json.loads(context.memory_context)['memories']), 1)
            self.assertFalse(loop.harness.memory.recall())
            self.assertEqual(len(service.trajectories()), 2)

    def test_native_memory_and_optional_harness_proposals_share_the_feedback_pipeline(self):
        from rsisys.config import build_demo
        from rsisys.loops.decision_making.feedback import enable_langmem_feedback, ModelHarnessProposer
        from tests.test_feedback_required_outputs import CONFIG

        class MixedFixture(NativeFixture):
            def generate(self, request):
                names = [t['function']['name'] for t in request.tools]
                if names != ['propose_harness_changes']:
                    return super().generate(request)
                self.requests.append(request)
                packet = json.loads(request.prompt)
                value = {'proposals': [{'kind': 'harness_change', 'subject': 'meta_agent',
                    'content': 'Test a clearer parser.', 'applicability': 'This fixture harness.',
                    'validation_plan': 'Construct and test through a later decision.',
                    'source_refs': [packet['source_refs'][0]]}]}
                if 'candidate_evidence' in packet:
                    value['proposals'][0]['evaluation_claims'] = []
                return GenerationResponse('', request.model, True, tool_calls=[{
                    'id': 'proposal', 'type': 'function', 'function': {
                        'name': names[0], 'arguments': json.dumps(value)}}])

        self.backend = MixedFixture()
        runner, _ = build_demo(CONFIG)
        with runner:
            loop = runner.decision_making
            enable_langmem_feedback(loop, memory_path=self.root / 'memory.sqlite', namespace=('pipeline', 'run'),
                path=self.root / 'feedback.json', backend=self.backend, executor=ImmediateExecutor(),
                harness_proposals=ModelHarnessProposer(self.root / 'proposals'))
            receipt = observation('native-pipeline')
            self.backend.value = {'content': 'An execution was observed.',
                'source_refs': ['feedback:' + result_id(receipt.payload())], 'evaluation_claims': []}
            ident = loop.feedback_service.observe(receipt)
            loop.feedback_service.refresh()
            row = loop.feedback_service.learning.state['jobs'][ident]
            self.assertEqual(row['status'], 'completed', row['response'])
            self.assertEqual(row['products']['context_memory']['charged_calls'], 2)
            self.assertEqual(row['products']['harness_proposals']['charged_calls'], 1)
            self.assertEqual(loop.feedback_service.learning.state['attempted'], 3)
            context = loop.harness._context('objective', loop.action_space)
            self.assertEqual(json.loads(context.memory_context)['provider'], 'langmem')
            self.assertEqual(len(json.loads(context.memory_context)['memories']), 1)
            self.assertEqual(len(loop.feedback_service.proposals()), 1)
            self.assertTrue(all(request.model == loop.harness.policy.model for request in self.backend.requests))
            segment = loop.feedback_service.extract_trajectory(ident)
            self.assertFalse(segment['training_ready'])
            self.assertEqual(len(loop.feedback_service.learning.context_record(ident)['model_exchanges']), 2)

    def test_large_receipt_keeps_original_and_retrieval_has_a_hard_size_limit(self):
        from rsisys.loops.decision_making.feedback import WorkingMemoryPolicy
        learning = self.learning()
        text = 'large public receipt ' * 10000
        ident = self.capture(learning, 'large', content=text, memory_content='A large receipt was observed.')
        # Native processing uses a bounded projection; the raw receipt remains exact.
        self.assertEqual(learning.state['jobs'][ident]['raw_input']['receipt']['result']['summary'], text)
        self.assertEqual(learning.state['jobs'][ident]['status'], 'completed')
        with self.provider._store() as store:
            for i in range(20):
                store.put(self.provider._memory_ns, str(i), {'kind': 'Memory', 'content': {
                    'content': 'bounded memory ' * 100, 'source_refs': ['fixture'], 'evaluation_claims': []}})
        view = learning.working_memory(policy=WorkingMemoryPolicy(max_bytes=2048))
        self.assertLessEqual(len(json_bytes(view)), 2048)
        self.assertTrue(view['more_available'])
        self.assertGreater(view['omitted_from_selection'], 0)

    def test_host_limit_blocks_library_retry_and_commits_no_partial_memory(self):
        learning = self.learning(max_calls=1)
        self.backend.invalid_first = True
        ident = self.capture(learning, 'budget')
        self.assertEqual(learning.state['jobs'][ident]['status'], 'failed')
        self.assertEqual(learning.state['attempted'], 1)
        self.assertEqual(len(self.backend.requests), 1)
        self.assertEqual(learning.working_memory()['memories'], [])
        later = learning.capture_receipt(observation('later'), MODEL)
        learning.poll()
        self.assertEqual(learning.state['jobs'][later]['status'], 'budget_exhausted')

    def test_exhausted_native_validation_is_failure_not_a_successful_empty_update(self):
        learning = self.learning()
        self.capture(learning, 'valid', content='Keep this valid memory')
        before = learning.working_memory()
        ident = self.capture(learning, 'invalid', memory_content='x' * 3001)
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'failed')
        self.assertIn('discarded invalid', row['error'])
        self.assertEqual(learning.working_memory(), before)
        self.assertEqual(learning.state['attempted'], len(self.backend.requests))

    def test_identity_failure_and_full_request_admission_never_write_memory(self):
        for failure in ('wrong_identity', 'huge_tokens'):
            with self.subTest(failure=failure):
                self.root = self.root / failure
                self.root.mkdir()
                self.backend = NativeFixture()
                setattr(self.backend, failure, True)
                learning = self.learning()
                ident = self.capture(learning, 'failure')
                self.assertEqual(learning.state['jobs'][ident]['status'], 'failed')
                self.assertEqual(learning.state['attempted'], 1)
                self.assertEqual(len(self.backend.requests), int(failure == 'wrong_identity'))
                self.assertEqual(learning.working_memory()['memories'], [])

    def test_restart_recovers_committed_result_without_repeating_model_call(self):
        learning = self.learning()
        ident = self.capture(learning, 'durable')
        row = deepcopy(learning.state['jobs'][ident])
        expected_view = learning.working_memory()
        learning.close()
        # Simulate power loss after the store's atomic commit, before the owner
        # journal settled the conservative reservation.
        state = deepcopy(learning.state)
        state['attempted'] = row['reserved_calls']
        state['jobs'][ident].update(status='running', response=None)
        atomic_write(self.root / 'feedback.json', json_bytes(state))
        reopened = self.learning()
        self.assertEqual(reopened.state['jobs'][ident]['status'], 'completed')
        self.assertEqual(reopened.state['attempted'], 2)
        self.assertEqual(reopened.working_memory(), expected_view)
        reopened.poll()
        self.assertEqual(len(self.backend.requests), 2)

    def test_uncommitted_interruption_retains_reservation_without_replay(self):
        learning = self.learning()
        receipt = observation('uncertain')
        ident = learning.capture_receipt(receipt, MODEL)
        state = deepcopy(learning.state)
        state['attempted'] = 6
        state['jobs'][ident].update(status='running', reserved_calls=6)
        learning._commit(state)
        learning.close()
        reopened = self.learning()
        self.assertEqual(reopened.state['jobs'][ident]['status'], 'interrupted')
        self.assertEqual(reopened.state['attempted'], 6)
        reopened.poll()
        self.assertEqual(self.backend.requests, [])

    def test_candidate_metric_binding_is_validated_by_native_schema(self):
        evidence = {'evaluations': [{'source_ref': 'evaluation:A', 'candidate_id': 'A',
            'protocol_ref': 'dev', 'status': 'completed', 'simulated': False,
            'quality_metrics': {'accuracy': 0.25}}]}
        schema = _research_schema({'evaluation:A'}, evidence)
        value = {'content': 'A measured result', 'source_refs': ['evaluation:A'],
            'evaluation_claims': [{'candidate_id': 'A', 'evaluation_ref': 'evaluation:A',
                'protocol_ref': 'dev', 'metrics': {'accuracy': 0.25}, 'simulated': False}]}
        schema.model_validate(value)
        value['evaluation_claims'][0]['candidate_id'] = 'B'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            schema.model_validate(value)

    def test_native_store_write_failures_are_visible_and_batches_roll_back(self):
        with _AtomicSqliteStore.from_conn_string(str(self.root / 'atomic.sqlite')) as store:
            store.setup()
            namespace = ('test',)
            schema = _research_schema({'source'}, {'evaluations': []})
            staged = _StagedStore(store, namespace, schema)
            with self.assertRaisesRegex(ValueError, 'unsupplied'):
                staged.put(namespace, 'invalid', {'kind': 'Memory',
                    'content': {'content': 'bad', 'source_refs': ['wrong'], 'evaluation_claims': []}})
            self.assertTrue(staged.errors)
            self.assertIsNone(store.get(namespace, 'invalid'))
            # An earlier DELETE must roll back when a later INSERT fails.
            # Upstream 3.1.3 commits that DELETE from its finally block.
            store.put(namespace, 'existing', {'keep': True})
            store.conn.execute("CREATE TRIGGER reject_bad BEFORE INSERT ON store "
                "WHEN NEW.key = 'bad' BEGIN SELECT RAISE(ABORT, 'fixture write failure'); END")
            with self.assertRaisesRegex(Exception, 'fixture write failure'):
                store.batch([PutOp(namespace, 'existing', None), PutOp(namespace, 'bad', {'x': 2})])
            self.assertIsNotNone(store.get(namespace, 'existing'))


if __name__ == '__main__':
    unittest.main()
