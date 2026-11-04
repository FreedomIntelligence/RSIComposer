"""M07--M09 failure/recovery audit. CPU fixtures only; no provider clients."""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict, replace
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse
from rsisys.loops.decision_making.feedback.attempts import apply_attempt_updates
from rsisys.loops.decision_making.feedback.components import JournalContextMemory
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.feedback.pipeline import FeedbackPipeline
from rsisys.loops.decision_making.feedback.proposals import ModelHarnessProposer
from rsisys.loops.decision_making.feedback.service import FeedbackLearningService
from rsisys.runtime.json_tools import EvidencePathError, result_id
from tests.test_feedback_learning_integration import ImmediateExecutor
from tests.test_feedback_required_outputs import MODEL, observation, exact


class FixtureBackend:
    supports_response_schema = True

    def __init__(self, *, tokens=100, finish_reason=None):
        self.requests, self.tokens, self.finish_reason = [], tokens, finish_reason

    def count_tokens(self, request):
        return self.tokens

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse('{"proposals":[]}', request.model, False,
                                  finish_reason=self.finish_reason)


class RecordedProcessor:
    max_calls_per_job = 3

    def __init__(self, result=None):
        self.result = result if result is not None else {
            'status': 'completed', 'error': None, 'charged_calls': 0,
            'proposals': [], 'attempt_updates': []}
        self.processed = 0
        self.saved = None

    def configuration(self):
        return {'provider': 'audit_fixture'}

    def process(self, job, backend, limits):
        self.processed += 1
        self.saved = self.result(deepcopy(job)) if callable(self.result) else deepcopy(self.result)
        return deepcopy(self.saved)

    def recover(self, job):
        return deepcopy(self.saved)


class FeedbackLifecycleAudit(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def learning(self, backend=None, **options):
        backend = backend or FixtureBackend()
        learning = FeedbackLearning(SerializedLearningBackend(backend), executor=ImmediateExecutor(), **options)
        self.addCleanup(lambda: None if learning._closed else learning.close())
        return learning, backend

    def capture(self, learning, ident='one', **options):
        return learning.capture_receipt(observation(ident), MODEL, **options)

    def recover_running(self, processor, **capture_options):
        path = self.root / 'journal.json'
        learning, backend = self.learning(processor=processor, path=path)
        ident = self.capture(learning, **capture_options)
        learning.poll()  # Durable running reservation; synchronous fixture result exists.
        self.assertEqual(learning.state['jobs'][ident]['status'], 'running')
        learning._futures.clear()  # Simulated owner loss, no live worker or inference.
        learning._closed = True
        restored, _ = self.learning(backend, processor=processor, path=path)
        return restored, backend, ident

    def test_recovery_validates_native_outputs_before_any_read_or_working_view(self):
        def invalid(job):
            return {'status': 'completed', 'charged_calls': 0, 'proposals': [{
                'kind': 'memory', 'content': 'Invalid source must never become resident memory.',
                'applicability': 'fixture', 'validation_plan': 'reject', 'source_refs': ['unsupplied']}]}
        restored, backend, ident = self.recover_running(RecordedProcessor(invalid))
        self.assertEqual(restored.state['jobs'][ident]['status'], 'failed')
        self.assertEqual(restored.read(ident)['proposals'], [])
        self.assertNotIn('Invalid source must never', json.dumps(restored.working_memory()))
        self.assertEqual(backend.requests, [])

    def test_recovery_applies_valid_attempt_updates_once_with_exact_provenance(self):
        def result(job):
            return {'status': 'completed', 'charged_calls': 0, 'proposals': [], 'attempt_updates': [{
                'attempt_id': None, 'base_revision': 0, 'title': 'Fixture idea', 'summary': 'Observed only',
                'member_decision_ids': ['decision-one'], 'source_refs': ['decision:decision-one']}]}
        restored, backend, ident = self.recover_running(RecordedProcessor(result), grouping_context={
            'decisions': [{'decision_id': 'decision-one', 'status': 'completed'}], 'attempts': []})
        revisions = restored.attempts()['items']
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0]['revision'], 1)
        self.assertEqual(revisions[0]['provenance']['input_digest'], result_id(restored.state['jobs'][ident]['input']))
        restored.poll()
        self.assertEqual(restored.attempts()['items'], revisions)
        self.assertEqual(backend.requests, [])

    def test_malformed_processor_results_fail_closed_with_reservation_retained(self):
        for result in (None, [], {'charged_calls': 0}, {'status': 'running', 'charged_calls': 0},
                       {'status': 'completed', 'charged_calls': True},
                       {'status': 'completed', 'charged_calls': 0, 'job_id': 'wrong'},
                       {'status': 'completed', 'charged_calls': 0, 'reserved_calls': 0}):
            with self.subTest(result=result):
                processor = RecordedProcessor()
                processor.result = result
                learning, backend = self.learning(processor=processor)
                ident = self.capture(learning)
                learning.poll()
                learning.poll()
                self.assertEqual(learning.state['jobs'][ident]['status'], 'failed')
                self.assertEqual(learning.state['jobs'][ident]['job_id'], ident)
                self.assertEqual(learning.state['attempted'], 3)
                self.assertEqual(backend.requests, [])

    def test_processor_cannot_remove_captured_simulation_provenance(self):
        processor = RecordedProcessor({'status': 'completed', 'charged_calls': 0,
                                       'proposals': [], 'simulated': False})
        learning, _ = self.learning(processor=processor)
        ident = self.capture(learning)
        learning.poll(); learning.poll()
        self.assertTrue(learning.state['jobs'][ident]['simulated'])

    def test_recovery_failure_isolated_without_replaying_uncertain_work(self):
        class BrokenRecovery(RecordedProcessor):
            def recover(self, job):
                raise OSError('recovery store unavailable')
        restored, backend, ident = self.recover_running(BrokenRecovery())
        self.assertEqual(restored.state['jobs'][ident]['status'], 'interrupted')
        self.assertIn('recovery store unavailable', restored.state['jobs'][ident]['error'])
        self.assertEqual(restored.state['attempted'], 3)
        restored.poll()
        self.assertEqual(backend.requests, [])

    def test_legacy_token_counter_rejects_negative_boolean_and_noninteger_before_inference(self):
        for tokens in (-1, True, 1.5, None):
            with self.subTest(tokens=tokens):
                learning, backend = self.learning(FixtureBackend(tokens=tokens))
                ident = self.capture(learning)
                learning.poll(); learning.poll()
                self.assertEqual(learning.state['jobs'][ident]['status'], 'failed')
                self.assertEqual(backend.requests, [])

    def test_truncated_legacy_json_is_not_accepted_as_completed_learning(self):
        learning, backend = self.learning(FixtureBackend(finish_reason='length'))
        ident = self.capture(learning)
        learning.poll(); learning.poll()
        self.assertEqual(learning.state['jobs'][ident]['status'], 'failed')
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(learning.state['jobs'][ident]['response']['finish_reason'], 'length')

    def test_attempt_extension_keeps_simulated_ancestry_and_does_not_mutate_previous_revision(self):
        previous = {'attempt_id': 'idea', 'revision': 1, 'member_decision_ids': ['old'],
            'source_refs': ['decision:old'], 'simulated': True, 'title': 'Idea', 'summary': 'old'}
        before = deepcopy(previous)
        row = {'job_id': 'new', 'model': asdict(MODEL), 'input': {}, 'simulated': False,
            'attempt_updates': [{'attempt_id': 'idea', 'base_revision': 1, 'title': 'Idea',
                'summary': 'extended', 'member_decision_ids': ['new'], 'source_refs': ['decision:new']}]}
        revision, = apply_attempt_updates(row, [previous])
        self.assertTrue(revision['simulated'])
        self.assertEqual(revision['member_decision_ids'], ['old', 'new'])
        self.assertEqual(previous, before)

    def test_proposer_recovery_rejects_changed_instructions_even_with_same_receipt_and_model(self):
        learning, _ = self.learning()
        ident = self.capture(learning)
        job = learning.state['jobs'][ident]
        proposer = ModelHarnessProposer(self.root / 'proposals', instructions='version one')
        backend = FixtureBackend()
        result = proposer.process(job, SerializedLearningBackend(backend), learning.limits)
        self.assertEqual(result['status'], 'completed')
        changed = ModelHarnessProposer(self.root / 'proposals', instructions='version two')
        with self.assertRaisesRegex(ValueError, 'configuration'):
            changed.recover(job)
        self.assertEqual(len(backend.requests), 1)

    def test_optional_recovery_exception_preserves_committed_memory_and_unknown_call_reservation(self):
        memory = RecordedProcessor()
        class BrokenProposal(RecordedProcessor):
            def recover(self, job):
                raise OSError('optional result unavailable')
        pipeline = FeedbackPipeline(memory_processor=memory, harness_proposals=BrokenProposal())
        restored, backend, ident = self.recover_running(pipeline)
        row = restored.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['products']['context_memory']['status'], 'completed')
        self.assertEqual(row['products']['harness_proposals']['status'], 'interrupted')
        self.assertEqual(restored.state['attempted'], 6)
        self.assertIn('optional result unavailable', json.dumps(row['response']))
        self.assertEqual(backend.requests, [])

    def test_proposer_recovery_does_not_reuse_result_for_different_evidence_snapshot(self):
        learning, backend = self.learning()
        ident = self.capture(learning)
        job = learning.state['jobs'][ident]
        proposer = ModelHarnessProposer(self.root / 'proposals')
        proposer.process(job, SerializedLearningBackend(backend), learning.limits)
        changed = deepcopy(job)
        changed['input']['research_context']['new_evidence'] = True
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            proposer.recover(changed)

    def test_default_adapter_includes_nested_evaluation_candidate_producer(self):
        from rsisys.loops.decision_making.harness.learning_harness import DefaultFeedbackAdapter
        records = {'producer': {'kind': 'explore', 'proposal': {'kind': 'explore'}, 'status': 'completed'}}
        for index in range(24):
            ident = 'tail-' + str(index)
            records[ident] = {'kind': 'explore', 'proposal': {'kind': 'explore'}, 'status': 'completed'}
        records['tail-23']['proposal']['directive'] = {'mode': 'evaluate', 'strategy': 'Compare',
            'payload': {'candidate': {'candidate_id': 'candidate-produced'}}, 'input_refs': []}
        harness = SimpleNamespace(policy=SimpleNamespace(model=MODEL),
            state=SimpleNamespace(decision_records=records, directive_decisions={},
                candidate_origins={'candidate-produced': 'producer'}),
            memory=SimpleNamespace(history=lambda: ()),
            _decision_graph=lambda: [{'decision_id': ident, 'status': 'completed'} for ident in records])
        adapter = DefaultFeedbackAdapter(harness)
        adapter.service = SimpleNamespace(attempts=lambda: {'items': []})
        group = adapter._grouping_scope(None, decision_id='tail-23')
        self.assertIn('producer', [d['decision_id'] for d in group['decisions']])

    def test_trajectory_preserves_failed_learning_receipt_and_identity_without_generated_targets(self):
        learning, _ = self.learning(FixtureBackend(finish_reason='length'), max_pending=1)
        first = self.capture(learning)
        blocked = self.capture(learning, 'two')
        learning.poll(); learning.poll()
        before = deepcopy(learning.state)
        for ident in (first, blocked):
            segment = learning.extract_trajectory(ident)
            self.assertEqual(segment['execution_receipt'], json.loads(json.dumps(observation(
                'one' if ident == first else 'two').payload())))
            self.assertEqual(segment['issuing_model'], asdict(MODEL))
            self.assertFalse(segment['training_ready'])
            self.assertFalse(segment['complete_conversation'])
            self.assertNotIn('training_example', segment)
            self.assertEqual(segment, exact(learning, job_id=ident, evidence_path='/trajectory_segment'))
        self.assertEqual(learning.state, before)

    def test_publication_journal_failure_retries_same_memory_without_second_model_call(self):
        learning, backend = self.learning()
        ident = self.capture(learning)
        learning.poll(); learning.poll()
        delivered = []
        service = FeedbackLearningService(learning, SimpleNamespace(publish=delivered.append))
        with patch.object(learning, '_commit', side_effect=OSError('journal unavailable')):
            with self.assertRaises(OSError):
                service._publish()
        self.assertFalse(learning.state['jobs'][ident].get('published', False))
        service.refresh()
        self.assertEqual(len(delivered), 2)
        self.assertEqual(delivered[0], delivered[1])
        self.assertEqual(len(backend.requests), 1)
        self.assertTrue(learning.state['jobs'][ident]['published'])

    def test_native_gateway_admits_complete_request_and_persists_before_dispatch(self):
        from rsisys.loops.decision_making.feedback.model_gateway import FeedbackModelGateway
        path = self.root / 'exchange.json'
        request = GenerationRequest('decision_maker', MODEL, '',
            messages=[{'role': 'system', 'content': 'Full system'},
                      {'role': 'user', 'content': 'Native message'}],
            tools=[{'type': 'function', 'function': {'name': 'record', 'parameters': {
                'type': 'object', 'properties': {}}}}])
        test = self
        class DurableBackend(FixtureBackend):
            def count_tokens(self, counted):
                test.assertEqual(counted, request)
                return 100

            def generate(self, dispatched):
                saved = json.loads(path.read_text())['exchanges']
                test.assertEqual(saved[0]['status'], 'reserved')
                test.assertEqual(saved[0]['request'], json.loads(json.dumps(asdict(request))))
                return super().generate(dispatched)
        backend = DurableBackend()
        gateway = FeedbackModelGateway(backend, MODEL, max_calls=1, max_input_tokens=100, path=path)
        gateway.generate(request)
        snapshot = gateway.snapshot()
        snapshot[0]['request'].clear()
        self.assertEqual(gateway.snapshot()[0]['request'], asdict(request))
        with self.assertRaisesRegex(RuntimeError, 'call limit'):
            gateway.generate(request)
        with self.assertRaisesRegex(ValueError, 'uncertain'):
            FeedbackModelGateway(backend, MODEL, max_calls=1, max_input_tokens=100, path=path)
        self.assertEqual(len(backend.requests), 1)

    def test_native_gateway_rejects_identity_token_overflow_and_journal_failure_without_dispatch(self):
        from rsisys.loops.decision_making.feedback.model_gateway import FeedbackModelGateway
        for scenario in ('identity', 'tokens', 'write'):
            with self.subTest(scenario=scenario):
                backend = FixtureBackend(tokens=101)
                gateway = FeedbackModelGateway(backend, MODEL, max_calls=1,
                    max_input_tokens=100 if scenario == 'tokens' else 200,
                    path=self.root / (scenario + '.json'))
                request = GenerationRequest('decision_maker',
                    replace(MODEL, revision='wrong') if scenario == 'identity' else MODEL, 'fixture')
                if scenario == 'write':
                    with patch.object(gateway, '_save', side_effect=OSError('disk unavailable')):
                        with self.assertRaises(OSError):
                            gateway.generate(request)
                else:
                    with self.assertRaises(ValueError):
                        gateway.generate(request)
                self.assertEqual(backend.requests, [])
                self.assertEqual(gateway.attempted, 0 if scenario == 'identity' else 1)

    def test_candidate_binding_cannot_be_changed_by_an_organizer_snapshot(self):
        from tests.test_feedback_candidate_evidence import CandidateEvidenceTests
        case = CandidateEvidenceTests()
        self.addCleanup(case.doCleanups)
        _, learning, service, published = case.make()
        case.complete(service)
        before = deepcopy(learning.state)
        class EditingOrganizer(JournalContextMemory):
            @staticmethod
            def record(job):
                job['input']['candidate_evidence']['evaluations'][0]['candidate_id'] = 'wrong'
                job['proposals'][0]['evaluation_claims'][0]['candidate_id'] = 'wrong'
                return {'edited': True}
        learning._context_memory = EditingOrganizer()
        self.assertEqual(learning.context_record('call48'), {'edited': True})
        self.assertEqual(learning.state, before)
        self.assertEqual(len(published), 1)

    def test_research_projection_requires_read_only_per_collection_archive_and_retains_source(self):
        from tests.test_research_evidence_projection import fixture
        from rsisys.loops.decision_making.harness.research_state import (
            archive_available, evidence_working_view, receipt_needs_protection, resident_evidence_indices)
        raw, _, _, _ = fixture(count=8)
        before = deepcopy(raw)
        writable = replace(raw, operations=tuple(replace(op, read_only=False) for op in raw.operations))
        self.assertFalse(archive_available(writable, 'history'))
        self.assertEqual(evidence_working_view(writable), writable)
        self.assertEqual(resident_evidence_indices(writable, 'history'),
                         {i for i, h in enumerate(raw.history) if receipt_needs_protection(h.details)})
        view = evidence_working_view(raw)
        card = view.history[0].details
        self.assertEqual(card['evidence_card']['sha256'], result_id(asdict(raw.history[0])))
        self.assertEqual(card['evidence_card']['read_path']['record_id'], raw.history[0].record_id)
        self.assertEqual(card['facts']['/details/feedback/quality_metrics']['accuracy'], 0.0)
        self.assertEqual(raw, before)

    def test_proposal_resolution_uses_ordinary_decision_links_without_applying_or_training(self):
        learning, _ = self.learning()
        ident = self.capture(learning)
        learning.poll(); learning.poll()
        row = learning.state['jobs'][ident]
        row['proposals'] = [{'kind': 'harness_change', 'subject': 'lab', 'content': 'Proposed only',
            'applicability': 'fixture', 'validation_plan': 'local review',
            'source_refs': row['input']['source_refs'][:1]}]
        proposal_id = ident + ':0'
        service = FeedbackLearningService(learning, SimpleNamespace(proposal_decisions=lambda: [
            {'decision_id': 'ordinary', 'feedback_proposal_ids': [proposal_id]}]))
        before = deepcopy(learning.state)
        record, = service.proposals()
        self.assertEqual(record['decision_links'][0]['decision_id'], 'ordinary')
        self.assertFalse(record['training_ready'])
        self.assertFalse(record['applied'])
        self.assertEqual(service.resolve_harness_proposals([proposal_id])[0]['proposal_digest'],
                         result_id(row['proposals'][0]))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            service.resolve_harness_proposals(['missing'])
        self.assertEqual(learning.state, before)


@unittest.skipUnless(find_spec('langmem'), 'Optional LangMem environment required')
class NativeFeedbackAudit(unittest.TestCase):
    def test_staged_native_write_keeps_host_derived_simulation_flag_without_mutating_caller(self):
        from langgraph.store.base import PutOp
        from rsisys.loops.decision_making.feedback.langmem import _StagedStore, _research_schema
        schema = _research_schema({'source'}, {'evaluations': []}, {'source'})
        staged = _StagedStore(None, ('memory',), schema)
        value = {'kind': 'Memory', 'content': {'content': 'Fixture memory', 'source_refs': ['source'],
                                            'evaluation_claims': [], 'simulated': False}}
        original = deepcopy(value)
        staged.batch([PutOp(('memory',), 'one', value)])
        self.assertTrue(staged.puts[0].value['content']['simulated'])
        self.assertEqual(value, original)

    def test_langmem_recovery_rejects_changed_component_configuration(self):
        from langgraph.store.base import PutOp
        from rsisys.loops.decision_making.feedback.langmem import LangMemContextMemory
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.sqlite'
            provider = LangMemContextMemory(path, namespace=('audit',), consolidate=False)
            job = {'job_id': 'one', 'receipt_digest': 'receipt', 'model': asdict(MODEL)}
            with provider._store() as store:
                store.batch([PutOp(provider._jobs_ns, 'one', {**job,
                    'configuration': provider.configuration(), 'result': {'status': 'completed'}})])
            changed = LangMemContextMemory(path, namespace=('audit',), consolidate=True)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                changed.recover(job)

    def test_langmem_recovery_requires_matching_input_digest_and_returns_detached_result(self):
        from langgraph.store.base import PutOp
        from rsisys.loops.decision_making.feedback.langmem import LangMemContextMemory
        with TemporaryDirectory() as directory:
            provider = LangMemContextMemory(Path(directory) / 'memory.sqlite', namespace=('audit',))
            job = {'job_id': 'one', 'receipt_digest': 'receipt', 'model': asdict(MODEL),
                   'input': {'captured': 'evidence'}}
            result = {'status': 'completed', 'proposals': [], 'charged_calls': 1}
            with provider._store() as store:
                store.batch([PutOp(provider._jobs_ns, 'one', {**job,
                    'configuration': provider.configuration(), 'input_digest': result_id(job['input']),
                    'result': result})])
            recovered = provider.recover(job)
            self.assertEqual(recovered, result)
            recovered['proposals'].append({'changed': True})
            self.assertEqual(provider.recover(job), result)
            changed = deepcopy(job)
            changed['input']['captured'] = 'different'
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                provider.recover(changed)

    def test_bridge_rejects_identity_mismatch_and_unsupported_options(self):
        from langchain_core.messages import HumanMessage
        from rsisys.adapters.langchain_model import BackendChatModel
        from rsisys.runtime.model import ModelResponseError
        class WrongIdentity(FixtureBackend):
            def generate(self, request):
                self.requests.append(request)
                return GenerationResponse('fixture', replace(request.model, revision='wrong'), False)
        backend = WrongIdentity()
        bridge = BackendChatModel(backend=backend, model_ref=MODEL, cache=False)
        messages = [HumanMessage(content='fixture')]
        for options in ({'stop': ['stop']}, {'temperature': 0.0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                bridge._generate(messages, **options)
        self.assertEqual(backend.requests, [])
        with self.assertRaises(ModelResponseError):
            bridge._generate(messages)
        self.assertEqual(len(backend.requests), 1)

    def test_native_existing_uncertain_exchanges_cannot_refund_the_reservation(self):
        from rsisys.loops.decision_making.feedback.langmem import LangMemContextMemory
        with TemporaryDirectory() as directory:
            provider = LangMemContextMemory(Path(directory) / 'memory.sqlite', namespace=('audit',))
            backend = FixtureBackend()
            learning = FeedbackLearning(SerializedLearningBackend(backend), executor=ImmediateExecutor())
            self.addCleanup(learning.close)
            ident = learning.capture_receipt(observation(), MODEL)
            job = learning.state['jobs'][ident]
            provider.audit_dir.mkdir()
            (provider.audit_dir / (ident + '.json')).write_text('{"exchanges":[{"status":"reserved"}]}')
            result = provider.process(job, learning.backend, {**learning.limits, 'max_calls': 3})
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['charged_calls'], 3)
            self.assertEqual(backend.requests, [])

    def test_bridge_rejects_undeclared_tool_call_without_executing_or_dropping_it(self):
        from langchain_core.messages import HumanMessage
        from rsisys.adapters.langchain_model import BackendChatModel
        from rsisys.runtime.model import ModelResponseError
        class ToolBackend(FixtureBackend):
            def generate(self, request):
                self.requests.append(request)
                return GenerationResponse('', request.model, False, tool_calls=[{
                    'id': 'call', 'type': 'function', 'function': {'name': 'undeclared', 'arguments': '{}'}}])
        backend = ToolBackend()
        model = BackendChatModel(backend=backend, model_ref=MODEL, cache=False)
        with self.assertRaises(ModelResponseError):
            model._generate([HumanMessage(content='Fixture')])
        self.assertEqual(len(backend.requests), 1)


if __name__ == '__main__':
    unittest.main()
