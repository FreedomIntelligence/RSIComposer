"""Repairable tool feedback and producer-event delivery, without real inference."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from rsisys.adapters.execution import LabExecutionTools
from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.application import Application, ApplicationConfig
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.delivery import FeedbackDelivery
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse
from rsisys.loops.contracts.receipts import ExecutionReceipt
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.feedback.pipeline import FeedbackPipeline
from rsisys.loops.decision_making.harness.research_state import bounded_evidence_view
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.research import ResearchAssembly
from rsisys.runtime.json_tools import result_id
from rsisys.snapshot.codec import encode, decode
from tests.test_feedback_learning_integration import ImmediateExecutor
from tests.test_feedback_required_outputs import MODEL, EmptyBackend


ROOT = Path(__file__).resolve().parents[1]


class RecordedMemory:
    def __init__(self):
        self.events = []

    def process(self, event, model):
        self.events.append(event)
        return {'recorded': event.event_id}


class ObservedContext:
    def __init__(self):
        self.events = []

    def context(self, query):
        return ''

    def observe(self, event):
        self.events.append(event)


class LocalLabDeliveryTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.memory, self.context = RecordedMemory(), ObservedContext()
        config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'),
                         data_dir=self.root / 'run', execution='local_process')
        def factory(context):
            self.lab = LocalProcessLab(context.sandbox, context.artifacts, timeout_seconds=5)
            return self.lab
        self.app = Application(config, lab_factory=factory, research=ResearchAssembly(
            context_memory=self.context, memory_processor=self.memory))
        self.addCleanup(self.app.close)
        self.runner = self.app.runner
        self.harness = self.runner.decision_making.harness
        self.service = self.runner.decision_making.feedback_service
        self.requests = []

    def invoke(self, name, **arguments):
        def generate(request):
            self.requests.append(request)
            return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': name,
                'arguments': arguments}), request.model, simulated=False)
        self.harness.policy.backend.generate = generate
        return self.runner.step('CPU delivery review fixture')

    def flush(self):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            self.service.refresh()
            if not any(r['status'] in ('queued', 'running') for r in self.service.learning.state['jobs'].values()):
                return
            time.sleep(.01)
        self.fail('Feedback fixture did not finish')

    def start(self, key='one', code='print("result")'):
        result = self.invoke('lab_start', key=key, argv=[sys.executable, '-c', code])
        ident = result.event['operation_result']['data']['job_id']
        self.lab.backend.wait(self.lab._jobs[ident][0])
        return ident

    def result_rows(self):
        return [r for r in self.service.learning.state['jobs'].values()
                if r['raw_input']['receipt']['operation']['name'] == 'lab_result']

    def result_events(self, events):
        return [e for e in events if e.request.get('name') == 'lab_result']

    def test_missing_file_is_failed_feedback_and_next_actual_decision_request_can_repair(self):
        failed = self.invoke('lab_read_file', path='missing.py')
        self.assertEqual(failed.kind, 'progressed')
        self.assertEqual(failed.event['operation_result']['status'], ExecutionStatus.FAILED)
        self.assertIn('Workspace file does not exist', failed.event['operation_result']['error'])
        fixed = self.invoke('lab_write_file', path='missing.py', content='print("fixed")')
        self.assertEqual(fixed.event['operation_result']['status'], ExecutionStatus.COMPLETED)
        self.assertIn('Workspace file does not exist', self.requests[-1].prompt)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.lab._jobs, {})

    def test_input_and_missing_executable_errors_are_returned_without_effects(self):
        for name, args in (
            ('lab_write_file', {'path': 'x'}),
            ('lab_start', {'key': 'bad', 'argv': 'bad'}),
            ('lab_start', {'key': 'bad', 'argv': []}),
            ('lab_start', {'key': 'missing', 'argv': [str(self.root / 'missing-program')]}),
            ('lab_poll', {'job_id': 'unknown'}),
            ('lab_read_file', {'path': 123}),
            ('lab_read_file', {'path': 'missing-parent/file'}),
        ):
            with self.subTest(name=name, args=args):
                step = self.invoke(name, **args)
                self.assertEqual(step.event['operation_result']['status'], ExecutionStatus.FAILED)
                self.assertIn('LabToolError', step.event['operation_result']['error'])
        self.assertEqual(self.lab._jobs, {})
        self.assertEqual(self.lab._keys, {})

    def test_permissions_identity_and_infrastructure_errors_still_raise_to_host(self):
        self.start()
        with self.assertRaisesRegex(ValueError, 'different arguments'):
            self.invoke('lab_start', key='one', argv=[sys.executable, '-c', 'print(2)'])
        self.lab.max_jobs = 1
        with self.assertRaisesRegex(PermissionError, 'admission limit'):
            self.invoke('lab_start', key='two', argv=[sys.executable, '-c', 'print(2)'])
        with self.assertRaisesRegex(ValueError, 'host limit'):
            self.invoke('lab_start', key='excess', argv=[sys.executable, '-c', 'pass'], timeout_seconds=6)
        with self.assertRaisesRegex(ValueError, 'relative workspace'):
            self.invoke('lab_read_file', path='../outside')
        self.invoke('lab_write_file', path='output', content='preserved')
        with patch.object(self.lab.artifacts, 'put', side_effect=OSError('fixture storage failure')):
            with self.assertRaisesRegex(OSError, 'fixture storage failure'):
                self.invoke('lab_export', path='output')

    def test_missing_workspace_is_infrastructure_failure_not_missing_tool(self):
        self.lab.workspace.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.invoke('lab_start', key='one', argv=[sys.executable, '-c', 'pass'])
        with self.assertRaises(FileNotFoundError):
            self.invoke('lab_read_file', path='missing')

    def test_repeat_collection_archives_every_call_but_processes_memory_once_then_explicitly_again(self):
        ident = self.start(code='from pathlib import Path; Path("count").write_text("once"); print("result")')
        self.invoke('lab_poll', job_id=ident)  # observation must not reserve the producer event
        self.invoke('lab_result', job_id=ident)
        self.invoke('lab_result', job_id=ident)
        self.flush()
        first, repeated = self.result_rows()
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(repeated['status'], 'archived')
        self.assertEqual(repeated['duplicate_of'], first['job_id'])
        self.assertEqual(len(self.result_events(self.memory.events)), 1)
        self.assertEqual(len(self.result_events(self.context.events)), 1)
        self.assertEqual(len(self.service.trajectories()), 4)
        self.assertIn('result', self.requests[-1].prompt)
        self.invoke('lab_result', job_id=ident, reprocess=True)
        self.flush()
        self.assertEqual(len(self.result_events(self.memory.events)), 2)
        self.assertEqual(len(self.result_events(self.context.events)), 2)
        self.assertEqual(self.result_rows()[-1]['status'], 'completed')
        self.assertEqual(len(self.service.trajectories()), 5)
        self.assertEqual((self.lab.workspace / 'count').read_text(), 'once')
        self.assertEqual(len(self.lab._jobs), 1)

    def test_identical_output_from_different_real_programs_is_not_deduplicated(self):
        for key in ('one', 'two'):
            self.invoke('lab_result', job_id=self.start(key))
        self.flush()
        self.assertEqual(len(self.result_events(self.memory.events)), 2)
        self.assertEqual(len(self.lab._jobs), 2)
        rows = self.result_rows()
        self.assertNotEqual(rows[0]['producer_event']['key'], rows[1]['producer_event']['key'])

    def test_failed_program_result_can_be_read_and_reprocessed_without_rerunning(self):
        ident = self.start(code='raise RuntimeError("observed failure")')
        for options in ({}, {}, {'reprocess': True}):
            step = self.invoke('lab_result', job_id=ident, **options)
            self.assertEqual(step.event['operation_result']['status'], ExecutionStatus.FAILED)
        self.flush()
        self.assertEqual([r['status'] for r in self.result_rows()], ['completed', 'archived', 'completed'])
        self.assertEqual(len(self.result_events(self.memory.events)), 2)
        self.assertEqual(len(self.lab._jobs), 1)


class ProducerEventDeliveryTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.backend = EmptyBackend()
        self.memory, self.context = RecordedMemory(), ObservedContext()

    def learning(self, **options):
        value = FeedbackLearning(SerializedLearningBackend(self.backend), executor=ImmediateExecutor(),
            processor=FeedbackPipeline(memory_processor=self.memory), context_memory=self.context, **options)
        self.addCleanup(lambda: None if value._closed else value.close())
        return value

    @staticmethod
    def receipt(task='collect-one', event='producer:job-1:result', *, reprocess=False,
                output='native result', lab='lab', run='run'):
        return ExecutionReceipt.from_output(task, output, status=ExecutionStatus.COMPLETED,
            request={'collector_specific': task}, lab_id=lab, run_id=run,
            decision_id='decision-' + task,
            feedback_delivery=FeedbackDelivery(event, reprocess) if event is not None else None)

    @staticmethod
    def flush(learning):
        learning.notify_context()
        learning.poll()
        learning.poll(dispatch=False)

    def test_arbitrary_native_output_uses_only_explicit_metadata_for_deduplication(self):
        learning = self.learning()
        one = learning.capture_receipt(self.receipt(), MODEL)
        two = learning.capture_receipt(self.receipt('collect-two'), MODEL)
        self.flush(learning)
        self.assertEqual(learning.state['jobs'][two]['duplicate_of'], one)
        self.assertEqual(len(self.memory.events), 1)
        self.assertEqual(len(self.context.events), 1)
        self.assertEqual(len(learning.trajectories()), 2)
        self.assertEqual(self.memory.events[0].identity['feedback_delivery']['event_id'], 'producer:job-1:result')

    def test_duplicate_receipt_retry_is_idempotent_and_cannot_change_processing_mode(self):
        learning = self.learning()
        learning.capture_receipt(self.receipt(), MODEL)
        repeated = self.receipt('collect-two')
        ident = learning.capture_receipt(repeated, MODEL)
        self.assertEqual(learning.capture_receipt(repeated, MODEL), ident)
        self.assertEqual(len(learning.state['jobs']), 2)
        with self.assertRaisesRegex(ValueError, 'mode cannot change'):
            learning.capture_receipt(repeated, MODEL, archive_only=True)

    def test_same_event_with_changed_result_is_identity_error_including_reprocess(self):
        learning = self.learning()
        learning.capture_receipt(self.receipt(), MODEL)
        for reprocess in (False, True):
            with self.subTest(reprocess=reprocess), self.assertRaisesRegex(ValueError, 'identity reused'):
                learning.capture_receipt(self.receipt('changed', output='different', reprocess=reprocess), MODEL)
        self.assertEqual(len(learning.state['jobs']), 1)

    def test_different_producer_lab_or_run_are_independent_events(self):
        learning = self.learning()
        for task, event, lab, run in (
            ('one', 'event', 'lab', 'run'), ('two', 'other', 'lab', 'run'),
            ('three', 'event', 'other-lab', 'run'), ('four', 'event', 'lab', 'other-run'),
        ):
            learning.capture_receipt(self.receipt(task, event, lab=lab, run=run), MODEL)
        for _ in range(4):
            self.flush(learning)
        self.assertEqual(len(self.memory.events), 4)

    def test_native_business_fields_do_not_control_delivery_and_legacy_remains_per_receipt(self):
        learning = self.learning()
        output = {'feedback_delivery': {'event_id': 'same'}, 'reprocess': False, 'job_id': 'same'}
        for task in ('one', 'two'):
            learning.capture_receipt(self.receipt(task, None, output=output), MODEL)
            self.flush(learning)
        self.assertEqual(len(self.memory.events), 2)
        self.assertTrue(all('producer_event' not in row for row in learning.state['jobs'].values()))

    def test_read_only_archive_does_not_reserve_processing(self):
        learning = self.learning()
        learning.capture_receipt(self.receipt('read'), MODEL, archive_only=True)
        processed = learning.capture_receipt(self.receipt('collect'), MODEL)
        self.flush(learning)
        self.assertEqual(learning.state['jobs'][processed]['status'], 'completed')
        self.assertEqual(len(self.memory.events), 1)

    def test_archive_only_reads_still_enforce_immutable_producer_identity(self):
        learning = self.learning()
        for existing_archive, next_archive in ((True, False), (False, True), (True, True)):
            event = f'event:{existing_archive}:{next_archive}'
            learning.capture_receipt(self.receipt('first-' + event, event), MODEL,
                                     archive_only=existing_archive)
            before = len(learning.state['jobs'])
            for reprocess in (False, True):
                with self.subTest(existing_archive=existing_archive, next_archive=next_archive,
                                  reprocess=reprocess), self.assertRaisesRegex(ValueError, 'identity reused'):
                    learning.capture_receipt(self.receipt('changed-' + event, event,
                        output='different', reprocess=reprocess), MODEL, archive_only=next_archive)
            self.assertEqual(len(learning.state['jobs']), before)

    def test_journal_mapping_order_cannot_change_original_processing_identity(self):
        path = self.root / 'feedback.json'
        first = self.learning(path=path)
        original = first.capture_receipt(self.receipt(), MODEL)
        self.flush(first)
        first.capture_receipt(self.receipt('duplicate-before-reload'), MODEL)
        first.capture_receipt(self.receipt('explicit-before-reload', reprocess=True), MODEL)
        self.flush(first)
        first.close()
        saved = json.loads(path.read_text())
        saved['jobs'] = dict(reversed(list(saved['jobs'].items())))
        path.write_text(json.dumps(saved))
        reopened = self.learning(path=path)
        duplicate = reopened.capture_receipt(self.receipt('after-reordered-reload'), MODEL)
        self.flush(reopened)
        self.assertEqual(reopened.state['jobs'][duplicate]['duplicate_of'], original)
        self.assertEqual(len(self.memory.events), 2)

    def test_processing_identity_survives_journal_reload(self):
        path = self.root / 'feedback.json'
        first = self.learning(path=path)
        original = first.capture_receipt(self.receipt(), MODEL)
        self.flush(first)
        first.close()
        reopened = self.learning(path=path)
        duplicate = reopened.capture_receipt(self.receipt('after-reload'), MODEL)
        self.flush(reopened)
        self.assertEqual(reopened.state['jobs'][duplicate]['duplicate_of'], original)
        self.assertEqual(len(self.memory.events), 1)
        explicit = reopened.capture_receipt(self.receipt('explicit', reprocess=True), MODEL)
        self.flush(reopened)
        self.assertEqual(reopened.state['jobs'][explicit]['status'], 'completed')
        self.assertEqual(len(self.memory.events), 2)

    def test_failed_capture_does_not_lose_the_only_memory_delivery(self):
        learning = self.learning()
        with patch.object(learning, '_commit', side_effect=OSError('fixture journal unavailable')):
            with self.assertRaises(OSError):
                learning.capture_receipt(self.receipt(), MODEL)
        ident = learning.capture_receipt(self.receipt('retry'), MODEL)
        self.flush(learning)
        self.assertEqual(learning.state['jobs'][ident]['status'], 'completed')
        self.assertEqual(len(self.memory.events), 1)

    def test_duplicate_skips_model_calls_and_explicit_reprocess_is_still_host_limited(self):
        class ModelMemory:
            max_model_calls = 1
            def process(self, event, model):
                response = model.generate(GenerationRequest('decision_maker', model.model, '{}'))
                return response.text
        self.memory = ModelMemory()
        learning = self.learning(max_calls=1)
        learning.capture_receipt(self.receipt(), MODEL)
        self.flush(learning)
        duplicate = learning.capture_receipt(self.receipt('duplicate'), MODEL)
        self.flush(learning)
        explicit = learning.capture_receipt(self.receipt('explicit', reprocess=True), MODEL)
        self.flush(learning)
        self.assertEqual(learning.state['jobs'][duplicate]['status'], 'archived')
        self.assertEqual(learning.state['attempted'], 1)
        self.assertEqual(len(self.backend.requests), 1)
        self.assertEqual(learning.state['jobs'][explicit]['status'], 'budget_exhausted')

    def test_failed_processing_needs_explicit_reprocess_instead_of_automatic_replay(self):
        class FailingMemory(RecordedMemory):
            def process(self, event, model):
                self.events.append(event)
                raise OSError('fixture product failed')
        self.memory = FailingMemory()
        learning = self.learning()
        original = learning.capture_receipt(self.receipt(), MODEL)
        self.flush(learning)
        self.assertEqual(learning.state['jobs'][original]['status'], 'failed')
        learning.capture_receipt(self.receipt('duplicate'), MODEL)
        self.flush(learning)
        self.assertEqual(len(self.memory.events), 1)
        learning.capture_receipt(self.receipt('explicit', reprocess=True), MODEL)
        self.flush(learning)
        self.assertEqual(len(self.memory.events), 2)


class DeliveryContractTests(unittest.TestCase):
    def test_large_result_views_keep_canonical_archive_identity_with_and_without_metadata(self):
        archive = OperationSpec('research_archive', 'Read original receipt', {
            'type': 'object', 'properties': {'collection': {'enum': ['history', 'operation_results']}}}, True)
        for delivery in (None, FeedbackDelivery('producer:terminal')):
            with self.subTest(delivery=delivery):
                result = OperationResult('Large native result', (), False,
                    data={'stdout': 'x' * 50000}, feedback_delivery=delivery)
                original = result.payload()
                history = HistoryEntry('record', 'operation', None, result.summary, False,
                                       details={'result': original})
                context = DecisionContext('fixed objective', (), (archive,), (result,), history=(history,))
                visible = bounded_evidence_view(context).operation_results[0]
                self.assertLess(len(json.dumps(visible.payload()).encode()), 8192)
                card = visible.data['evidence_card']
                self.assertEqual(card['sha256'], result_id(original))
                self.assertEqual(card['read_path']['record_id'], result_id(original))
                self.assertEqual(visible.feedback_delivery, delivery)
                self.assertEqual(context.operation_results[0].payload(), original)
                self.assertEqual(history.details['result'], original)

    def test_optional_delivery_is_checked_and_supported_by_the_explicit_snapshot_codec(self):
        for event, reprocess in (('', False), ('   ', False), ('event', 1)):
            with self.subTest(event=event), self.assertRaises((TypeError, ValueError)):
                FeedbackDelivery(event, reprocess)
        with self.assertRaises(TypeError):
            OperationResult('result', (), False, feedback_delivery={'event_id': 'untyped'})
        result = OperationResult('result', (), False, feedback_delivery=FeedbackDelivery('producer:event'))
        self.assertEqual(decode(encode(result)), result)
        legacy = OperationResult('legacy', (), True)
        encoded = encode(legacy)
        del encoded['fields']['feedback_delivery']
        restored = decode(encoded)
        expected = asdict(legacy)
        del expected['feedback_delivery']
        self.assertEqual(restored.payload(), expected)
        self.assertNotIn('feedback_delivery', ExecutionReceipt.from_operation('one', '',
            OperationCall('any_tool', {}), restored).payload())
        with self.assertRaisesRegex(ValueError, 'delivery identities differ'):
            ExecutionReceipt('task', '', 'operation', {}, result.payload(),
                             feedback_delivery=FeedbackDelivery('wrong-event'))

    def test_only_provider_declared_tool_errors_are_converted_by_the_lab_adapter(self):
        class Provider:
            error = None
            def describe(self):
                return (OperationSpec('arbitrary', '', {'type': 'object'}),)
            def execute(self, call):
                raise self.error
        provider = Provider()
        tools = LabExecutionTools(provider)
        call = OperationCall('arbitrary', {})
        provider.error = LabToolError('repair this input')
        result = tools.execute(call)
        self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertIn('repair this input', result.error)
        for error in (PermissionError('denied'), OSError('store'), ValueError('identity'), TypeError('provider bug')):
            provider.error = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                tools.execute(call)
