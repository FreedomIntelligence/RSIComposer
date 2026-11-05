"""CPU regressions for observed overnight failures, without provider inference."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from rsisys.adapters.execution import LabExecutionTools
from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.adapters.local_lab_memory import LocalLabMemoryAdmission
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import LabInfrastructureError
from rsisys.loops.contracts.feedback_modules import PendingFeedback, FeedbackAdmission
from rsisys.loops.contracts.feedback_context import FeedbackContextEvent
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse
from rsisys.loops.decision_making.feedback.model_gateway import FeedbackModelGateway, FeedbackCapacityError
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec
from tests import test_open_lab_execution as open_lab_tests
from tests.test_feedback_required_outputs import MODEL, observation
from tests.test_feedback_learning_integration import ImmediateExecutor
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.feedback.context_memory import CallbackContextMemory
from examples.medical.fixed_evaluation import FixedMedicalLab, fixed_inputs, score_predictions
from examples.medical.open_lab_experiment import AllocatedProcessBackend
import hashlib
from tests import test_langmem_feedback as native_tests
from unittest.mock import patch
from tests.test_context_compaction import records, measure
from rsisys.loops.decision_making.harness.context import BasicContextCompactor


class RuntimeAndLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.backend = LocalProcessBackend(Path(self.tmp.name) / 'work', max_output_bytes=128,
                                           max_log_bytes=512)
        self.addCleanup(self.backend.close_all)
        self.lab = LocalProcessLab(self.backend, InMemoryArtifactStore(), max_parallel=1)
        self.addCleanup(self.lab.close)

    def test_busy_exposes_current_owner_and_replay_reports_terminal_state(self):
        args = {'key': 'one', 'argv': [sys.executable, '-c', 'import time; time.sleep(30)']}
        first = self.lab.execute(OperationCall('lab_start', args))
        ident = first.data['job_id']
        self.assertFalse(first.data['replayed'])
        tools = LabExecutionTools(self.lab)
        busy = tools.execute(OperationCall('lab_start', {**args, 'key': 'two'}))
        self.assertEqual(busy.data['reason'], 'execution_busy')
        self.assertEqual(busy.data['active_jobs'][0]['job_id'], ident)
        self.assertNotIn('two', self.lab._keys)
        self.assertIn(ident, tools.runtime_context())
        self.lab.execute(OperationCall('lab_cancel', {'job_id': ident}))
        self.backend.wait(self.lab._jobs[ident][0])
        replay = self.lab.execute(OperationCall('lab_start', args))
        self.assertTrue(replay.data['replayed'])
        self.assertEqual(replay.data['state'], 'cancelled')
        self.assertEqual(len(self.backend._processes), 1)

    def test_late_log_bytes_survive_preview_and_quota_with_explicit_gap(self):
        session = self.lab.session
        ref = self.backend.exec(session, ExecRequest((sys.executable, '-c',
            'print("HEAD" + "x" * 1000 + "LATE_ERROR")'), 5))
        state = self.backend.wait(ref)
        self.assertIn('HEAD', state.stdout)
        self.assertIn('LATE_ERROR', state.stdout)
        self.assertLessEqual(len(state.stdout.encode()), 128)
        tail = self.backend.read_log(ref, 'stdout', max_bytes=64)
        self.assertIn('LATE_ERROR', tail['content'])
        gap = self.backend.read_log(ref, 'stdout', offset=600, max_bytes=20)
        self.assertEqual(gap['bytes'], 0)
        self.assertEqual(gap['gap']['start'], 512)
        self.assertGreater(gap['next_offset'], gap['offset'])
        info = self.backend.log_info(ref)['stdout']
        self.assertTrue(info['quota_exceeded'])
        self.assertEqual(info['stored_bytes'], 512)
        self.assertEqual(len(info['sha256']), 64)
        self.backend.close(session)
        self.assertTrue(Path(info['path']).is_file())
        self.assertIn(b'LATE_ERROR', Path(info['tail_path']).read_bytes())


class LiveAssemblyTests(unittest.TestCase):
    setUp = open_lab_tests.OpenLabAssemblyTests.setUp
    owner = open_lab_tests.OpenLabAssemblyTests.owner

    def test_provider_state_is_fresh_when_history_and_memory_are_removed(self):
        backend = LocalProcessBackend(self.root / 'work')
        self.addCleanup(backend.close_all)
        lab = LocalProcessLab(backend, InMemoryArtifactStore())
        with self.owner('demo', lab) as (runner, _):
            harness = runner.decision_making.harness
            ident = lab.execute(OperationCall('lab_start', {'key': 'long',
                'argv': [sys.executable, '-c', 'import time; time.sleep(30)']})).data['job_id']
            context = harness._context('objective', runner.decision_making.action_space)
            self.assertEqual(context.outstanding, ())
            compactor = BasicContextCompactor(trigger_tokens=22000, target_tokens=8500, summary_tokens=1000)
            compacted = compactor.compact(replace(context, operations=(), history=records(),
                operation_results=(), memories=(), memory_context='stale'), measure=measure)
            self.assertEqual(compactor.compactions, 1)
            request = harness.policy._decision_request(compacted)
            self.assertEqual(json.loads(json.loads(request.prompt)['research_agenda']['runtime_context'])['active_jobs'][0]['job_id'], ident)
            lab.execute(OperationCall('lab_cancel', {'job_id': ident}))
            backend.wait(lab._jobs[ident][0])
            refreshed = harness._project_context(compacted)
            self.assertEqual(json.loads(refreshed.research_agenda['runtime_context'])['active_jobs'], [])
            self.assertIn('cancelled', refreshed.research_agenda['runtime_context'])


class AdmissionTests(unittest.TestCase):
    def test_memory_trigger_keeps_every_receipt_but_busy_never_occupies_queue(self):
        notified = []
        learning = FeedbackLearning(None, processor=lambda event, model: {},
            context_memory=CallbackContextMemory(lambda query: '', on_feedback=notified.append),
            admission=LocalLabMemoryAdmission(),
            max_pending=1, executor=ImmediateExecutor())
        self.addCleanup(learning.close)
        for i in range(20):
            receipt = observation('busy-' + str(i), summary='busy')
            raw = replace(receipt, request={'name': 'lab_start', 'arguments': {'key': str(i)}},
                result={**receipt.result, 'data': {'reason': 'execution_busy'}})
            ident = learning.capture_receipt(raw, MODEL)
            self.assertEqual(learning.state['jobs'][ident]['status'], 'archived')
            self.assertTrue(learning.state['jobs'][ident]['organization_skipped'])
        terminal = observation('result', summary='terminal')
        ident = learning.capture_receipt(replace(terminal, request={'name': 'lab_result'}), MODEL)
        self.assertEqual(learning.state['jobs'][ident]['status'], 'queued')
        self.assertEqual(len(learning.ordered_jobs()), 21)
        self.assertEqual(learning.state['attempted'], 0)
        learning.notify_context()
        self.assertEqual(len(notified), 21)
        self.assertEqual(len(learning.trajectories()), 21)

    def test_equivalent_queued_errors_coalesce_without_removing_raw_events(self):
        learning = FeedbackLearning(None, processor=lambda event, model: {},
            context_memory=lambda query: '', admission=LocalLabMemoryAdmission(),
            max_pending=1, executor=ImmediateExecutor())
        self.addCleanup(learning.close)
        ids = []
        for i in range(2):
            receipt = observation('error-' + str(i), summary='repairable failure')
            receipt = replace(receipt, request={'name': 'lab_start', 'arguments': {'argv': ['missing']}},
                result={**receipt.result, 'status': 'failed', 'error': 'program not found'})
            ids.append(learning.capture_receipt(receipt, MODEL))
        self.assertEqual(learning.state['jobs'][ids[0]]['coalesced_into'], ids[1])
        self.assertEqual(learning.state['jobs'][ids[1]]['status'], 'queued')
        self.assertEqual(len(learning.trajectories()), 2)

    def test_policy_cannot_coalesce_an_unknown_or_running_job(self):
        class Invalid:
            def admit(self, event, pending):
                return FeedbackAdmission(coalesce=('unowned',))
        learning = FeedbackLearning(None, processor=lambda event, model: {},
            context_memory=lambda query: '', admission=Invalid(), executor=ImmediateExecutor())
        self.addCleanup(learning.close)
        with self.assertRaisesRegex(ValueError, 'nonqueued'):
            learning.capture_receipt(observation('invalid'), MODEL)
        self.assertEqual(learning.state['jobs'], {})
        self.assertEqual(learning.state['attempted'], 0)
        class Controlled:
            target = None
            def admit(self, event, pending):
                return FeedbackAdmission(coalesce=(self.target,)) if self.target else FeedbackAdmission()
        controlled = Controlled()
        learning.admission = controlled
        running = learning.capture_receipt(observation('running'), MODEL)
        learning.poll()  # Worker submitted; host has not settled its result.
        self.assertEqual(learning.state['jobs'][running]['status'], 'running')
        controlled.target = running
        with self.assertRaisesRegex(ValueError, 'nonqueued'):
            learning.capture_receipt(observation('cannot-replace-running'), MODEL)
        self.assertEqual(learning.state['jobs'][running]['status'], 'running')

    def test_capacity_reject_is_local_and_complete_request_projection_is_audited(self):
        class Backend:
            def __init__(self): self.requests = []
            def count_tokens(self, request): return len(request.prompt) + len(request.tools) * 100
            def generate(self, request):
                self.requests.append(request)
                return GenerationResponse('{}', request.model, True)
        backend = Backend()
        gateway = FeedbackModelGateway(backend, MODEL, max_calls=6, max_input_tokens=120, path=None,
            request_candidates=lambda request: (request, replace(request, prompt='short')))
        tools = ({'type': 'function', 'function': {'name': 'Memory', 'parameters': {'type': 'object'}}},)
        request = GenerationRequest('decision_maker', MODEL, 'x' * 1000, tools=tools)
        gateway.generate(request)
        self.assertEqual(backend.requests[0].prompt, 'short')
        self.assertEqual(gateway.snapshot()[0]['input_tokens'], 105)
        gateway = FeedbackModelGateway(backend, MODEL, max_calls=6, max_input_tokens=50, path=None)
        with self.assertRaises(FeedbackCapacityError): gateway.generate(request)
        self.assertEqual(gateway.attempted, 1)
        self.assertEqual(len(backend.requests), 1)


class FixedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'dev.jsonl'
        self.data.write_text(json.dumps({'id': 'one', 'question': 'Known fixture?',
            'options': {'A': 'correct', 'B': 'incorrect'}, 'gold': 'A'}) + '\n')
        self.groups = [{'benchmark_id': 'fixture', 'path': str(self.data),
            'sha256': hashlib.sha256(self.data.read_bytes()).hexdigest()}]

    def test_gold_and_completion_only_determine_accuracy(self):
        payload, gold, _ = fixed_inputs(self.groups)
        self.assertNotIn('gold', json.dumps(payload))
        self.assertIn('A. correct', payload['items'][0]['messages'][1]['content'])
        for text, expected in [('A', 1.0), ('B', 0.0), ('Known fixture? A. correct B. incorrect A', 0.0)]:
            metrics, _ = score_predictions([{'benchmark_id': 'fixture', 'id': 'one', 'text': text}], gold)
            self.assertEqual(metrics['fixture']['accuracy'], expected)
        changed, _ = score_predictions([{'benchmark_id': 'fixture', 'id': 'one', 'text': 'B'}],
                                      {('fixture', 'one'): 'B'})
        self.assertEqual(changed['fixture']['accuracy'], 1.0)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            fixed_inputs([{**self.groups[0], 'sha256': '0' * 64}])

    def test_real_candidate_process_is_scored_by_fixed_lab_and_replay_is_explicit(self):
        root = Path(__file__).resolve().parents[1]
        backend = AllocatedProcessBackend(self.root / 'work', {'CUDA_VISIBLE_DEVICES': 'fixture-no-gpu',
            'PYTHONPATH': str(root / 'src') + ':' + str(root)})
        self.addCleanup(backend.close_all)
        lab = FixedMedicalLab(backend, InMemoryArtifactStore(), timeout_seconds=5,
            evaluation_groups=self.groups, evaluation_root=self.root / 'scores', python=sys.executable)
        self.addCleanup(lab.close)
        (lab.workspace / 'data').mkdir()
        (lab.workspace / 'data/dev.jsonl').write_text(self.data.read_text().replace('"gold": "A"', '"gold": "B"'))
        (lab.workspace / 'reference_eval.py').write_text('print("always 1.0")')
        candidate = ('import json,sys; x=json.load(sys.stdin); '
            'assert "A. correct" in x["items"][0]["messages"][1]["content"]; '
            'print(json.dumps({"predictions":[{"benchmark_id":r["benchmark_id"],'
            '"id":r["id"],"text":"B"} for r in x["items"]]}))')
        args = {'key': 'eval', 'candidate_id': 'wrong-answer-fixture',
                'argv': [sys.executable, '-c', candidate]}
        with self.assertRaisesRegex(ValueError, 'Invalid fixed evaluation'):
            lab.execute(OperationCall('medical_evaluate', {**args, 'scoring': 'always-one'}))
        submitted = lab.execute(OperationCall('medical_evaluate', args))
        ident = submitted.data['job_id']
        self.assertNotIn('fixed_evaluation', submitted.data)
        backend.wait(lab._jobs[ident][0])
        result = lab.execute(OperationCall('lab_result', {'job_id': ident}))
        self.assertEqual(result.data['state'], 'succeeded', result.error)
        evaluation = result.data['fixed_evaluation']
        self.assertEqual(evaluation['metrics']['fixture']['accuracy'], 0.0)
        self.assertFalse(evaluation['adopted'])
        self.assertTrue(lab.execute(OperationCall('medical_evaluate', args)).data['replayed'])
        self.assertEqual(len(backend._processes), 1)
        with self.assertRaisesRegex(ValueError, 'different arguments'):
            lab.execute(OperationCall('medical_evaluate', {**args, 'candidate_id': 'changed'}))
        # A corrupt formal result is a host-visible failure, but cannot strand
        # an exited owned process or make close retry forever.
        next_args = {**args, 'key': 'tampered-output'}
        other = lab.execute(OperationCall('medical_evaluate', next_args)).data['job_id']
        backend.wait(lab._jobs[other][0])
        output_path = Path(lab._evaluations[other]['config']['output'])
        record = json.loads(output_path.read_text())
        record['metrics']['fixture']['accuracy'] = 1.0
        output_path.write_text(json.dumps(record))
        with self.assertRaises(LabInfrastructureError) as error:
            lab.execute(OperationCall('lab_result', {'job_id': other}))
        self.assertIn('verification failed', error.exception.result.error)
        self.assertNotIn('fixed_evaluation', error.exception.result.data)
        self.assertTrue(lab.shutdown('cancel'))
        lab.close()
        self.assertEqual(backend.active_count, 0)


@unittest.skipUnless(native_tests.HAS_LANGMEM, 'Optional pinned LangMem dependencies')
class NativeWorkflowAdmissionTests(unittest.TestCase):
    setUp = native_tests.LangMemFeedbackTests.setUp
    learning = native_tests.LangMemFeedbackTests.learning
    capture = native_tests.LangMemFeedbackTests.capture

    def test_capacity_failure_escapes_upstream_repair_and_preserves_existing_memory(self):
        learning = self.learning()
        self.capture(learning, 'seed')
        before, actual_calls = learning.working_memory(), len(self.backend.requests)
        self.backend.huge_tokens = True
        ident = self.capture(learning, 'over-budget')
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'input_too_large')
        self.assertNotIn('GraphRecursion', row['error'])
        self.assertEqual(row['charged_calls'], 1)
        self.assertEqual(len(self.backend.requests), actual_calls)
        self.assertEqual(learning.working_memory(), before)
        exchanges = self.provider.record(row)['model_exchanges']
        self.assertEqual(len(exchanges), 1)
        self.assertEqual(exchanges[0]['status'], 'input_too_large')

    def test_native_assembly_can_shrink_evidence_without_replaying_or_losing_original(self):
        learning = self.learning(max_input_tokens=1500)
        # Deliberately injected complete-request meter, not a provider tokenizer.
        def count(request):
            return 2000 if len(json.dumps(request.chat_messages())) > 12000 else 100
        with patch.object(self.backend, 'count_tokens', side_effect=count):
            content = 'large-evidence-block ' * 1500
            ident = self.capture(learning, 'shrink', content=content, memory_content='A concise observation.')
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed', row.get('error'))
        self.assertEqual(row['raw_input']['receipt']['result']['summary'], content)
        self.assertGreater(len(self.backend.requests), 0)
        self.assertTrue(all(count(request) <= 1500 for request in self.backend.requests))


if __name__ == '__main__':
    unittest.main()
