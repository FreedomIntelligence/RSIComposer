"""Public receipts, optional controls and Lab adapters, with no model service."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from rsisys.adapters.lab import ExplorationLab
from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import ExplorationProgress, LoopStepResult, OperationResult
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lab import LabSpec, LabTask
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.metrics import InvalidMetrics
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.receipts import ExecutionReceipt, receipt_facts
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions, builtin_directives
from rsisys.loops.decision_making.feedback import FeedbackEvidence
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.harness.commands import CommandHandler
from rsisys.runtime.run_policy import RunDecision, TimeRunPolicy
from rsisys.runtime.runner import Runner
from tests.test_feedback_learning_integration import Backend, ImmediateExecutor


CONFIG = DemoConfig.load(Path('configs/demo.json'))
MODEL = ModelRef('fixture', 'meta-agent', 'v1')


class Script:
    model = MODEL

    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return next(self.decisions)


class MinimalHarness:
    def __init__(self, kind='progressed'):
        self.calls, self.kind = 0, kind

    def step(self, objective, action_space):
        self.calls += 1
        return LoopStepResult(self.kind)

    def pending(self):
        return None

    def receive_feedback(self, directive, feedback):
        raise AssertionError('No task was dispatched')


class MetricContractsTests(unittest.TestCase):
    def test_all_feedback_groups_reject_invalid_measurements_before_publication(self):
        for field in ('quality_metrics', 'training_metrics', 'execution_metrics'):
            for values in ({'metric': float('nan')}, {'metric': float('inf')},
                           {'metric': -float('inf')}, {'metric': True}, {'metric': '0.5'},
                           {'metric': 10 ** 1000}, {'': 1}, {1: 1}, None):
                with self.subTest(field=field, values=values), self.assertRaises(InvalidMetrics):
                    Feedback('task', ExecutionStatus.COMPLETED, 'raw', True, **{field: values})

    def test_provider_boundary_produces_deliverable_failed_receipt_with_original_sources(self):
        raw = {'accuracy': float('nan')}
        feedback = Feedback.from_backend('task', ExecutionStatus.COMPLETED, 'provider output', True,
            quality_metrics=raw, training_metrics={'loss': 0.125},
            evidence_refs=('raw-provider-output',), adopted_candidate_ids=('candidate',))
        self.assertEqual(feedback.status, ExecutionStatus.FAILED)
        self.assertEqual(feedback.quality_metrics, {})
        self.assertEqual(feedback.training_metrics, {'loss': 0.125})
        self.assertEqual(feedback.evidence_refs, ('raw-provider-output',))
        self.assertEqual(feedback.adopted_candidate_ids, ())
        self.assertIn("quality_metrics['accuracy']", feedback.error)
        self.assertIn('nan', feedback.error)
        self.assertNotEqual(raw['accuracy'], raw['accuracy'])
        json.dumps(asdict(feedback), allow_nan=False)
        with self.assertRaises(TypeError):
            Feedback.from_backend('task', 'completed', 'raw', True, quality_metrics=raw)

    def test_metric_inputs_are_detached_and_zero_is_preserved(self):
        metrics = {'accuracy': 0, 'loss': 0.125}
        feedback = Feedback('task', ExecutionStatus.COMPLETED, 'raw', True, quality_metrics=metrics)
        metrics['accuracy'] = float('nan')
        self.assertEqual(feedback.quality_metrics, {'accuracy': 0, 'loss': 0.125})

    def test_custom_handler_failure_can_be_delivered_retried_and_cancelled(self):
        class Handler(CommandHandler):
            calls = 0

            def prepare(self, harness, state):
                pass

            def advance(self, harness, state, action_space):
                self.calls += 1
                return Feedback.from_backend(state.directive.directive_id, ExecutionStatus.COMPLETED,
                    'external provider', True, quality_metrics={'accuracy': float('inf')},
                    evidence_refs=('external-raw-output',))

        handler = Handler()

        def factory(context):
            from rsisys.assembly import build_default_exploration
            from rsisys.loops.exploration.harness.commands import ExplorationCommands
            loop = build_default_exploration(context)
            loop.harness.commands = ExplorationCommands(loop.harness, {'execute': handler})
            context.exploration_directives = DirectiveDefinitions(
                item for item in builtin_directives() if item.spec.mode == 'execute')
            return loop

        with TemporaryDirectory() as directory:
            config = replace(ApplicationConfig.load(Path('configs/local.json')),
                             data_dir=Path(directory), execution='mock')
            with Application(config, exploration_factory=factory) as app:
                directive = Directive('external-task', 'objective', 'check provider')
                app.runner.decision_making.harness.policy = Script(Decision('explore', directive=directive))
                self.assertEqual(app.runner.run('objective', 1).rounds_completed, 1)
                loop = app.runner.decision_making
                feedback = loop.harness.state.last_feedback
                self.assertEqual(feedback.status, ExecutionStatus.FAILED)
                loop.receive_feedback(directive, feedback)
                self.assertEqual(app.runner.exploration.cancel(directive.directive_id).feedback, feedback)
                self.assertEqual(handler.calls, 1)
                self.assertIsNone(loop.pending())

    def test_bad_tool_receipt_is_recorded_as_failure_without_poisoning_pending_record(self):
        for status in ('completed', JobStatus.FAILED):
            with self.subTest(status=status):
                runner, _ = build_demo(CONFIG)
                with runner:
                    loop = runner.decision_making
                    loop.harness.policy = Script(Decision('invoke',
                        operation=OperationCall('prepare_data', {'text': 'side effect once'})), Decision('finish'))
                    tool = Mock(side_effect=lambda call: OperationResult('raw result', (), True,
                        status=status, error='error' if status == JobStatus.FAILED else None))
                    loop.harness.tools = type('Tools', (), {'execute': staticmethod(tool)})()
                    with self.assertRaises(TypeError):
                        loop.step('objective')
                    self.assertIsNone(loop.harness.state.pending_operation_record)
                    self.assertEqual(loop.harness.state.operation_results[-1].status, ExecutionStatus.FAILED)
                    self.assertEqual(loop.step('objective').kind, 'finished')
                    tool.assert_called_once()

    def test_operation_structured_metrics_and_data_must_be_persistable(self):
        for data in ({'quality_metrics': {'accuracy': float('nan')}}, {'raw': float('inf')},
                     {'training_metrics': {'loss': True}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                OperationResult('raw', (), True, data=data)
        with self.assertRaisesRegex(ValueError, 'cannot carry an error'):
            OperationResult('raw', (), True, error='')
        result = OperationResult('raw', (), True)
        self.assertEqual(ExecutionReceipt.from_operation('task', 'objective',
            OperationCall('execute_code', {}), result).result['status'], ExecutionStatus.COMPLETED)


class OptionalControlTests(unittest.TestCase):
    def test_minimal_harness_runs_under_default_time_and_custom_policy_and_host_stops_it(self):
        class CustomPolicy:
            def evaluate(self, context):
                return RunDecision()

        for policy in (None, TimeRunPolicy(60), CustomPolicy()):
            with self.subTest(policy=type(policy).__name__):
                harness = MinimalHarness()
                loop = DecisionMakingLoop(harness, DemoDecisionActionSpace())
                clock = [0]
                runner = Runner(loop, type('EmptyLab', (), {'pending': lambda self: None})(),
                                run_policy=policy, clock=lambda: clock[0])
                self.assertEqual(loop.run_control_capabilities(), frozenset())
                self.assertEqual(runner.step('objective').kind, 'progressed')
                self.assertEqual(harness.calls, 1)
                if isinstance(policy, TimeRunPolicy):
                    clock[0] = 61
                    self.assertEqual(runner.step('objective').kind, 'finished')
                    self.assertEqual(harness.calls, 1)

    def test_hints_and_restart_are_discovered_independently(self):
        class Hints(MinimalHarness):
            def update_run_control(self, context):
                self.hints = context

        harness = Hints()
        loop = DecisionMakingLoop(harness, DemoDecisionActionSpace())
        runner = Runner(loop, object(), run_policy=TimeRunPolicy(60))
        self.assertEqual(loop.run_control_capabilities(), {'update_run_control'})
        self.assertEqual(runner.step('objective').kind, 'progressed')
        self.assertIsInstance(harness.hints, dict)
        harness.kind = 'finished'
        self.assertEqual(runner.step('objective').kind, 'waiting')
        runner._restart_at = 0
        with self.assertRaisesRegex(NotImplementedError, 'restart_research'):
            runner.step('objective')


class LabContractTests(unittest.TestCase):
    def task(self, **changes):
        return replace(LabTask('task', 'decision', 'lab', 'submit-key',
            Directive('task', 'subtask objective', 'execute once'), run_id='run', agent_id='researcher',
            issuing_model=MODEL), **changes)

    def test_existing_exploration_has_lab_identity_lifecycle_and_idempotent_submission(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            endpoint = runner.exploration
            endpoint.accept = Mock(wraps=endpoint.accept)
            lab = ExplorationLab(LabSpec('lab', 'fixed objective', ('execute',)), endpoint)
            task = self.task()
            self.assertEqual(lab.describe().objective, 'fixed objective')
            self.assertFalse(lab.submit(task).terminal)
            self.assertFalse(lab.submit(task).terminal)
            endpoint.accept.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'terminal'):
                lab.result(task.task_id)
            self.assertTrue(lab.advance(task.task_id).terminal)
            receipt = lab.result(task.task_id)
            self.assertEqual(receipt.request, asdict(task.directive))
            self.assertEqual(receipt.result, asdict(endpoint.inspect(task.task_id).feedback))
            self.assertEqual((receipt.task_id, receipt.decision_id, receipt.lab_id, receipt.run_id, receipt.agent_id),
                             ('task', 'decision', 'lab', 'run', 'researcher'))
            self.assertEqual(receipt.issuing_model, MODEL)
            self.assertEqual(lab.cancel(task.task_id).feedback.status, ExecutionStatus.COMPLETED)
            lab.acknowledge(task.task_id)

    def test_wrong_lab_unsupported_mode_and_reused_identity_are_rejected_before_execution(self):
        endpoint = Mock()
        task = self.task()
        endpoint.accept.return_value = ExplorationProgress('task', 'accepted', task.directive)
        lab = ExplorationLab(LabSpec('lab', 'fixed objective', ('execute',)), endpoint)
        for bad in (replace(task, lab_id='other'),
                    replace(task, directive=replace(task.directive, mode='sft'))):
            with self.assertRaises(ValueError):
                lab.submit(bad)
        endpoint.accept.assert_not_called()
        lab.submit(task)
        for bad in (replace(task, decision_id='other'),
                    replace(task, task_id='other', directive=replace(task.directive, directive_id='other'))):
            with self.assertRaises(ValueError):
                lab.submit(bad)
        endpoint.accept.assert_called_once()

    def test_mismatched_endpoint_progress_and_early_acknowledgement_are_rejected(self):
        endpoint = Mock()
        task = self.task()
        endpoint.accept.return_value = ExplorationProgress('task', 'accepted', task.directive)
        endpoint.inspect.return_value = ExplorationProgress('other', 'accepted',
            replace(task.directive, directive_id='other'))
        lab = ExplorationLab(LabSpec('lab', 'fixed objective', ('execute',)), endpoint)
        lab.submit(task)
        with self.assertRaisesRegex(ValueError, 'different directive'):
            lab.acknowledge('task')
        endpoint.acknowledge.assert_not_called()


class ReceiptObservationTests(unittest.TestCase):
    def install(self, runner):
        backend = Backend()
        loop = runner.decision_making
        enable_feedback_learning(loop, backend=backend, executor=ImmediateExecutor())
        self.addCleanup(loop.feedback_service.close)
        return loop, backend

    def test_local_write_is_observed_and_feedback_navigation_does_not_observe_itself(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop, backend = self.install(runner)
            loop.harness.policy = Script(
                Decision('invoke', decision_id='write', operation=OperationCall('prepare_data', {'text': 'code output'})),
                Decision('invoke', operation=OperationCall('feedback_learning_read', {})), Decision('finish'))
            first = loop.step('objective')
            learning = loop.feedback_service.learning
            self.assertEqual(learning.state['attempted'], 1)
            row = next(iter(learning.state['jobs'].values()))
            raw = row['raw_input']['receipt']
            self.assertEqual(raw['operation'], {'name': 'prepare_data', 'arguments': {'text': 'code output'}})
            self.assertEqual(raw['result'], asdict(first.operation_result))
            self.assertEqual(raw['task']['decision_id'], 'write')
            self.assertNotIn('directive', raw)
            self.assertEqual(row['raw_input']['research_context']['issuing_decision_model'], asdict(MODEL))
            self.assertEqual(loop.step('objective').kind, 'progressed')
            self.assertEqual(len(learning.state['jobs']), 1)
            self.assertEqual(learning.state['attempted'], 1)
            self.assertEqual(loop.step('objective').kind, 'finished')
            self.assertEqual(len(backend.requests), 1)

    def test_observation_commit_response_loss_retries_receipt_without_reexecuting_tool(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop, backend = self.install(runner)
            policy = Script(Decision('invoke', operation=OperationCall('prepare_data', {'text': 'once'})))
            loop.harness.policy = policy
            execute = loop.harness.tools.execute
            loop.harness.tools.execute = Mock(wraps=execute)
            capture = loop.feedback_service.learning.capture_receipt
            remaining = [1]

            def lose_response(*args, **kwargs):
                ident = capture(*args, **kwargs)
                if remaining[0]:
                    remaining[0] = 0
                    raise OSError('observation commit response lost')
                return ident

            loop.feedback_service.learning.capture_receipt = lose_response
            with self.assertRaises(OSError):
                loop.step('objective')
            self.assertIsNotNone(loop.harness.state.pending_operation_record)
            self.assertEqual(loop.step('objective').kind, 'progressed')
            self.assertIsNone(loop.harness.state.pending_operation_record)
            loop.harness.tools.execute.assert_called_once()
            self.assertEqual(policy.calls, 1)
            self.assertEqual(len(loop.feedback_service.learning.state['jobs']), 1)
            self.assertEqual(len(backend.requests), 1)

    def test_denied_feedback_navigation_does_not_observe_itself(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop, backend = self.install(runner)
            loop.action_space.base.permissions = ActionPermissions(operations=frozenset())
            loop.harness.policy = Script(Decision('invoke', decision_id='denied-read',
                operation=OperationCall('feedback_learning_read', {})))
            with self.assertRaises(PermissionError):
                loop.step('objective')
            record = loop.harness.state.decision_records['denied-read']
            self.assertEqual(record['status'], 'failed')
            self.assertIn('PermissionError', record['error'])
            self.assertIsNone(loop.harness.state.pending_operation_record)
            self.assertEqual(loop.feedback_service.learning.state['jobs'], {})
            self.assertEqual(backend.requests, [])

    def test_async_submission_is_not_a_completion_and_resume_is_observed_once(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop, backend = self.install(runner)
            loop.harness.policy = Script(Decision('invoke', decision_id='async',
                operation=OperationCall('prepare_data', {'text': 'async work'})))
            job = JobRef('external-job', 'external-backend')
            loop.harness.tools.execute = Mock(return_value=job)
            self.assertEqual(loop.step('objective').kind, 'waiting')
            self.assertEqual(loop.feedback_service.learning.state['jobs'], {})
            result = OperationResult('confirmed completion', (), True, data={'execution_metrics': {'steps': 1}})
            loop.harness.resume(job, result)
            self.assertEqual(len(loop.feedback_service.learning.state['jobs']), 1)
            raw = next(iter(loop.feedback_service.learning.state['jobs'].values()))['raw_input']['receipt']
            self.assertEqual(raw['task']['job'], asdict(job))
            with self.assertRaises(ValueError):
                loop.harness.resume(job, result)
            self.assertEqual(len(backend.requests), 1)

    def test_common_observer_works_without_default_harness_and_namespaces_receipts(self):
        from tests.test_feedback_decoupling import AlienHarness, AlienAdapter
        harness, backend = AlienHarness(), Backend()
        loop = DecisionMakingLoop(harness, object())
        enable_feedback_learning(loop, adapter=AlienAdapter(harness), backend=backend,
                                 executor=ImmediateExecutor())
        self.addCleanup(loop.feedback_service.close)
        result = OperationResult('measured output', (), True,
                                 data={'quality_metrics': {'accuracy': 0}, 'candidate_id': 'unbound'})
        receipt = ExecutionReceipt.from_operation('same-id', 'objective',
            OperationCall('execute_code', {'code': 'fixture'}), result, decision_id='decision')
        ident = loop.feedback_service.observe(receipt, evidence=FeedbackEvidence())
        self.assertEqual(loop.feedback_service.observe(receipt), ident)
        loop.feedback_service.refresh()
        row = loop.feedback_service.learning.state['jobs'][ident]
        self.assertIsNone(row['input'].get('candidate_evidence'))
        self.assertEqual(receipt_facts(row['raw_input']['receipt'])['quality_metrics'], {'accuracy': 0})
        memory = json.loads(harness.summaries[0].content)
        self.assertEqual(memory['receipt_facts']['task_id'], 'same-id')
        self.assertEqual(memory['receipt_facts']['source'], 'operation')
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            loop.feedback_service.observe(replace(receipt, result=asdict(replace(result, summary='changed'))))
        directive = Directive('same-id', 'objective', 'execute')
        feedback = Feedback('same-id', ExecutionStatus.COMPLETED, 'Lab outcome', True)
        for lab_id in ('lab-a', 'lab-b'):
            loop.feedback_service.observe(ExecutionReceipt.from_feedback(directive, feedback,
                decision_id='decision', lab_id=lab_id))
            loop.feedback_service.refresh()
        self.assertEqual(len(loop.feedback_service.learning.state['jobs']), 3)
        self.assertFalse(hasattr(harness, 'policy'))
        self.assertFalse(hasattr(harness, 'state'))

    def test_lab_receipt_reaches_independent_observer_with_exact_issuer_and_task_identity(self):
        from tests.test_feedback_decoupling import AlienHarness, AlienAdapter
        runner, _ = build_demo(CONFIG)
        with runner:
            task = LabContractTests().task()
            lab = ExplorationLab(LabSpec('lab', 'fixed objective', ('execute',)), runner.exploration)
            lab.submit(task)
            lab.advance(task.task_id)
            observer = DecisionMakingLoop(AlienHarness(), object())
            enable_feedback_learning(observer, adapter=AlienAdapter(observer.harness), backend=Backend(),
                                     executor=ImmediateExecutor())
            self.addCleanup(observer.feedback_service.close)
            ident = observer.feedback_service.observe(lab.result(task.task_id))
            observer.feedback_service.refresh()
            row = observer.feedback_service.learning.state['jobs'][ident]
            identity = row['raw_input']['receipt']['identity']
            self.assertEqual(identity['lab_id'], 'lab')
            self.assertEqual(identity['decision_id'], task.decision_id)
            self.assertEqual(identity['issuing_model'], asdict(MODEL))
            self.assertEqual(row['raw_input']['research_context']['issuing_decision_model'], asdict(MODEL))
            self.assertNotEqual(row['model'], asdict(MODEL))
            self.assertEqual(row['status'], 'completed')

    def test_operation_projection_is_explicit_and_original_output_is_readable(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop, _ = self.install(runner)
            receipt = ExecutionReceipt.from_operation('large-output', 'objective',
                OperationCall('execute_code', {'code': 'print result'}),
                OperationResult('output summary', (), True, data={
                    'stdout': 'prefix-' + 'x' * 10000 + '-tail', 'quality_metrics': {'score': 0}}))
            ident = loop.feedback_service.observe(receipt, evidence=FeedbackEvidence())
            learning = loop.feedback_service.learning
            row = learning.state['jobs'][ident]
            original = row['raw_input']['receipt']['result']['data']['stdout']
            self.assertEqual(original, receipt.result['data']['stdout'])
            self.assertEqual(row['input']['receipt']['result']['data']['quality_metrics'], {'score': 0})
            descriptors = row['input']['evidence_packaging']['records']
            self.assertTrue(any(r['raw_ref'] == '/receipt' and r['coverage'] == 'excerpt'
                                for r in descriptors))
            pointer, offset, chunks = '/receipt/result/data/stdout', 0, []
            while True:
                page = learning.read(ident, evidence_path=pointer, offset=offset, limit=16)
                chunks.append(page['text'])
                if page['next_offset'] is None:
                    break
                offset = page['next_offset']
            self.assertEqual(json.loads(''.join(chunks)), original)


if __name__ == '__main__':
    unittest.main()
