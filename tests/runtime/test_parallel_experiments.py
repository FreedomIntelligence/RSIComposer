"""Executable concurrency contracts using deterministic asynchronous mock jobs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
from unittest.mock import Mock

from examples.components.parallel_experiments import BASE, build_demo, run_demo
from rsisys.loops.contracts.coordination import DirectiveUpdate, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.exploration.harness import ConcurrentExplorationHarness
from rsisys.runtime.runner import Runner
from rsisys.snapshot.local import LocalSnapshotCapture


class FinishPolicy:
    def decide(self, context):
        return Decision('finish')


class ParallelExperimentTests(unittest.TestCase):
    def start_two(self):
        runner, children, policy = build_demo()
        runner.step('task')
        runner.step('task')
        runner.decision_making.harness.policy = FinishPolicy()
        runner.step('task')
        self.assertEqual(set(children), {'slow', 'fast'})
        return runner, children, policy

    def drain(self, runner):
        results = []
        for _ in range(80):
            result = runner.step('task')
            results.append(result)
            if result.kind == 'finished':
                return results
        self.fail('runner failed to finish')

    def complete_demo(self):
        runner, children, policy = build_demo()
        self.drain(runner)
        return runner, children, policy

    def test_real_example_standard_apis_update_out_of_order_compare(self):
        report = run_demo()
        self.assertTrue(report['simulated'])
        self.assertEqual(report['max_in_flight'], 2)
        self.assertEqual(report['completion_order'][:2], ['fast', 'slow'])
        self.assertEqual(report['update_version'], 1)
        self.assertEqual([row['candidate_id'] for row in report['ranking']], ['fast', 'slow'])
        self.assertIn('experiment_update', report['tools_used'])
        self.assertIn('experiment_compare', report['tools_used'])

    def test_isolated_mutable_components_and_explicit_same_base(self):
        runner, children, _ = self.start_two()
        left, right = children['slow'], children['fast']
        for name in ('policy', 'state', 'memory', 'environment', 'training', 'evaluation'):
            self.assertIsNot(getattr(left, name), getattr(right, name))
        self.assertEqual(left.policy.model, right.policy.model)
        left.state.instructions = 'changed only slow'
        self.assertNotEqual(left.state.instructions, right.state.instructions)
        self.assertNotEqual(left.learner.submissions[0][1], right.learner.submissions[0][1])
        views = runner.exploration.execution_views()
        self.assertEqual(set(views), {'slow', 'fast'})
        self.assertEqual(views['slow'].model, BASE)
        self.assertNotEqual(views['slow'].progress.job.backend, views['fast'].progress.job.backend)
        with self.assertRaisesRegex(RuntimeError, 'pending_all'):
            runner.exploration.pending()
        with self.assertRaisesRegex(RuntimeError, 'pending_all'):
            runner.decision_making.pending()

    def test_factory_reuse_and_mutable_alias_are_rejected_before_accept(self):
        _, children, _ = self.start_two()
        child = children['slow']
        for alias in (False, True):
            first = children['slow']
            second = children['fast'] if alias else first
            if alias:
                second.memory = first.memory
            values = iter((first, second))
            harness = ConcurrentExplorationHarness(lambda directive: next(values))
            # Use already accepted directives; accept is itself idempotent.
            harness.accept(child.commands.receipts['slow'].directive)
            with self.assertRaisesRegex(ValueError, 'isolation|shared mutable'):
                harness.accept(Directive('new', 'task', ''))

    def test_capacity_duplicate_ids_and_ack_require_terminal(self):
        runner, _, _ = self.start_two()
        harness = runner.exploration.harness
        before = runner.exploration.inspect('slow')
        self.assertEqual(harness.accept(before.directive).directive, before.directive)
        with self.assertRaisesRegex(ValueError, 'different content'):
            harness.accept(replace(before.directive, strategy='other'))
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            harness.accept(Directive('overflow', 'task', ''))
        with self.assertRaisesRegex(RuntimeError, 'unfinished'):
            harness.acknowledge('slow')

    def test_cancel_is_targeted_and_ambiguous_cancel_is_rejected(self):
        runner, children, _ = self.start_two()
        with self.assertRaisesRegex(ValueError, 'directive_id'):
            runner.cancel()
        runner.cancel('slow')
        self.drain(runner)
        self.assertEqual(runner.exploration.inspect('slow').feedback.status, ExecutionStatus.CANCELLED)
        self.assertEqual(runner.exploration.inspect('fast').feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(children['slow'].learner.cancellations), 1)
        self.assertEqual(children['fast'].learner.cancellations, [])
        self.assertFalse(runner.busy)

    def test_terminal_delivery_retry_after_another_completion_never_reexecutes(self):
        for failure_at in ('receive', 'ack'):
            with self.subTest(failure_at=failure_at):
                runner, children, _ = self.start_two()
                receive = runner.decision_making.receive_feedback
                acknowledge = runner.exploration.acknowledge
                failed = []

                def lose_receive(directive, feedback):
                    receive(directive, feedback)
                    if directive.directive_id == 'fast' and not failed:
                        failed.append(True)
                        raise OSError('delivery response lost')

                def lose_ack(ident):
                    acknowledge(ident)
                    if ident == 'fast' and not failed:
                        failed.append(True)
                        raise OSError('ack response lost')

                if failure_at == 'receive':
                    runner.decision_making.receive_feedback = lose_receive
                else:
                    runner.exploration.acknowledge = lose_ack
                for _ in range(10):
                    try:
                        runner.step('task')
                    except OSError:
                        break
                self.assertTrue(failed)
                fast_polls = len(children['fast'].learner.polls)
                slow = runner.exploration.inspect('slow')
                children['slow'].learner.queue._get(slow.job).remaining_polls = 0
                self.drain(runner)
                self.assertEqual(runner.decision_making.harness.state.completed, 2)
                self.assertEqual(len(children['fast'].learner.polls), fast_polls)
                self.assertEqual(len(children['fast'].learner.submissions), 1)
                self.assertEqual(len(children['slow'].learner.submissions), 1)
                self.assertFalse(runner.busy)

    def test_failed_view_notification_after_ack_does_not_leave_dead_slot(self):
        runner, children, _ = self.start_two()
        update = runner.decision_making.update_execution
        failed = []

        def lose(view):
            update(view)
            if (view.progress is None and 'fast' in runner.exploration.harness._acknowledged
                    and not failed):
                failed.append(True)
                raise OSError('view notification lost after ack')

        runner.decision_making.update_execution = lose
        with self.assertRaises(OSError):
            for _ in range(10):
                runner.step('task')
        self.assertNotIn('fast', runner._slots)
        self.drain(runner)
        self.assertEqual(runner.decision_making.harness.state.completed, 2)
        self.assertEqual(len(children['fast'].learner.submissions), 1)

    def test_reconstructed_runner_recovers_unacknowledged_terminal_receipts(self):
        runner, children, _ = self.start_two()
        receive = runner.decision_making.receive_feedback

        def lost(directive, feedback):
            receive(directive, feedback)
            if directive.directive_id == 'fast':
                raise OSError('response lost')

        runner.decision_making.receive_feedback = lost
        with self.assertRaises(OSError):
            for _ in range(8):
                runner.step('task')
        runner.decision_making.receive_feedback = receive
        reconstructed = Runner(runner.decision_making, runner.exploration, parallel=True)
        self.drain(reconstructed)
        self.assertEqual(runner.decision_making.harness.state.completed, 2)
        self.assertEqual(len(children['fast'].learner.submissions), 1)

    def test_out_of_order_feedback_dedup_rejects_changed_receipts(self):
        runner, _, _ = self.complete_demo()
        harness = runner.decision_making.harness
        state = harness.state
        first = deepcopy(state.delivered_feedback['fast'])
        harness.receive_feedback(*first)  # Much later than last_feedback.
        self.assertEqual(state.completed, 4)
        with self.assertRaisesRegex(ValueError, 'original receipt'):
            harness.receive_feedback(first[0], replace(first[1], summary='changed'))
        first[1].quality_metrics['injected'] = 1
        self.assertNotIn('injected', state.delivered_feedback['fast'][1].quality_metrics)

    def test_update_version_idempotency_input_copy_and_terminal_retry(self):
        runner, children, _ = self.start_two()
        update = DirectiveUpdate('slow', 0, 'u1', {'remaining_polls': 1})
        receipt = runner.update(update)
        self.assertEqual(receipt.version, 1)
        self.assertEqual(runner.update(deepcopy(update)), receipt)
        self.assertEqual(runner.exploration.inspect('fast').version, 0)
        self.assertEqual(len(children['slow'].learner.updates), 1)
        with self.assertRaisesRegex(ValueError, 'key reused'):
            runner.update(replace(update, changes={'remaining_polls': 2}))
        with self.assertRaisesRegex(ValueError, 'Stale'):
            runner.update(replace(update, idempotency_key='u2'))
        update.changes['remaining_polls'] = 99
        self.assertEqual(receipt.request.changes, {'remaining_polls': 1})
        self.drain(runner)
        self.assertEqual(runner.update(receipt.request), receipt)
        with self.assertRaisesRegex(RuntimeError, 'terminal'):
            runner.update(DirectiveUpdate('slow', 1, 'new', {'remaining_polls': 1}))

    def test_uncertain_update_allows_status_and_retries_only_exact_intent(self):
        runner, children, _ = self.start_two()
        backend = children['slow'].learner
        backend.lose_update_response = True
        update = DirectiveUpdate('slow', 0, 'lost', {'remaining_polls': 2})
        with self.assertRaises(OSError):
            runner.update(update)
        self.assertEqual(runner.exploration.inspect('slow').version, 0)
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            runner.update(replace(update, idempotency_key='different'))
        progress = runner.exploration.advance('slow')
        self.assertFalse(progress.terminal)
        self.assertEqual(progress.version, 0)
        self.assertEqual(children['slow'].commands.receipts['slow'].pending_update, update)
        self.assertTrue(runner.exploration.advance('fast').terminal)
        self.assertEqual(runner.update(update).version, 1)
        self.assertEqual(len(backend.updates), 1)
        self.drain(runner)

    def test_unavailable_update_receipt_never_blocks_status_cancel_or_terminal_delivery(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                runner, children, _ = self.start_two()
                backend = children['slow'].learner
                apply_update = backend.update
                attempts = []

                def unavailable(job, request):
                    attempts.append(deepcopy(request))
                    apply_update(job, request)
                    raise TimeoutError('Update receipt endpoint remains unavailable')

                backend.update = unavailable
                request = DirectiveUpdate('slow', 0, 'uncertain', {'remaining_polls': 1})
                with self.assertRaises(TimeoutError):
                    runner.update(request)
                self.assertEqual(runner.exploration.inspect('slow').version, 0)
                with self.assertRaisesRegex(RuntimeError, 'another update'):
                    runner.update(replace(request, idempotency_key='replacement'))
                if cancel:
                    runner.cancel('slow')
                    with self.assertRaisesRegex(RuntimeError, 'cancelling'):
                        runner.update(request)
                self.drain(runner)
                state = children['slow'].commands.receipts['slow']
                self.assertEqual(state.feedback.status,
                                 ExecutionStatus.CANCELLED if cancel else ExecutionStatus.COMPLETED)
                self.assertEqual(state.pending_update, request)
                self.assertEqual(state.version, 0)  # Unknown response is never counted as confirmed.
                self.assertEqual(attempts, [request])
                self.assertEqual(len(backend.submissions), 1)
                self.assertEqual(len(backend.cancellations), int(cancel))
                self.assertEqual(children['fast'].learner.cancellations, [])
                self.assertEqual(runner.decision_making.harness.state.completed, 2)
                self.assertFalse(runner.busy)
                with self.assertRaisesRegex(RuntimeError, 'terminal'):
                    runner.update(request)
                with self.assertRaisesRegex(RuntimeError, 'terminal'):
                    runner.update(replace(request, idempotency_key='another'))
                self.assertEqual(attempts, [request])

    def test_parallel_width_one_polls_and_cancels_without_reentering_serial_decision(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                runner, children, _ = build_demo()
                harness = runner.decision_making.harness
                harness.max_outstanding = 1
                runner.exploration.harness.max_in_flight = 1
                runner.step('task')
                finish = Mock(decide=Mock(return_value=Decision('finish')))
                harness.policy = finish
                self.assertFalse(runner.decision_making.can_step())
                with self.assertRaisesRegex(RuntimeError, 'Consume pending'):
                    runner.decision_making.step('task')
                self.assertEqual(runner.step('task').kind, 'waiting')
                finish.decide.assert_not_called()
                if cancel:
                    runner.cancel('slow')
                results = self.drain(runner)
                self.assertEqual([r.kind for r in results].count('completed'), 1)
                self.assertEqual(harness.state.completed, 1)
                self.assertEqual(runner.exploration.inspect('slow').feedback.status,
                                 ExecutionStatus.CANCELLED if cancel else ExecutionStatus.COMPLETED)
                self.assertEqual(len(children['slow'].learner.submissions), 1)
                self.assertEqual(len(children['slow'].learner.cancellations), int(cancel))
                self.assertTrue(runner.decision_making.can_step())
                finish.decide.assert_called_once()
                self.assertFalse(runner.busy)

    def test_execution_exception_delivers_failure_then_retires_parallel_slot(self):
        from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
        from types import SimpleNamespace

        class FailingEnvironment:
            def reset(self, directive):
                raise OSError('environment unavailable')

            def close(self):
                pass

        runner, _, _ = build_demo()
        harness = runner.decision_making.harness
        harness.policy = Mock(decide=Mock(side_effect=[
            Decision('explore', directive=Directive('failure', 'task', 'fail')),
            Decision('finish')]))
        runner.exploration.harness.factory = lambda directive: DefaultExplorationHarness(
            SimpleNamespace(), FailingEnvironment(), harness.experience)
        with self.assertRaisesRegex(OSError, 'environment unavailable'):
            runner.step('task')
        self.assertEqual(harness.state.completed, 1)
        self.assertEqual(harness.state.last_feedback.status, ExecutionStatus.FAILED)
        self.assertFalse(runner.decision_making.pending_all())
        self.assertFalse(runner.exploration.pending_all())
        self.assertFalse(runner.busy)
        self.assertFalse(runner._slots)
        self.assertEqual(runner.step('task').kind, 'finished')
        self.assertEqual(runner.step('task').kind, 'finished')
        self.assertEqual(harness.state.completed, 1)

    def test_backend_unsupported_and_invalid_update_leave_target_usable(self):
        runner, children, _ = self.start_two()
        unsupported = DirectiveUpdate('slow', 0, 'bad', {'learning_rate': 0.1})
        with self.assertRaisesRegex(NotImplementedError, 'only remaining_polls'):
            runner.update(unsupported)
        self.assertIsNone(children['slow'].commands.receipts['slow'].pending_update)
        backend = children['slow'].learner
        backend.update = None
        with self.assertRaisesRegex(NotImplementedError, 'does not support'):
            runner.update(replace(unsupported, changes={'remaining_polls': 1}))
        runner.cancel('slow')
        self.drain(runner)
        for args in [('slow', -1, 'key', {'x': 1}), ('slow', True, 'key', {'x': 1}),
                     ('slow', 0, '', {'x': 1}), ('slow', 0, 'key', {}), ('slow', 0, 'key', {'x': float('nan')})]:
            with self.assertRaises(ValueError):
                DirectiveUpdate(*args)

    def test_model_controls_use_existing_permissions_and_allow_research_at_capacity(self):
        runner, children, _ = self.start_two()
        loop = runner.decision_making
        loop.harness.state.finished = False
        loop.harness.policy = Mock(decide=Mock(return_value=Decision('invoke', operation=OperationCall(
            'experiment_inspect', {'directive_id': 'slow'}))))
        self.assertEqual(loop.step('task').kind, 'progressed')
        self.assertEqual(len(loop.pending_all()), 2)
        loop.action_space.permissions = ActionPermissions(operations=frozenset({'experiment_inspect'}))
        loop.harness.policy.decide.return_value = Decision('invoke', operation=OperationCall('experiment_cancel', {'directive_id': 'slow'}))
        with self.assertRaises(PermissionError):
            loop.step('task')
        self.assertFalse(children['slow'].commands.receipts['slow'].cancel_requested)
        review_names = {spec.name for spec in authorized_space(loop.action_space, 'review').describe()}
        self.assertEqual(review_names, {'experiment_inspect'})
        loop.action_space.permissions = ActionPermissions()
        self.assertEqual(loop.step('task').kind, 'progressed')
        self.assertTrue(children['slow'].commands.receipts['slow'].cancel_requested)
        self.assertFalse(children['fast'].commands.receipts['fast'].cancel_requested)

    def test_new_dispatch_at_capacity_is_rejected_without_ownership_mutation(self):
        runner, _, _ = self.start_two()
        harness = runner.decision_making.harness
        harness.state.finished = False
        harness.policy = Mock(decide=Mock(return_value=Decision('explore', directive=Directive('third', 'task', ''))))
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            runner.decision_making.step('task')
        self.assertNotIn('third', harness.state.issued_directive_ids)
        self.assertEqual(len(harness.pending_all()), 2)

    def test_review_record_failures_are_isolated_by_directive(self):
        runner, _, _ = self.start_two()
        harness = runner.decision_making.harness
        policy = Mock(review=Mock(return_value=ReviewDecision('continue', reason='checked')))
        harness.policy = policy
        append = harness.memory.append
        failed = []

        def lose(entry):
            if entry.task_id == 'slow' and entry.summary == 'Review: continue' and not failed:
                failed.append(True)
                raise OSError('record unavailable')
            append(entry)

        harness.memory.append = lose
        slow, fast = runner.exploration.inspect('slow'), runner.exploration.inspect('fast')
        with self.assertRaises(OSError):
            harness.review('task', slow)
        self.assertIn('slow', harness.state.review_records)
        harness.review('task', fast)
        self.assertEqual(policy.review.call_count, 2)
        harness.review('task', slow)
        self.assertEqual(policy.review.call_count, 2)
        self.assertFalse(harness.state.review_records)

    def test_compare_requires_full_base_and_compatible_completed_evaluations(self):
        runner, _, _ = self.complete_demo()
        harness = runner.decision_making.harness
        state = harness.state
        original = deepcopy(state.candidate_evaluations['fast'])
        for changed in (replace(original, protocol_ref='other'), replace(original, simulated=False),
                        replace(original, status=ExecutionStatus.FAILED),
                        replace(original, quality_metrics={}), replace(original, quality_metrics={'quality': float('nan')})):
            state.candidate_evaluations['fast'] = changed
            with self.assertRaises(ValueError):
                harness.compare_candidates(('slow', 'fast'), metric='quality')
        state.candidate_evaluations['fast'] = original
        state.candidate_bases['fast'] = replace(BASE, model_id='different-model-same-revision')
        with self.assertRaisesRegex(ValueError, 'full model bases'):
            harness.compare_candidates(('slow', 'fast'), metric='quality')
        del state.candidate_bases['fast']
        with self.assertRaisesRegex(ValueError, 'known full'):
            harness.compare_candidates(('slow', 'fast'), metric='quality')
        state.candidate_bases['fast'] = BASE
        result = harness.compare_candidates(('slow', 'fast'), metric='quality', higher_is_better=False)
        self.assertEqual(result[0].candidate_id, 'slow')
        result[0].quality_metrics['quality'] = 999
        self.assertEqual(state.candidate_evaluations['slow'].quality_metrics['quality'], 0.7)

    def test_apply_is_targeted_requires_ack_and_checks_full_model_identity(self):
        runner, children, _ = self.complete_demo()
        concurrent = runner.exploration.harness
        target = children['eval-slow']
        self.assertEqual(concurrent.apply_candidate('fast', target_directive_id='eval-slow', expected_model=BASE).revision, 'fast')
        self.assertEqual(target.policy.model.revision, 'fast')
        self.assertEqual(children['slow'].policy.model, BASE)
        with self.assertRaisesRegex(ValueError, 'model changed'):
            concurrent.apply_candidate('slow', target_directive_id='eval-slow', expected_model=BASE)
        target.policy.model = replace(BASE, model_id='different')
        with self.assertRaisesRegex(ValueError, 'stale'):
            concurrent.apply_candidate('slow', target_directive_id='eval-slow', expected_model=target.policy.model)
        concurrent._acknowledged.remove('eval-fast')
        with self.assertRaisesRegex(RuntimeError, 'unacknowledged'):
            concurrent.apply_candidate('slow', target_directive_id='eval-fast', expected_model=BASE)

    def test_model_can_yield_without_external_job_and_capacity_is_not_advertised(self):
        from rsisys.loops.contracts.model_service import GenerationResponse
        from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
        runner, _, _ = self.start_two()
        loop = runner.decision_making
        loop.harness.state.finished = False
        backend = Mock(generate=Mock(return_value=GenerationResponse('{"kind":"wait"}', BASE, True)))
        loop.harness.policy = ModelDecisionPolicy(backend, BASE, directives=loop.action_space.directives)
        result = loop.step('task')
        self.assertEqual(result.kind, 'waiting')
        self.assertIsNone(result.job)
        self.assertIsNone(loop.harness.state.pending_job)
        self.assertFalse(loop.harness.state.finished)
        context = json.loads(backend.generate.call_args.args[0].prompt)
        self.assertEqual(context['directive_modes'], [])
        self.assertEqual(context['directives'], [])
        self.assertEqual(len(context['outstanding']), 2)
        self.assertIn('experiment_update', {spec['name'] for spec in context['operations']})
        self.drain(runner)  # Parent already observed FinishPolicy; it still drains both jobs.
        context = loop.harness._context('task', loop.action_space)
        self.assertIn('train', context.directive_modes)

    def test_custom_concurrent_harness_snapshot_is_explicitly_rejected(self):
        runner, _, _ = build_demo()
        with TemporaryDirectory() as directory:
            app = Mock(store=object(), custom_assembly=False, runner=runner)
            app.config.data_dir = Path(directory)
            with self.assertRaisesRegex(TypeError, 'snapshot integration|parallel'):
                LocalSnapshotCapture(app).capture('unsupported')


if __name__ == '__main__':
    unittest.main()
