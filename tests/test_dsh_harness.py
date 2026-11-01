"""DSH integration boundaries without an SDK or model service dependency."""
from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import patch
import time
import tempfile
import unittest

from rsisys.config import DemoConfig, build_demo
from rsisys.application import Application, ApplicationConfig
from rsisys.runtime.run_policy import TimeRunPolicy
from rsisys.loops.contracts.coordination import ReviewDecision, ExplorationProgress
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.harness.dsh import DSHConfig, dsh_decision_factory

CONFIG = DemoConfig.load(Path(__file__).resolve().parents[1] / 'configs/demo.json')
DSH = DSHConfig(Path('/tmp/rsisys-test-home'), Path('/tmp'), 'fixture-model', 'research-1')


class Runtime:
    def __init__(self, answers, gate=None):
        self.answers = iter(answers)
        self.prompts = []
        self.opened = []
        self.closed = False
        self.gate = gate
        self.entered = Event()

    def start_session(self, ident):
        self.opened.append(('start', ident))
        return self

    def resume_session(self, ident):
        self.opened.append(('resume', ident))
        return self

    def run(self, prompt, on_notification=None):
        self.prompts.append(prompt)
        self.entered.set()
        if self.gate:
            self.gate.wait(3)
        value = next(self.answers)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(final_response=value, finish_reason='completed')

    def close(self):
        self.closed = True
        if self.gate:
            self.gate.set()


class DSHTests(unittest.TestCase):
    def build(self, runtime, *, config=DSH, permissions=None):
        with patch('rsisys.loops.decision_making.harness.dsh._sdk_runtime', return_value=runtime):
            runner, store = build_demo(CONFIG, decision_factory=dsh_decision_factory(config),
                                       action_permissions=permissions)
        self.addCleanup(runner.close)
        return runner

    def next_ready(self, harness, space):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = harness.step('Research', space)
            if result.kind != 'waiting':
                return result
            time.sleep(.002)
        self.fail('Harness never became ready')

    def test_session_reused_and_actions_execute_on_runner_thread(self):
        runtime = Runtime(['{"kind":"invoke","operation":"prepare_data","arguments":{"text":"x"}}',
                           '{"kind":"finish","reason":"done"}'])
        runner = self.build(runtime)
        loop = runner.decision_making
        called = []
        original = loop.harness.tools._bindings['prepare_data']
        loop.harness.tools._bindings['prepare_data'] = lambda arguments: (called.append(get_ident()), original(arguments))[1]
        self.assertEqual(self.next_ready(loop.harness, loop.action_space).kind, 'progressed')
        self.assertEqual(self.next_ready(loop.harness, loop.action_space).kind, 'finished')
        self.assertEqual(called, [get_ident()])
        self.assertEqual(runtime.opened, [('start', DSH.session_id)])
        self.assertIn('Prepared input artifact', runtime.prompts[-1])
        self.assertEqual(loop.harness.state.calls_with_missing_usage, 2)

    def test_incremental_delivery_keeps_the_complete_latest_receipt(self):
        import json
        from rsisys.loops.decision_making.harness.context import IdentityContextCompactor
        runtime = Runtime(['{"kind":"invoke","operation":"prepare_data","arguments":{"text":"x"}}',
                           '{"kind":"finish","reason":"done"}'])
        loop = self.build(runtime).decision_making
        receipt = OperationResult('Long raw receipt '+('S'*1000), (), False,
            data={'unknown_domain_field': {'raw': '完整最新反馈'*15000, 'tail': 'END-OF-RAW'}})
        loop.harness.tools._bindings['prepare_data'] = lambda arguments: receipt
        self.next_ready(loop.harness, loop.action_space)
        self.next_ready(loop.harness, loop.action_space)
        latest = json.loads(runtime.prompts[-1].split('Current research observations (data):\n')[1])
        self.assertEqual(latest['operation_results'][-1]['data'], receipt.data)
        self.assertEqual(latest['operation_results'][-1]['summary'], receipt.summary)
        self.assertNotIn('objective', latest)
        self.assertIn('directives', latest)  # Live schemas survive native compaction.
        self.assertIn('RSISys action protocol for this turn:\nReturn one JSON', runtime.prompts[-1])
        self.assertIsInstance(loop.harness.context_compactor, IdentityContextCompactor)
        self.assertIsInstance(loop.harness.policy.parser.context_compactor, IdentityContextCompactor)

    def test_transport_cursor_survives_recovery_without_replaying_history(self):
        runtime = Runtime(['{"kind":"invoke","operation":"prepare_data","arguments":{"text":"cursor"}}'])
        source = self.build(runtime).decision_making
        self.next_ready(source.harness, source.action_space)
        state = source.harness.export_recovery_state()
        restored_runtime = Runtime(['{"kind":"finish"}'])
        target = self.build(restored_runtime).decision_making
        target.harness.restore_recovery_state(state)
        self.next_ready(target.harness, target.action_space)
        self.assertEqual(restored_runtime.opened, [('resume', DSH.session_id)])
        self.assertNotIn('"objective"', restored_runtime.prompts[-1])
        self.assertIn('"directives"', restored_runtime.prompts[-1])

    def test_native_config_requires_explicit_chat_provider_and_valid_budgets(self):
        self.assertEqual((DSH.trigger_tokens, DSH.target_tokens, DSH.summary_tokens), (80000, 24000, 4096))
        with self.assertRaisesRegex(ValueError, 'base_url'):
            replace(DSH, provider='openai-completions')
        with self.assertRaisesRegex(ValueError, 'summary_tokens'):
            replace(DSH, target_tokens=1024)
        with self.assertRaisesRegex(ValueError, 'provider'):
            replace(DSH, provider='implicit-qwen-fallback')

    def test_post_compaction_aggregate_budget_withholds_oversized_proposal(self):
        from rsisys.loops.decision_making.harness.context import ContextBudgetError
        class Metered(Runtime):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                result.events = [{'type': 'compaction/summary'}, {'type': 'assistant/message',
                    'data': {'usage': {'inputTokens': 1, 'outputTokens': 50, 'totalTokens': 25050}}}]
                return result
        loop = self.build(Metered(['{"kind":"finish"}'])).decision_making
        loop.step('Research')
        loop.harness.policy.future.exception(timeout=2)
        with self.assertRaisesRegex(ContextBudgetError, 'proposal withheld'):
            loop.step('Research')
        self.assertFalse(loop.harness.state.finished)
        observation = loop.harness.policy.backend.last_compaction
        self.assertEqual(observation['full_input_tokens'], 25000)  # Includes cached input, not just uncached 1.
        self.assertTrue(observation['full_request_checked'])

    def test_post_compaction_missing_aggregate_usage_is_explicitly_unknown(self):
        class Unmetered(Runtime):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                result.events = [{'type': 'compaction/summary'}]
                return result
        loop = self.build(Unmetered(['{"kind":"finish"}'])).decision_making
        self.assertEqual(self.next_ready(loop.harness, loop.action_space).kind, 'finished')
        observation = loop.harness.policy.backend.last_compaction
        self.assertIsNone(observation['full_input_tokens'])
        self.assertFalse(observation['full_request_checked'])
        self.assertEqual(observation['count_method'], 'unknown')

    def test_waiting_ticks_do_not_duplicate_model_calls_or_history(self):
        gate = Event()
        runtime = Runtime(['{"kind":"finish"}'], gate)
        runner = self.build(runtime)
        loop = runner.decision_making
        self.assertEqual(loop.step('x').kind, 'waiting')
        self.assertTrue(runtime.entered.wait(1))
        before = len(loop.harness.memory.history(limit=100))
        for _ in range(10):
            self.assertEqual(loop.step('x').kind, 'waiting')
        self.assertEqual(len(runtime.prompts), 1)
        self.assertEqual(len(loop.harness.memory.history(limit=100)), before)
        self.assertEqual(loop.harness.state.calls_with_missing_usage, 0)
        gate.set()
        self.assertEqual(self.next_ready(loop.harness, loop.action_space).kind, 'finished')

    def test_permission_rechecked_after_background_generation(self):
        gate = Event()
        runtime = Runtime(['{"kind":"invoke","operation":"prepare_data","arguments":{"text":"x"}}'], gate)
        runner = self.build(runtime)
        loop = runner.decision_making
        loop.step('x')
        self.assertTrue(runtime.entered.wait(1))
        loop.action_space.permissions = ActionPermissions(operations=())
        gate.set()
        with self.assertRaises(PermissionError):
            self.next_ready(loop.harness, loop.action_space)
        self.assertFalse(loop.harness.state.operation_results[-1].artifacts)

    def test_model_error_is_latched_without_resubmission(self):
        runtime = Runtime([TimeoutError('response lost')])
        runner = self.build(runtime)
        loop = runner.decision_making
        loop.step('x')
        loop.harness.policy.future.exception(timeout=2)
        for _ in range(2):
            with self.assertRaisesRegex(TimeoutError, 'response lost'):
                loop.step('x')
        self.assertEqual(len(runtime.prompts), 1)
        self.assertEqual(loop.harness.pending_research_job().backend, 'dsh-session')

    def test_malformed_output_is_not_a_finish_or_automatic_retry(self):
        runtime = Runtime(['not JSON'])
        runner = self.build(runtime)
        loop = runner.decision_making
        loop.step('x')
        loop.harness.policy.future.exception(timeout=2)
        with self.assertRaises(ValueError):
            loop.step('x')
        self.assertFalse(loop.harness.state.finished)
        self.assertEqual(loop.harness.state.calls_with_missing_usage, 1)
        self.assertEqual(len(runtime.prompts), 1)

    def test_explicit_resume_uses_restore_not_start(self):
        runtime = Runtime(['{"kind":"finish"}'])
        runner = self.build(runtime, config=replace(DSH, resume=True))
        loop = runner.decision_making
        self.next_ready(loop.harness, loop.action_space)
        self.assertEqual(runtime.opened, [('resume', DSH.session_id)])

    def test_research_cycle_restart_preserves_dsh_session(self):
        runtime = Runtime(['{"kind":"finish"}', '{"kind":"finish"}'])
        runner = self.build(runtime)
        loop = runner.decision_making
        self.next_ready(loop.harness, loop.action_space)
        loop.restart_research({'research_round': 2})
        self.next_ready(loop.harness, loop.action_space)
        self.assertEqual(len(runtime.prompts), 2)
        self.assertEqual(len(runtime.opened), 1)
        self.assertIn('research_round', runtime.prompts[-1])

    def test_cancel_only_owns_inference(self):
        runtime = Runtime(['{"kind":"finish"}'], Event())
        runner = self.build(runtime)
        loop = runner.decision_making
        loop.step('x')
        self.assertTrue(runtime.entered.wait(1))
        loop.harness.cancel_research()
        loop.harness.policy.future.result(timeout=2) if loop.harness.policy.future is not None else None
        self.assertTrue(loop.harness.cancel_research())
        self.assertTrue(runtime.closed)
        self.assertIsNone(loop.harness.pending_research_job())
        self.assertFalse(loop.harness.state.finished)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            loop.step('x')

    def test_time_stop_closes_inference_without_executing_late_action(self):
        runtime = Runtime(['{"kind":"invoke","operation":"prepare_data","arguments":{"text":"late"}}'], Event())
        runner = self.build(runtime)
        runner.run_policy = TimeRunPolicy(1)
        runner._guard_actions = True
        now = [0.0]
        runner._clock = lambda: now[0]
        self.assertEqual(runner.step('x').kind, 'waiting')
        self.assertTrue(runtime.entered.wait(1))
        now[0] = 2
        result = runner.step('x')
        deadline = time.monotonic() + 2
        while result.kind != 'finished' and time.monotonic() < deadline:
            time.sleep(.002)
            result = runner.step('x')
        self.assertEqual(result.kind, 'finished')
        self.assertTrue(result.event['run_stop']['shutdown_complete'])
        self.assertTrue(runtime.closed)
        self.assertEqual(runner.decision_making.harness.state.operation_results, [])

    def test_completed_proposal_recovered_without_another_model_call(self):
        runtime = Runtime(['{"kind":"finish","reason":"saved"}'])
        source = self.build(runtime).decision_making
        source.step('x')
        source.harness.policy.future.result(timeout=2)
        state = source.harness.export_recovery_state()
        target_runtime = Runtime([])
        target = self.build(target_runtime).decision_making
        target.harness.restore_recovery_state(state)
        result = target.step('x')
        self.assertEqual(result.kind, 'finished')
        self.assertEqual(result.reason, 'saved')
        self.assertEqual(target_runtime.prompts, [])
        self.assertEqual(target.harness.state.calls_with_missing_usage + source.harness.state.calls_with_missing_usage, 1)

    def test_interrupted_inference_needs_explicit_native_resume(self):
        from rsisys.runtime.recovery import RecoveryRequired
        runtime = Runtime(['{"kind":"finish"}'], Event())
        source = self.build(runtime).decision_making
        source.step('x')
        self.assertTrue(runtime.entered.wait(1))
        state = source.harness.export_recovery_state()
        source.harness.close()
        replacement = Runtime(['{"kind":"finish"}'])
        target = self.build(replacement).decision_making
        target.harness.restore_recovery_state(state)
        with self.assertRaises(RecoveryRequired):
            target.step('x')
        self.assertEqual(replacement.prompts, [])
        target.harness.resume_interrupted_inference(evidence='Verified native session; no actions executed')
        self.assertEqual(replacement.opened, [('resume', DSH.session_id)])
        self.assertEqual(replacement.prompts, [])
        self.assertEqual(self.next_ready(target.harness, target.action_space).kind, 'finished')
        self.assertEqual(len(replacement.prompts), 1)

    def test_recovery_rejects_different_session(self):
        source = self.build(Runtime([])).decision_making
        state = source.harness.export_recovery_state()
        target = self.build(Runtime([]), config=replace(DSH, session_id='another')).decision_making
        with self.assertRaisesRegex(ValueError, 'original'):
            target.harness.restore_recovery_state(state)

    def test_durable_application_restores_buffered_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(ApplicationConfig.load(Path(__file__).resolve().parents[1] / 'configs/local.json'),
                             data_dir=Path(directory) / 'run', execution='mock')
            runtime = Runtime(['{"kind":"finish","reason":"durable"}'])
            factory = dsh_decision_factory(DSH)
            with patch('rsisys.loops.decision_making.harness.dsh._sdk_runtime', return_value=runtime):
                with Application(config, decision_factory=factory, durable=True) as app:
                    app.runner.step(config.objective)
                    app.runner.decision_making.harness.policy.future.result(timeout=2)
                    # Persist the asynchronously completed proposal at an ordinary mutation boundary.
                    app.recovery.mutate('checkpoint_fixture', (), lambda: None)
            target_runtime = Runtime([])
            with patch('rsisys.loops.decision_making.harness.dsh._sdk_runtime', return_value=target_runtime):
                with Application.resume(config.data_dir, decision_factory=factory) as app:
                    result = app.runner.step(config.objective)
                    self.assertEqual(result.kind, 'finished')
                    self.assertEqual(target_runtime.prompts, [])

    def test_repeated_recovery_keeps_native_session_identity(self):
        runtime = Runtime(['{"kind":"finish"}'])
        source = self.build(runtime).decision_making
        source.step('x')
        source.harness.policy.future.result(timeout=2)
        first = source.harness.export_recovery_state()
        middle = self.build(Runtime([])).decision_making
        middle.harness.restore_recovery_state(first)
        self.assertEqual(middle.step('x').kind, 'finished')
        second = middle.harness.export_recovery_state()
        self.assertTrue(second['session_opened'])
        replacement = Runtime(['{"kind":"finish"}'])
        target = self.build(replacement).decision_making
        target.harness.restore_recovery_state(second)
        self.next_ready(target.harness, target.action_space)
        self.assertEqual(replacement.opened, [('resume', DSH.session_id)])

    def test_async_review_retains_one_cumulative_tool_budget(self):
        runtime = Runtime([
            '{"kind":"explore","strategy":"test","mode":"execute","arguments":{}}',
            '{"kind":"invoke","operation":"prepare_data","arguments":{"text":"one"}}',
            '{"kind":"invoke","operation":"prepare_data","arguments":{"text":"two"}}',
            '{"action":"continue","next_review_seconds":null,"reason":"review complete"}',
        ])
        runner = self.build(runtime, permissions=ActionPermissions.unrestricted())
        loop = runner.decision_making
        loop.harness.review_tool_limit = 2
        directive = self.next_ready(loop.harness, loop.action_space).directive
        progress = ExplorationProgress(directive.directive_id, 'running', directive)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = loop.review('Research', progress)
            if result.reason == 'review complete':
                break
            self.assertIsNotNone(loop.harness.state.review_session)
            time.sleep(.002)
        else:
            self.fail('Async review did not finish')
        self.assertEqual(len(loop.harness.state.operation_results), 0)  # review receipts stay scoped
        self.assertEqual(len(loop.harness.memory.history(kind='operation')), 2)
        final_context = __import__('json').loads(runtime.prompts[-1].split('Current research observations (data):\n')[1])
        self.assertEqual(final_context['operations'], [])
        reviews = [entry for entry in loop.harness.memory.history(kind='decision')
                   if entry.summary.startswith('Review:')]
        self.assertEqual(len(reviews), 1)
        self.assertIsNone(loop.harness.state.review_session)

    def test_completed_response_is_accounted_when_stopping(self):
        gate = Event()
        runtime = Runtime(['{"kind":"finish"}'], gate)
        loop = self.build(runtime).decision_making
        loop.step('x')
        self.assertTrue(runtime.entered.wait(1))
        self.assertEqual(loop.harness.state.calls_with_missing_usage, 0)
        gate.set()
        loop.harness.policy.future.result(timeout=2)
        self.assertTrue(loop.harness.cancel_research())
        self.assertEqual(loop.harness.state.calls_with_missing_usage, 1)
        self.assertIsNone(loop.harness.pending_research_job())

    def test_config_rejects_ambiguous_paths_and_invalid_windows(self):
        with self.assertRaises(ValueError):
            replace(DSH, home=Path('relative'))
        with self.assertRaises(ValueError):
            replace(DSH, max_tokens=25001)
        with self.assertRaises(ValueError):
            replace(DSH, session_id='')


if __name__ == '__main__':
    unittest.main()
