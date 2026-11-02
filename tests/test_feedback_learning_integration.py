"""Integration checks for opt-in feedback learning; all model output is a fixture."""
from concurrent.futures import Future
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from rsisys.assembly import AssemblyContext, AssemblyRunner, build_default_exploration
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.feedback_learning import FeedbackLearning, SerializedLearningBackend, validate_proposals
from rsisys.loops.decision_making.harness.learning_harness import feedback_learning_decision_factory
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.state import CandidateEvaluation
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.model import MockModelBackend
from rsisys.runtime.sandbox import MockSandboxBackend


class ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


class Backend(MockModelBackend):
    def __init__(self, *, simulated=True):
        self.requests = []
        self.bad_source = False
        self.simulated = simulated

    def generate(self, request):
        packet = json.loads(request.prompt)
        if 'receipt' not in packet:
            return super().generate(request)
        self.requests.append(request)
        proposal = dict(kind='memory', content='Observed fixture result, not a quality claim.',
                        applicability='This fixture only', validation_plan='Replay the receipt',
                        source_refs=['unknown'] if self.bad_source else [packet['source_refs'][0]])
        if 'candidate_evidence' in packet:
            proposal['evaluation_claims'] = []
        return GenerationResponse(json.dumps({'proposals': [proposal]}), request.model, simulated=self.simulated)


class FeedbackLearningIntegrationTests(unittest.TestCase):
    def learning(self, **kwargs):
        backend = Backend()
        learning = FeedbackLearning(SerializedLearningBackend(backend), executor=ImmediateExecutor(), **kwargs)
        self.addCleanup(lambda: None if learning._closed else learning.close())
        return backend, learning

    def capture(self, learning, ident='one'):
        return learning.capture(Directive(ident, 'objective', 'strategy'),
            Feedback(ident, ExecutionStatus.COMPLETED, 'observed', True), ModelRef('fixture', 'decision', 'v1'))

    def test_capture_does_not_infer_and_repeated_receipt_does_not_repeat_call(self):
        backend, learning = self.learning()
        ident = self.capture(learning)
        self.assertEqual(backend.requests, [])
        learning.poll()
        learning.poll()
        self.assertEqual(self.capture(learning), ident)
        learning.poll()
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(learning.state['jobs'][ident]['status'], 'completed')

    def test_failed_proposal_retains_response_and_never_becomes_memory(self):
        backend, learning = self.learning()
        backend.bad_source = True
        ident = self.capture(learning)
        learning.poll()
        learning.poll()
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'failed')
        self.assertIn('unknown', row['response']['text'])
        self.assertEqual(row['proposals'], [])
        self.assertEqual(len(backend.requests), 1)

    def test_budget_survives_journal_reopen(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'learning.json'
            backend, learning = self.learning(path=path, max_calls=1)
            self.capture(learning)
            learning.poll()
            learning.poll()
            learning.close()
            second = FeedbackLearning(SerializedLearningBackend(backend), path=path, max_calls=1,
                                      executor=ImmediateExecutor())
            ident = self.capture(second, 'two')
            second.poll()
            self.assertEqual(second.state['jobs'][ident]['status'], 'budget_exhausted')
            self.assertEqual(len(backend.requests), 1)
            second.close()

    def test_completed_redelivery_after_reopen_preserves_result_and_remaining_budget(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'learning.json'
            backend, learning = self.learning(path=path, max_calls=2)
            ident = self.capture(learning)
            learning.poll()
            learning.poll()
            completed = learning.read(ident)
            self.assertEqual(completed['status'], 'completed')
            self.assertTrue(completed['proposals'])
            learning.close()

            restored = FeedbackLearning(SerializedLearningBackend(backend), path=path, max_calls=2,
                                        executor=ImmediateExecutor())
            try:
                self.assertEqual(restored.read(ident), completed)
                self.assertEqual(self.capture(restored), ident)
                restored.poll()
                restored.poll()
                self.assertEqual(restored.read(ident), completed)
                self.assertEqual(restored.state['attempted'], 1)
                self.assertEqual(len(restored.state['jobs']), 1)
                self.assertEqual(len(backend.requests), 1)

                second = self.capture(restored, 'two')
                restored.poll()
                restored.poll()
                self.assertEqual(restored.read(second)['status'], 'completed')
                self.assertEqual(restored.state['attempted'], 2)
                self.assertEqual(len(backend.requests), 2)
                exhausted = self.capture(restored, 'three')
                restored.poll()
                self.assertEqual(restored.read(exhausted)['status'], 'budget_exhausted')
                self.assertEqual(restored.state['attempted'], 2)
                self.assertEqual(len(backend.requests), 2)
            finally:
                restored.close()

    def test_async_capture_preserves_model_identity_and_reports_unfinished_close(self):
        entered, release = Event(), Event()

        class DelayedBackend(Backend):
            def generate(self, request):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError('fixture transport timeout')
                return super().generate(request)

        backend = DelayedBackend()
        learning = FeedbackLearning(SerializedLearningBackend(backend))
        try:
            ident = self.capture(learning)
            learning.poll()
            self.assertTrue(entered.wait(1))
            self.assertEqual(learning.state['jobs'][ident]['status'], 'running')
            # Redelivery after a policy update cannot retarget the original job.
            self.assertEqual(learning.capture(Directive('one', 'objective', 'strategy'),
                Feedback('one', ExecutionStatus.COMPLETED, 'observed', True),
                ModelRef('fixture', 'decision', 'v2')), ident)
            with self.assertRaises(TimeoutError):
                learning.close(timeout=0)
            self.assertEqual(learning.state['jobs'][ident]['status'], 'running')
        finally:
            release.set()
            learning.close()
        self.assertEqual(backend.requests[0].model.revision, 'v1')
        self.assertEqual(learning.state['jobs'][ident]['status'], 'completed')

    def test_recovery_does_not_replay_uncertain_inference(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'learning.json'
            backend, learning = self.learning(path=path)
            ident = self.capture(learning)
            learning.close()
            journal = json.loads(path.read_text())
            journal['jobs'][ident]['status'] = 'running'
            journal['attempted'] = 1
            path.write_text(json.dumps(journal))
            restored = FeedbackLearning(SerializedLearningBackend(backend), path=path,
                                        executor=ImmediateExecutor())
            try:
                restored.poll()
                self.assertEqual(restored.state['jobs'][ident]['status'], 'interrupted')
                self.assertEqual(restored.state['attempted'], 1)
                self.assertEqual(backend.requests, [])
            finally:
                restored.close()

    def test_duplicate_json_and_unverified_training_target_are_checked(self):
        with self.assertRaises(ValueError):
            validate_proposals('{"proposals":[],"proposals":[]}', ['receipt'])
        row = dict(kind='training_data', content='Candidate example', applicability='fixture',
                   validation_plan='independent replay', source_refs=['receipt'],
                   training_example=dict(input='observation', target='proposed action', target_status='unverified'))
        self.assertEqual(validate_proposals(json.dumps({'proposals': [row]}), ['receipt']), [row])
        row['training_example']['target_status'] = 'verified'
        with self.assertRaises(ValueError):
            validate_proposals(json.dumps({'proposals': [row]}), ['receipt'])

    def assembly(self, directory, *, simulated=True):
        context = AssemblyContext(BasicExperienceSystem(InMemoryExperienceStore()),
            FileArtifactStore(Path(directory) / 'artifacts'), {'fixture': Backend(simulated=simulated)}, MockSandboxBackend(),
            {role: ModelRef('fixture', role, 'v1') for role in ('decision_maker', 'actor')},
            data_dir=Path(directory))
        self.addCleanup(context.close)
        exploration = build_default_exploration(context)
        decision = feedback_learning_decision_factory(executor=ImmediateExecutor())(context, exploration)
        return context, exploration, decision

    def test_unverified_memory_keeps_all_history_before_and_after_memory_changes(self):
        with TemporaryDirectory() as directory:
            context, exploration, decision = self.assembly(directory)
            AssemblyRunner(decision, exploration, context).run('objective', rounds=3, max_decision_steps=12)
            harness = decision.harness
            view = harness._context('objective')
            raw = harness.memory.history()
            self.assertTrue(view.memories)
            self.assertTrue(all(memory.status == 'unverified' for memory in view.memories))
            self.assertEqual(view.history, raw)
            for memory in view.memories:
                harness.memory.remember(replace(memory, content=memory.content + ' modified'))
            self.assertEqual(harness._context('objective').history, raw)
            context.close()

    def test_uncited_contradictory_earlier_attempt_result_stays_in_history(self):
        with TemporaryDirectory() as directory:
            context, _, loop = self.assembly(directory)
            try:
                harness = loop.harness
                first = Decision('invoke', decision_id='prepare', research_attempt_title='Validate an idea',
                                 operation=OperationCall('prepare_data', {'text': 'fixture'}))
                with patch.object(harness.policy, 'decide', return_value=first):
                    harness.step('objective', loop.action_space)
                attempt = harness.state.decision_records['prepare']['research_attempt_id']
                negative = HistoryEntry('earlier-negative', 'operation', None,
                    'Independent validation rejected the earlier result', False,
                    details={'decision_id': 'prepare', 'quality_metrics': {'score': 0.0}})
                harness.memory.append(negative)
                directive = Directive('later-experiment', 'objective', 'Check the apparent improvement')
                later = Decision('explore', decision_id='later', directive=directive, research_attempt_id=attempt)
                with patch.object(harness.policy, 'decide', return_value=later):
                    harness.step('objective', loop.action_space)
                harness.receive_feedback(directive, Feedback(directive.directive_id,
                    ExecutionStatus.COMPLETED, 'Later result appears positive', False,
                    quality_metrics={'score': 1.0}))
                # Keep the contradiction outside the normal latest-two protection.
                for ident in ('tail-one', 'tail-two'):
                    harness.memory.append(HistoryEntry(ident, 'operation', None, 'Other observation', False))
                view = harness._context('objective')
                learning = harness.feedback_learning
                job = learning.read(research_attempt_id=attempt)['items'][0]['job_id']
                row = learning.state['jobs'][job]
                self.assertEqual(row['status'], 'completed')
                self.assertIn(negative.record_id, [item['record_id'] for item in row['input']['history']])
                self.assertTrue(row['proposals'])
                for proposal in row['proposals']:
                    self.assertEqual(proposal['source_refs'], [row['input']['source_refs'][0]])
                self.assertTrue(view.memories)
                self.assertTrue(all(negative.summary not in memory.content for memory in view.memories))
                self.assertIn(negative, view.history[:-2])
                self.assertEqual(view.history, harness.memory.history())
            finally:
                context.close()

    def test_delayed_feedback_preserves_v1_issuer_and_v2_learning_model(self):
        with TemporaryDirectory() as directory:
            context, _, loop = self.assembly(directory)
            try:
                harness = loop.harness
                issuer = harness.policy.model
                directive = Directive('delayed', 'objective', 'Await an experiment result')
                proposal = Decision('explore', decision_id='issued-v1', directive=directive,
                                    research_attempt_title='Delayed experiment')
                with patch.object(harness.policy, 'decide', return_value=proposal):
                    harness.step('objective', loop.action_space)
                self.assertEqual(harness.state.directive_provenance[directive.directive_id].decision_model, issuer)
                learner = replace(issuer, revision='v2')
                harness.policy.update_model(learner)
                feedback = Feedback(directive.directive_id, ExecutionStatus.COMPLETED, 'Delayed result', False)
                harness.receive_feedback(directive, feedback)
                view = harness._context('objective')
                learning = harness.feedback_learning
                job = learning.read()['items'][0]['job_id']
                scope = learning.read(job)['research_context']
                self.assertEqual(scope['issuing_decision_model'], asdict(issuer))
                self.assertEqual(scope['learning_decision_model'], asdict(learner))
                self.assertEqual(learning.state['jobs'][job]['status'], 'completed')
                self.assertEqual(learning.state['jobs'][job]['model'], asdict(learner))
                requests = context.backends['fixture'].requests
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].model, learner)
                self.assertEqual(json.loads(requests[0].prompt)['research_context'], scope)
                self.assertEqual(len(view.memories), 1)
                published = json.loads(view.memories[0].content)
                self.assertEqual(published['research_context'], scope)
                self.assertEqual(published['learned_by'], asdict(learner))

                # Redelivery under a newer policy must preserve the captured identities.
                harness.policy.update_model(replace(learner, revision='v3'))
                harness.receive_feedback(directive, feedback)
                self.assertEqual(learning.read(job)['research_context'], scope)
                self.assertEqual(learning.state['attempted'], 1)
                self.assertEqual(len(requests), 1)
            finally:
                context.close()

    def test_simulation_propagates_from_selected_history_evaluations_or_response(self):
        for history_simulated, evaluation_simulated, response_simulated in (
                (False, False, False), (True, False, False),
                (False, True, False), (False, False, True)):
            with self.subTest(history=history_simulated, evaluation=evaluation_simulated,
                              response=response_simulated), TemporaryDirectory() as directory:
                context, _, loop = self.assembly(directory, simulated=response_simulated)
                try:
                    harness = loop.harness
                    directive = Directive('mixed-evidence', 'objective', 'Validate mixed evidence')
                    proposal = Decision('explore', decision_id='mixed', directive=directive)
                    with patch.object(harness.policy, 'decide', return_value=proposal):
                        harness.step('objective', loop.action_space)
                    for ident, simulated in (('real-history', False), ('mixed-history', history_simulated),
                                             ('unrelated-simulated-history', True)):
                        task = 'unrelated' if ident.startswith('unrelated') else directive.directive_id
                        harness.memory.append(HistoryEntry(ident, 'operation', task, 'Earlier observation', simulated))
                    for ident, simulated in (('real-evaluation', False), ('mixed-evaluation', evaluation_simulated)):
                        harness.state.candidate_evaluations[ident] = CandidateEvaluation(
                            ident, directive.directive_id, 'protocol://fixture', (),
                            ExecutionStatus.COMPLETED, {'score': 0.5}, simulated)
                    harness.receive_feedback(directive, Feedback(directive.directive_id,
                        ExecutionStatus.COMPLETED, 'Real experiment receipt', False))
                    view = harness._context('objective')
                    learning = harness.feedback_learning
                    job = learning.read()['items'][0]['job_id']
                    row = learning.state['jobs'][job]
                    expected = history_simulated or evaluation_simulated or response_simulated
                    self.assertEqual(row['status'], 'completed')
                    self.assertFalse(row['input']['receipt']['directive']['simulated'])
                    self.assertFalse(row['input']['receipt']['feedback']['simulated'])
                    selected = {item['record_id']: item['simulated'] for item in row['input']['history']}
                    self.assertIs(selected['real-history'], False)
                    self.assertIs(selected['mixed-history'], history_simulated)
                    self.assertNotIn('unrelated-simulated-history', selected)
                    self.assertEqual([item['simulated'] for item in row['input']['evaluations']],
                                     [False, evaluation_simulated])
                    self.assertIs(row['response']['simulated'], response_simulated)
                    self.assertIs(learning.read(job)['simulated'], expected)
                    self.assertEqual(len(view.memories), 1)
                    self.assertIs(json.loads(view.memories[0].content)['simulated'], expected)
                    result = harness.tools.execute(OperationCall('feedback_learning_read', {'job_id': job}))
                    self.assertIs(result.simulated, expected)
                finally:
                    context.close()

    def test_read_operation_obeys_permissions_and_live_permission_changes(self):
        with TemporaryDirectory() as directory:
            context, _, decision = self.assembly(directory)
            source = decision.action_space
            source.base.permissions = ActionPermissions(operations=frozenset())
            view = authorized_space(source)
            call = OperationCall('feedback_learning_read', {})
            self.assertNotIn(call.name, [spec.name for spec in view.describe()])
            with self.assertRaises(PermissionError):
                view.validate(call)
            source.base.permissions = ActionPermissions()
            self.assertEqual(view.validate(call), call)
            context.close()

    def test_attempt_context_matches_monitor_and_preserves_out_of_order_receipts(self):
        from examples.qwen_toy.monitor import Monitor
        from rsimonitor import project_snapshot
        with TemporaryDirectory() as directory:
            context, _, loop = self.assembly(directory)
            harness = loop.harness
            harness.max_outstanding = 4
            monitor = Monitor(Path(directory), context.roles['actor'], 'objective')

            def step(proposal):
                monitor.decision(proposal, harness._context('objective', loop.action_space), 1)
                with patch.object(harness.policy, 'decide', return_value=proposal):
                    harness.step('objective', loop.action_space)
                monitor.sync_research_attempts(harness.state.decision_records)

            step(Decision('invoke', decision_id='a1', research_attempt_title='Same title',
                          operation=OperationCall('prepare_data', {'text': 'fixture'})))
            attempt_a = harness.state.decision_records['a1']['research_attempt_id']
            directive_b = Directive('db', 'objective', 'Alternative experiment')
            step(Decision('explore', decision_id='b1', directive=directive_b,
                          research_attempt_title='Same title', depends_on_decision_ids=('a1',)))
            attempt_b = harness.state.decision_records['b1']['research_attempt_id']
            self.assertNotEqual(attempt_a, attempt_b)
            directive_a = Directive('da', 'objective', 'First experiment')
            step(Decision('explore', decision_id='a2', directive=directive_a,
                          research_attempt_id=attempt_a))

            # B returns while A is also pending; receipt ownership remains explicit.
            for directive in (directive_b, directive_a):
                harness.receive_feedback(directive, Feedback(directive.directive_id,
                    ExecutionStatus.COMPLETED, 'Fixture result', True))
            harness._context('objective')
            learning = harness.feedback_learning
            b_job = learning.read(research_attempt_id=attempt_b)['items'][0]['job_id']
            a_job = learning.read(research_attempt_id=attempt_a)['items'][0]['job_id']
            b_scope = learning.read(b_job)['research_context']
            a_scope = learning.read(a_job)['research_context']
            self.assertEqual([r['decision_id'] for r in b_scope['member_decisions']], ['b1'])
            self.assertEqual(b_scope['member_decisions'][0]['depends_on_decision_ids'], ('a1',))
            self.assertEqual([r['decision_id'] for r in a_scope['member_decisions']], ['a1', 'a2'])
            self.assertEqual(a_scope['attempt_completion'], 'not_reported')
            supplied = learning.state['jobs'][a_job]['input']['history']
            self.assertTrue(any(r['details'].get('decision_id') == 'a1' for r in supplied))
            self.assertFalse(any(r['details'].get('decision_id') == 'b1' for r in supplied))

            public = project_snapshot(monitor.data, 'fixture-run')
            public_ids = {row['decision_id']: row['research_attempt_id'] for row in public['decisions']}
            self.assertEqual(public_ids, {'a1': attempt_a, 'a2': attempt_a, 'b1': attempt_b})
            for memory in harness._context('objective').memories:
                scope = json.loads(memory.content)['research_context']
                self.assertEqual(scope['research_attempt_id'], public_ids[scope['decision_id']])

            directive_c = Directive('da-next', 'objective', 'Continue the same idea')
            step(Decision('explore', decision_id='a3', directive=directive_c, research_attempt_id=attempt_a))
            harness.receive_feedback(directive_c, Feedback(directive_c.directive_id,
                ExecutionStatus.COMPLETED, 'Later fixture evidence', True))
            harness._context('objective')
            self.assertEqual(learning.read(research_attempt_id=attempt_a)['total'], 2)
            self.assertEqual(learning.read(a_job)['research_context'], a_scope)
            context.close()

    def test_legacy_feedback_is_not_assigned_to_an_attempt(self):
        _, learning = self.learning()
        ident = self.capture(learning)
        learning.poll()
        learning.poll()
        self.assertEqual(learning.read(ident)['research_context'], {})
        self.assertIsNone(learning.read()['items'][0]['research_attempt_id'])
        self.assertEqual(learning.read(research_attempt_id='unknown')['total'], 0)
        with self.assertRaises(ValueError):
            learning.read(ident, research_attempt_id='unknown')
