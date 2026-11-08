"""Retrospective grouping uses model fixtures, never real research claims."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.harness.attempt_learning import apply_attempt_updates
from rsisys.loops.decision_making.harness.feedback_learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.harness.decision import Decision
from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
from tests.feedback import test_feedback_learning_integration as integration

MODEL = ModelRef('fixture', 'decision', 'v1')


def update(members, previous=None):
    return dict(attempt_id=previous['attempt_id'] if previous else None,
                base_revision=previous['revision'] if previous else 0,
                title='Compare prompt options', summary='Observed fixture only; no adoption inferred.',
                member_decision_ids=members, source_refs=['decision:' + m for m in members])


class Backend:
    supports_response_schema = True
    def __init__(self):
        self.requests = []
        self.mutate = lambda result: result

    def count_tokens(self, request):
        return 100  # fixture only

    def generate(self, request):
        self.requests.append(request)
        packet = json.loads(request.prompt)
        group = packet['grouping_context']
        members = [r['decision_id'] for r in group['decisions']]
        previous = group['attempts'][0] if group['attempts'] else None
        result = {'proposals': [], 'attempt_updates': [update(members, previous)]}
        return GenerationResponse(json.dumps(self.mutate(result)), request.model, simulated=True)


class AttemptLearningTests(unittest.TestCase):
    def make(self, **kwargs):
        backend = Backend()
        learning = FeedbackLearning(SerializedLearningBackend(backend), executor=ImmediateExecutor(), **kwargs)
        self.addCleanup(learning.close)
        return backend, learning

    def capture(self, learning, ident, members):
        return learning.capture(Directive(ident, 'objective', 'Change prompt and compare accuracy'),
            Feedback(ident, ExecutionStatus.COMPLETED, 'Fixture feedback', True), MODEL,
            grouping_context={'decisions': [{'decision_id': m, 'status': 'completed'} for m in members],
                              'attempts': []})

    def test_incremental_grouping_revision_persistence_and_single_shared_call(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'learning.json'
            backend, learning = self.make(path=path)
            first = self.capture(learning, 'configure', ['change'])
            learning.poll(); learning.poll()
            initial = learning.attempts()['items'][0]
            second = self.capture(learning, 'evaluate', ['change', 'evaluate'])
            learning.poll(); learning.poll()
            view = learning.attempts(['change', 'evaluate', 'query'])
            self.assertEqual(view['items'][0]['attempt_id'], initial['attempt_id'])
            self.assertEqual(view['items'][0]['revision'], 2)
            self.assertEqual(view['revisions'][0], initial)
            self.assertEqual(view['ungrouped_decision_ids'], ['query'])
            self.assertEqual(learning.state['attempted'], 2)
            self.assertEqual(len(backend.requests), 2)
            self.assertEqual(learning.read(research_attempt_id=initial['attempt_id'])['total'], 2)
            self.assertEqual(learning.read(second)['attempt_revisions'][0]['provenance']['model'], asdict(MODEL))
            learning.close()
            restored = FeedbackLearning(SerializedLearningBackend(backend), path=path, executor=ImmediateExecutor())
            self.addCleanup(restored.close)
            self.assertEqual(restored.attempts(['change', 'evaluate', 'query']), view)
            self.assertEqual(self.capture(restored, 'configure', ['change']), first)
            restored.poll(); restored.poll()
            self.assertEqual(len(backend.requests), 2)

    def test_unknown_member_or_source_preserves_raw_response_and_does_not_publish(self):
        for field, bad in [('member_decision_ids', ['invented']), ('source_refs', ['unknown']),
                           ('base_revision', 10), ('attempt_id', 'made-up')]:
            with self.subTest(field=field):
                backend, learning = self.make()
                def mutate(result):
                    result['attempt_updates'][0][field] = bad
                    return result
                backend.mutate = mutate
                ident = self.capture(learning, 'experiment', ['one'])
                learning.poll(); learning.poll()
                row = learning.read(ident)
                self.assertEqual(row['status'], 'completed')
                self.assertTrue(row['grouping_error'])
                self.assertEqual(learning.attempts()['items'], [])
                self.assertIn(str(bad[0] if isinstance(bad, list) else bad), learning.state['jobs'][ident]['response']['text'])

    def test_conflicting_new_group_does_not_steal_existing_members(self):
        backend, learning = self.make()
        self.capture(learning, 'first', ['one']); learning.poll(); learning.poll()
        before = learning.attempts()
        def mutate(result):
            result['attempt_updates'][0].update(attempt_id=None, base_revision=0)
            return result
        backend.mutate = mutate
        ident = self.capture(learning, 'second', ['one', 'two']); learning.poll(); learning.poll()
        self.assertIn('Conflicting', learning.read(ident)['grouping_error'])
        self.assertEqual(learning.attempts(), before)

    def test_stale_revision_rejected_atomically(self):
        _, learning = self.make()
        ident = self.capture(learning, 'first', ['one']); learning.poll(); learning.poll()
        latest = deepcopy(learning.attempts()['items'])
        stale = deepcopy(learning.state['jobs'][ident])
        stale['attempt_updates'] = [update(['one'], latest[0])]
        latest[0]['revision'] += 1
        with self.assertRaisesRegex(ValueError, 'Stale'):
            apply_attempt_updates(stale, latest)
        self.assertEqual(latest[0]['revision'], 2)

    def test_empty_or_legacy_output_leaves_records_ungrouped(self):
        for output in ({'proposals': [], 'attempt_updates': []}, {'proposals': []}):
            backend, learning = self.make()
            backend.mutate = lambda _: output
            self.capture(learning, 'one', ['change']); learning.poll(); learning.poll()
            self.assertEqual(learning.attempts(['change'])['ungrouped_decision_ids'], ['change'])
            self.assertEqual(learning.state['attempted'], 1)

    def test_queue_refreshes_catalog_without_replaying_or_rewriting_receipts(self):
        backend, learning = self.make()
        self.capture(learning, 'first', ['one'])
        second = self.capture(learning, 'second', ['one', 'two'])
        original = deepcopy(learning.state['jobs'][second]['input']['receipt'])
        learning.poll(); learning.poll(); learning.poll()
        self.assertEqual(len(learning.attempts()['items']), 1)
        self.assertEqual(learning.attempts()['items'][0]['revision'], 2)
        self.assertEqual(learning.state['jobs'][second]['input']['receipt'], original)
        self.assertEqual(len(backend.requests), 2)

    def test_bounded_update_preserves_members_outside_current_window(self):
        _, learning = self.make()
        self.capture(learning, 'first', ['old']); learning.poll(); learning.poll()
        self.capture(learning, 'second', ['new']); learning.poll(); learning.poll()
        view = learning.attempts()
        self.assertEqual(len(view['items']), 1)
        self.assertEqual(view['items'][0]['member_decision_ids'], ['old', 'new'])
        self.assertEqual(view['items'][0]['source_refs'], ['decision:old', 'decision:new'])

    def test_identical_titles_do_not_merge_disjoint_ideas(self):
        backend, learning = self.make()
        self.capture(learning, 'first', ['one']); learning.poll(); learning.poll()
        def new_idea(result):
            result['attempt_updates'][0].update(attempt_id=None, base_revision=0)
            return result
        backend.mutate = new_idea
        self.capture(learning, 'second', ['two']); learning.poll(); learning.poll()
        self.assertEqual(len(learning.attempts()['items']), 2)

    def test_grouping_does_not_mutate_execution_ledger_and_becomes_memory_and_monitor_view(self):
        from experiments.qwen_toy.monitor import Monitor
        from rsimonitor import project_snapshot
        with TemporaryDirectory() as directory:
            context, _, loop = integration.FeedbackLearningIntegrationTests.assembly(self, directory)
            harness = loop.harness
            backend = Backend()
            with patch.object(context.backends['fixture'], 'generate', side_effect=backend.generate):
                for ident in ('change', 'evaluate'):
                    directive = Directive('directive-' + ident, 'objective', 'Concrete fixture action')
                    decision = Decision('explore', decision_id=ident, directive=directive,
                                        reason='Change one factor and compare against incumbent')
                    with patch.object(harness.policy, 'decide', return_value=decision):
                        harness.step('objective', loop.action_space)
                    harness.receive_feedback(directive, Feedback(directive.directive_id,
                        ExecutionStatus.COMPLETED, 'Observed fixture only', True))
                    harness._context('objective')
            records = deepcopy(harness.state.decision_records)
            view = harness.attempt_interpretations()
            self.assertEqual(len(view['items']), 1)
            self.assertEqual(view['items'][0]['member_decision_ids'], ['change', 'evaluate'])
            self.assertTrue(all(r['research_attempt_id'] is None for r in records.values()))
            self.assertEqual(records, harness.state.decision_records)
            memories = [json.loads(m.content) for m in harness._context('objective').memories]
            self.assertTrue(any(m.get('attempt_interpretations') for m in memories))
            monitor = Monitor(Path(directory), context.roles['actor'], 'objective')
            monitor.sync_attempt_interpretations(view)
            snapshot = project_snapshot(monitor.data, 'fixture')
            self.assertEqual(snapshot['research_attempts']['items'][0]['attempt_id'], view['items'][0]['attempt_id'])
            self.assertEqual(records, harness.state.decision_records)
            context.close()
