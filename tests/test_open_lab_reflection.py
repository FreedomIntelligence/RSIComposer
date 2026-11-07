from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from examples.medical.open_lab_reflection import PeriodicReflection
from examples.medical.open_lab_evolution import bind_evolution, EvolutionLabSlot
from examples.medical.open_lab_experiment import TimedOpenPolicy
from rsisys.loops.contracts.candidate import SystemVersion
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef, GenerationResponse
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, ContextBudgetError
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.loops.decision_making.harness.policy.prompts import DecisionPrompts
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.recovery import RecoveryRequired


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = 1000
        self.reflection = PeriodicReflection(self.root / 'reflection.json', 1800, clock=lambda: self.now)
        self.memory = DecisionMemory()
        self.memory.append(HistoryEntry('decision-1:operation', 'operation', None, 'Original failure', False,
            details={'decision_id': 'decision-1', 'operation': {'name': 'lab_poll', 'arguments': {}},
                     'result': {'status': 'failed', 'error': 'raw failure ' * 1200}}))
        self.processors = []
        self.harness = SimpleNamespace(state=SimpleNamespace(decision_records={
            'decision-1': {'status': 'completed', 'proposal': {'reason': 'raw failure investigation'}}}),
            memory=self.memory, add_context_processor=self.processors.append)
        self.tools = ResearchTools(ResearchAgenda(), self.memory)
        self.registry = SimpleNamespace(current=SystemVersion('initial', {}))
        self.reflection.bind(self.harness, self.registry)
        self.report = {'window_id': '1', 'outcome': 'no_change', 'analysis': 'Keep evidence windows intact.',
                       'decision_ids': ['decision-1'], 'system_version': 'initial'}

    def read_page(self, cursor=None, record_id='decision-1:operation'):
        args = {'collection': 'history', 'record_id': record_id, 'cursor': cursor,
                'evidence_path': None, 'limit': 8}
        result = self.tools._archive(args)
        ident = f'archive-{len(self.memory.history())}'
        self.memory.append(HistoryEntry(ident, 'operation', None, result.summary, False,
            details={'operation': {'name': 'research_archive', 'arguments': args}, 'result': result.payload()}))
        return result.data['next_cursor']

    def test_old_citations_cannot_replace_recent_failure_coverage(self):
        self.now = 2800
        self.read_all()
        self.harness.state.decision_records['recent'] = {'objective': 'Debug current training'}
        self.memory.append(HistoryEntry('recent:operation', 'operation', None, 'Recent failure', False,
            details={'decision_id': 'recent', 'operation': {'name': 'training_result', 'arguments': {}},
                     'result': OperationResult('New failure', (), False, ExecutionStatus.FAILED,
                                               'collator mismatch').payload()}))
        self.reflection.begin({'window_id': '1'})
        self.assertEqual(self.reflection.snapshot()['required_decision_ids'], ['recent', 'decision-1'])
        with self.assertRaisesRegex(LabToolError, 'recent trajectory'):
            self.reflection.report(self.report)
        self.assertIsNone(self.read_page(record_id='recent:operation'))
        receipt = self.reflection.report({**self.report, 'decision_ids': ['recent', 'decision-1']}).data
        self.assertEqual(receipt['required_decision_ids'], ['recent', 'decision-1'])
        self.assertFalse(receipt['improvement_verified'])

    def test_hour_deadline_survives_restart_and_timeout_keeps_backlog(self):
        self.now = 2800
        initial = self.reflection.begin({'window_id': '1'}).data
        self.now = 4000
        self.assertEqual(self.reflection.begin({'window_id': '1'}).data, initial)
        replacement = PeriodicReflection(self.reflection.path, 1800, clock=lambda: self.now)
        replacement.bind(self.harness, self.registry)
        self.assertEqual(replacement.snapshot()['active_review']['deadline_at'], 6400)
        context = DecisionContext('Continue', (),
            (OperationSpec('lab_poll', 'Poll', {}), OperationSpec('research_archive', 'Archive', {}),
             *replacement.specifications()), ())
        in_progress = replacement.project_context(context)
        self.now = 6400
        from rsisys.loops.decision_making.harness.reflection import ReflectionTools
        from unittest.mock import Mock
        base = Mock()
        self.harness.state.pending_operation_decision_id = 'late-action'
        self.harness.state.decision_records['late-action'] = {'objective': in_progress.objective}
        rejected = ReflectionTools(base, replacement).execute(OperationCall('system_activate', {}))
        self.assertEqual(rejected.status, ExecutionStatus.CANCELLED)
        self.assertFalse(rejected.data['executed'])
        base.execute.assert_not_called()
        self.harness.state.decision_records['late-action']['objective'] = 'Ordinary research'
        ReflectionTools(base, replacement).execute(OperationCall('lab_poll', {}))
        base.execute.assert_called_once()
        expired = replacement.project_context(context)
        self.assertEqual(expired.research_agenda['reflection_attention']['status'], 'incomplete_time_limit')
        self.assertNotIn('reflect_report', [spec.name for spec in expired.operations])
        self.assertEqual(replacement.state['reports'], [])
        with self.assertRaises(LabToolError):
            replacement.report(self.report)
        with self.assertRaises(LabToolError):
            replacement.begin({'window_id': '1'})
        self.now = 8200
        self.assertEqual([spec.name for spec in replacement.project_context(context).operations], ['reflect_begin'])
        resumed = replacement.begin({'window_id': '1'}).data
        self.assertEqual(resumed['required_decision_ids'], initial['required_decision_ids'])
        self.assertEqual(len(replacement.state['incomplete_reviews']), 1)
        self.assertEqual(replacement.state['incomplete_reviews'][0]['status'], 'timed_out')

    def test_old_read_evidence_does_not_open_report_before_recent_required_pages(self):
        self.now = 2800
        self.read_all()
        self.harness.state.decision_records['recent'] = {'objective': 'Investigate recent failure'}
        self.memory.append(HistoryEntry('recent:operation', 'operation', None, 'Recent failure', False,
            details={'decision_id': 'recent', 'operation': {'name': 'lab_result', 'arguments': {}},
                'result': OperationResult('Failed', (), False, ExecutionStatus.FAILED, 'loader error').payload()}))
        self.reflection.begin({'window_id': '1'})
        context = DecisionContext('Continue', (),
            (OperationSpec('research_archive', 'Archive', {}), *self.reflection.specifications()), ())
        due = self.reflection.project_context(context)
        self.assertEqual([spec.name for spec in due.operations], ['research_archive'])
        review = due.research_agenda['scheduled_reflection']
        self.assertEqual(review['next_read']['arguments'], {'collection': 'history',
            'record_id': 'recent:operation', 'evidence_path': None, 'cursor': None, 'limit': 8})
        self.assertFalse(review['evidence_progress']['report_ready'])
        self.read_page(record_id='recent:operation')
        ready = self.reflection.project_context(context)
        self.assertIn('reflect_report', [spec.name for spec in ready.operations])
        self.assertEqual(ready.research_agenda['scheduled_reflection']['report_arguments']['decision_ids'], ['recent', 'decision-1'])

    def test_partial_field_reads_provide_a_full_record_repair_path(self):
        self.now = 2800
        self.reflection.begin({'window_id': '1'})
        args = {'collection': 'history', 'record_id': 'decision-1:operation',
                'evidence_path': '/details/result', 'limit': 8}
        result = self.tools._archive(args)
        self.memory.append(HistoryEntry('partial', 'operation', None, result.summary, False,
            details={'operation': {'name': 'research_archive', 'arguments': args}, 'result': result.payload()}))
        with self.assertRaises(LabToolError) as raised:
            self.reflection.report(self.report)
        data = raised.exception.data
        self.assertEqual(data['reason'], 'raw_evidence_incomplete')
        next_read = data['next_reads'][0]['arguments']
        self.assertIsNone(next_read['evidence_path'])
        self.assertIsNone(next_read['cursor'])
        self.assertEqual(next_read['record_id'], 'decision-1:operation')
        self.assertEqual(self.reflection.state['reports'], [])

    def test_navigation_resumes_furthest_verified_cursor_after_first_page_reread(self):
        self.now = 2800
        self.reflection.begin({'window_id': '1'})
        first = self.read_page()
        second = self.read_page(first)
        self.assertIsNotNone(second)
        self.read_page()  # The real model repeatedly restarted at this page.
        context = DecisionContext('Continue', (),
            (OperationSpec('research_archive', 'Archive', {}), *self.reflection.specifications()), ())
        next_read = self.reflection.project_context(context).research_agenda['scheduled_reflection']['next_read']
        self.assertEqual(next_read['arguments']['cursor'], second)
        cursor = next_read['arguments']['cursor']
        while cursor is not None:
            cursor = self.read_page(cursor)
        ready = self.reflection.project_context(context)
        report = next(spec for spec in ready.operations if spec.name == 'reflect_report')
        self.assertEqual(report.input_schema['properties']['decision_ids']['maxItems'], 1)
        self.assertEqual(self.reflection.report(self.report).data['outcome'], 'no_change')

    def test_eighty_operations_do_not_remove_build_test_or_activation_ports(self):
        self.now = 2800
        self.reflection.begin({'window_id': '1'})
        self.read_all()
        for i in range(80):
            ident = f'review-read-{i}'
            self.harness.state.decision_records[ident] = {
                'objective': 'Scheduled harness reflection window 1. Review slot 1.'}
            self.memory.append(HistoryEntry(ident, 'operation', None, 'Read evidence', False,
                details={'decision_id': ident, 'operation': {'name': 'research_archive', 'arguments': {}},
                         'result': OperationResult('Read', (), False).payload()}))
        context = DecisionContext('Continue', (), tuple(OperationSpec(name, name, {}) for name in
            ('research_archive', 'lab_write_file', 'lab_start', 'lab_poll', 'lab_result', 'lab_export', 'system_register', 'system_activate'))
            + self.reflection.specifications(), ())
        due = self.reflection.project_context(context)
        self.assertTrue({'lab_write_file', 'lab_start', 'lab_poll', 'lab_result', 'lab_export', 'system_register', 'system_activate'} <= {s.name for s in due.operations})
        self.assertIsNone(due.research_agenda['scheduled_reflection']['operations_remaining'])

    def read_all(self):
        cursor = self.read_page()
        while cursor is not None:
            cursor = self.read_page(cursor)

    def test_review_snapshot_covers_all_old_failures_and_does_not_chase_new_arrivals(self):
        for i in range(40):
            ident = f'old-failure-{i}'
            self.harness.state.decision_records[ident] = {'objective': 'Research'}
            self.memory.append(HistoryEntry(ident + ':operation', 'operation', None, ident, False,
                details={'decision_id': ident, 'operation': {'name': 'custom_result', 'arguments': {}},
                         'result': OperationResult('Failed', (), False, ExecutionStatus.FAILED, ident).payload()}))
        self.now = 2800
        started = self.reflection.begin({'window_id': '1'}).data
        self.assertEqual(len(started['required_decision_ids']), 41)
        self.harness.state.decision_records['arrived-later'] = {'objective': self.reflection._review_prefix()}
        self.memory.append(HistoryEntry('arrived-later:operation', 'operation', None, 'New result', False,
            details={'decision_id': 'arrived-later', 'operation': {'name': 'custom_result', 'arguments': {}},
                     'result': OperationResult('Candidate test failed', (), False,
                                               ExecutionStatus.FAILED, 'test assertion').payload()}))
        for ident in started['required_decision_ids']:
            cursor = self.read_page(record_id=ident + ':operation')
            while cursor:
                cursor = self.read_page(cursor, record_id=ident + ':operation')
        self.assertEqual(self.reflection.snapshot()['required_decision_ids'], started['required_decision_ids'])
        result = self.reflection.report({**self.report, 'decision_ids': started['required_decision_ids']})
        self.assertEqual(len(result.data['archive_evidence']), 41)
        self.now = 4600
        next_review = self.reflection.begin({'window_id': '2'}).data
        self.assertEqual(next_review['required_decision_ids'], ['arrived-later'])

    def test_review_time_is_capped_by_absolute_host_deadline(self):
        self.now = 2800
        reflection = PeriodicReflection(self.root / 'short.json', 1800, clock=lambda: self.now,
            started_at=1000, deadline_epoch=2900)
        reflection.bind(self.harness, self.registry)
        self.assertEqual(reflection.begin({'window_id': '1'}).data['deadline_at'], 2900)
        self.now = 2900
        with self.assertRaises(LabToolError):
            reflection.begin({'window_id': '1'})
        for value in (0, True, 3601):
            with self.assertRaises(ValueError):
                PeriodicReflection(self.root / 'bad-limit.json', 1800, max_review_seconds=value)

    def test_reflection_preserves_memory_consolidation_and_custom_host_capabilities(self):
        self.now = 2800
        self.reflection.begin({'window_id': '1'})
        self.read_all()
        context = DecisionContext('Research', (),
            (OperationSpec('custom_candidate_test', 'Authorized test', {}),
             OperationSpec('memory_review', 'Disposition', {}), OperationSpec('research_archive', 'Read', {})), (),
            research_agenda={'memory_consolidation': {'pending_count': 1}})
        projected = self.reflection.project_context(context)
        self.assertEqual({s.name for s in projected.operations}, {s.name for s in context.operations})
        self.harness.agent_memory = SimpleNamespace(require_review=True, pending_review=lambda: {'pending_count': 1})
        with self.assertRaisesRegex(LabToolError, 'Memory'):
            self.reflection.report(self.report)

    def test_cadence_noop_persistence_and_exact_idempotent_replay(self):
        with self.assertRaises(LabToolError):
            self.reflection.report(self.report)
        self.now = 2800
        self.reflection.begin({'window_id': '1'})
        self.assertTrue(self.reflection.snapshot()['due'])
        self.read_all()
        receipt = self.reflection.report(self.report).data
        self.assertFalse(receipt['improvement_verified'])
        self.assertEqual(len(receipt['decision_record_sha256']['decision-1']), 64)
        self.assertEqual(self.reflection.snapshot()['next_due_at'], 4600)
        replacement = PeriodicReflection(self.reflection.path, 1800, clock=lambda: self.now)
        replacement.bind(self.harness, self.registry)
        replacement.restore(self.reflection.capture())
        self.assertEqual(replacement.report(self.report).data, receipt)
        with self.assertRaises(LabToolError):
            replacement.report({**self.report, 'analysis': 'conflicting replay'})

    def test_late_review_does_not_burst_and_unknown_receipt_blocks_restore(self):
        saved = self.reflection.capture()
        self.now = 10000
        self.reflection.begin({'window_id': '1'})
        self.read_all()
        self.reflection.report(self.report)
        self.assertEqual(self.reflection.snapshot()['next_due_at'], 11800)
        self.assertFalse(self.reflection.snapshot()['due'])
        with self.assertRaises(RecoveryRequired):
            self.reflection.restore(saved)
        with self.assertRaises(ValueError):
            PeriodicReflection(self.reflection.path, 900)

    def test_report_requires_existing_evidence_and_actual_component_change(self):
        self.now = 2800
        self.reflection.begin({'window_id': '1'})
        self.read_all()
        for changed in ({'decision_ids': ['invented']}, {'outcome': 'changed'},
                        {'system_version': 'invented'}, {'analysis': ''}):
            with self.assertRaises(LabToolError):
                self.reflection.report({**self.report, **changed})
        artifacts = InMemoryArtifactStore()
        ref = artifacts.put(b'{}', kind='system_component')
        self.registry.current = SystemVersion('actual-activation', {'meta.harness': ref})
        receipt = self.reflection.report({**self.report, 'outcome': 'changed',
                                         'system_version': 'actual-activation'}).data
        self.assertTrue(receipt['activation_verified'])
        self.assertFalse(receipt['improvement_verified'])

    def test_repair_attempt_preserves_original_cadence_and_reviews_once(self):
        self.now = 10000
        replacement = PeriodicReflection(self.root / 'new-attempt.json', 1800,
            clock=lambda: self.now, started_at=1000)
        replacement.bind(self.harness, self.registry)
        self.assertTrue(replacement.snapshot()['due'])
        self.read_all()
        replacement.begin({'window_id': '1'})
        replacement.report(self.report)
        self.assertEqual(replacement.snapshot()['next_due_at'], 11800)
        self.assertEqual(len(replacement.capture()['reports']), 1)
        with self.assertRaisesRegex(ValueError, 'recovered reflection start'):
            PeriodicReflection(replacement.path, 1800, clock=lambda: self.now, started_at=1001)
        for value in (-1, True, float('nan'), float('inf'), 10001):
            with self.assertRaises(ValueError):
                PeriodicReflection(self.root / 'invalid.json', 1800, clock=lambda: self.now, started_at=value)

    def test_due_turn_starts_timer_then_can_test_while_reading_evidence(self):
        operations = (OperationSpec('research_archive', 'Archive', {}),
            OperationSpec('lab_poll', 'Poll running Lab job', {}), *self.reflection.specifications())
        context = DecisionContext('Continue medical research', (), operations, ())
        policy = TimedOpenPolicy(SimpleNamespace(supports_response_schema=True), ModelRef('fixture', 'meta', 'v1'))
        policy.bind_reflection(self.reflection)
        self.assertIs(self.reflection.project_context(context), context)
        self.now = 2800
        before = self.reflection.capture()
        due = self.reflection.project_context(context)
        self.assertEqual([s.name for s in due.operations], ['reflect_begin'])
        self.assertEqual(self.reflection.capture(), before)  # Projection cannot start a timer.
        self.reflection.begin({'window_id': '1'})
        due = self.reflection.project_context(context)
        self.assertEqual({s.name for s in due.operations}, {'research_archive', 'lab_poll'})
        response = GenerationResponse(json.dumps({'kind': 'invoke', 'operation': 'lab_poll', 'arguments': {}}),
                                      policy.model, False)
        self.assertEqual(policy._parse_decision(response, due).operation.name, 'lab_poll')
        with self.assertRaisesRegex(LabToolError, 'complete raw operation history'):
            self.reflection.report(self.report)
        self.read_all()
        ready = self.reflection.project_context(context)
        self.assertIn('reflect_report', [s.name for s in ready.operations])
        self.assertEqual(self.reflection.report(self.report).data['outcome'], 'no_change')
        self.now = 4600
        self.assertIs(self.reflection.project_context(context), context)  # No new work to review.

    def test_harness_only_ports_activate_and_restore_without_model_mutation(self):
        model = ModelRef('meta', 'qwen38-27b', 'frozen')
        dm = self.harness
        dm.policy = SimpleNamespace(prompts=DecisionPrompts())
        dm.experience_limit, dm.memory_limit, dm.review_tool_limit, dm.max_outstanding = 12, 8, 8, 1
        dm.current_model = lambda: model
        artifacts = InMemoryArtifactStore()
        native = SimpleNamespace(describe=lambda: (), workspace=self.root, require_admission=lambda: None)
        slot = EvolutionLabSlot(native, self.reflection)
        loop = SimpleNamespace(harness=dm)
        registry = bind_evolution(SimpleNamespace(artifacts=artifacts, data_dir=self.root),
            loop, slot, object(), model, config={'meta_model_updates': False})
        self.assertEqual(registry.activator.models, {})
        self.assertEqual(registry.current.model_for('decision_maker'), model)
        self.assertIn('reflect_report', [s.name for s in slot.describe()])
        from rsisys.loops.contracts.candidate import Candidate
        ref = artifacts.put(b'{"experience_limit":16,"memory_limit":8,"review_tool_limit":8,"max_outstanding":1}',
                            kind='system_component')
        registry.register(Candidate('harness-v2', 'frozen', {'meta.harness': ref},
                                    ('decision_maker',), 'initial'))
        registry.adopt(('harness-v2',), version='v2', idempotency_key='apply')
        self.assertEqual(dm.experience_limit, 16)
        loop.evolution_state.capture()
        loop.evolution_state.restore(loop.evolution_state.capture())
        self.assertEqual(registry.current.model_for('decision_maker'), model)

    def test_report_retry_retains_archive_and_compacts_old_failures(self):
        self.now = 2800
        self.reflection.begin({'window_id': '1'})
        self.read_all()
        results = []
        for i in range(12):
            result = OperationResult('Recorded program failure', (), False,
                status=ExecutionStatus.FAILED, error=f'failure-{i} ' + 'original diagnostic ' * 800)
            results.append(result)
            self.memory.append(HistoryEntry(f'failed-{i}', 'operation', None, result.summary, False,
                details={'operation': {'name': 'lab_poll', 'arguments': {}}, 'result': result.payload()}))
        for i in range(15):
            ident = f'review-{i}'
            self.harness.state.decision_records[ident] = {
                'objective': 'Scheduled harness reflection window 1. Read original evidence.'}
            result = OperationResult('Invalid report citation', (), False,
                status=ExecutionStatus.FAILED, error='Cite existing decision IDs')
            results.append(result)
            self.memory.append(HistoryEntry(ident + ':operation', 'operation', None, result.summary, False,
                details={'decision_id': ident, 'operation': {'name': 'reflect_report', 'arguments': {}},
                         'result': result.payload()}))
        context = DecisionContext('Continue research', (),
            (*(cap.definition.spec for cap in self.tools.describe()),
             OperationSpec('lab_poll', 'Poll', {}), *self.reflection.specifications()),
            tuple(results), history=self.memory.history())
        original_history = deepcopy(self.memory.history())
        due = self.reflection.project_context(context)
        self.assertTrue({'research_archive', 'reflect_report', 'lab_poll'} <= {s.name for s in due.operations})
        config = dict(trigger_tokens=48000, target_tokens=16000, summary_tokens=2048,
                      max_target_tokens=40000)
        measure = lambda view: len(json.dumps(asdict(view)).encode())
        # The former report-only gate made recoverable failures irreducible.
        report_only = replace(due, operations=tuple(s for s in due.operations if s.name == 'reflect_report'))
        with self.assertRaises(ContextBudgetError):
            BasicContextCompactor(**config).compact(report_only, measure=measure)
        compactor = BasicContextCompactor(**config)
        compacted = compactor.compact(due, measure=measure)
        self.assertLessEqual(measure(compacted), 40000)
        self.assertEqual(compacted.operation_results[-1].payload(), results[-1].payload())
        self.assertEqual(self.memory.history(), original_history)
        receipt = self.reflection.report(self.report).data
        self.assertEqual(receipt['outcome'], 'no_change')


if __name__ == '__main__':
    unittest.main()
