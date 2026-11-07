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
        self.assertEqual(self.reflection.snapshot()['required_decision_ids'], ['recent'])
        with self.assertRaisesRegex(LabToolError, 'recent trajectory'):
            self.reflection.report(self.report)
        self.assertIsNone(self.read_page(record_id='recent:operation'))
        receipt = self.reflection.report({**self.report, 'decision_ids': ['recent']}).data
        self.assertEqual(receipt['required_decision_ids'], ['recent'])
        self.assertFalse(receipt['improvement_verified'])

    def test_invalid_reviews_exhaust_a_bounded_slot_and_leave_ordinary_research_available(self):
        self.now = 2800
        context = DecisionContext('Continue', (),
            (OperationSpec('lab_poll', 'Poll', {}), OperationSpec('research_archive', 'Archive', {}),
             *self.reflection.specifications()), ())
        for i in range(16):
            ident = 'invalid-' + str(i)
            self.harness.state.decision_records[ident] = {
                'objective': 'Scheduled harness reflection window 1. Review slot 1.'}
            self.memory.append(HistoryEntry(ident, 'operation', None, 'Invalid report', False,
                details={'decision_id': ident, 'operation': {'name': 'reflect_report', 'arguments': {}},
                         'result': OperationResult('Rejected', (), False, ExecutionStatus.FAILED,
                                                   'duplicate decision IDs').payload()}))
        projected = self.reflection.project_context(context)
        self.assertEqual(projected.objective, 'Continue')
        self.assertIn('lab_poll', [spec.name for spec in projected.operations])
        self.assertNotIn('scheduled_reflection', projected.research_agenda)
        self.assertEqual(projected.research_agenda['reflection_attention']['status'], 'incomplete_budget_exhausted')
        self.assertEqual(self.reflection.state['reports'], [])
        self.now = 4600
        self.assertIn('scheduled_reflection', self.reflection.project_context(context).research_agenda)

    def read_all(self):
        cursor = self.read_page()
        while cursor is not None:
            cursor = self.read_page(cursor)

    def test_cadence_noop_persistence_and_exact_idempotent_replay(self):
        with self.assertRaises(LabToolError):
            self.reflection.report(self.report)
        self.now = 2800
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
        replacement.report(self.report)
        self.assertEqual(replacement.snapshot()['next_due_at'], 11800)
        self.assertEqual(len(replacement.capture()['reports']), 1)
        with self.assertRaisesRegex(ValueError, 'recovered reflection start'):
            PeriodicReflection(replacement.path, 1800, clock=lambda: self.now, started_at=1001)
        for value in (-1, True, float('nan'), float('inf'), 10001):
            with self.assertRaises(ValueError):
                PeriodicReflection(self.root / 'invalid.json', 1800, clock=lambda: self.now, started_at=value)

    def test_due_turn_cannot_poll_or_report_without_complete_raw_archive_pages(self):
        operations = (OperationSpec('research_archive', 'Archive', {}),
            OperationSpec('lab_poll', 'Poll running Lab job', {}), *self.reflection.specifications())
        context = DecisionContext('Continue medical research', (), operations, ())
        policy = TimedOpenPolicy(SimpleNamespace(supports_response_schema=True), ModelRef('fixture', 'meta', 'v1'))
        policy.bind_reflection(self.reflection)
        self.assertIs(self.reflection.project_context(context), context)
        self.now = 2800
        due = self.reflection.project_context(context)
        self.assertEqual([s.name for s in due.operations], ['research_archive'])
        self.assertIn('Scheduled harness reflection window 1', due.objective)
        self.assertEqual({b['properties']['operation']['const'] for b in policy.decision_schema(due)['oneOf']},
                         {'research_archive'})
        response = GenerationResponse(json.dumps({'kind': 'invoke', 'operation': 'lab_poll', 'arguments': {}}),
                                      policy.model, False)
        with self.assertRaisesRegex(ValueError, 'scheduled reflection'):
            policy._parse_decision(response, due)
        cursor = self.read_page()
        self.assertIsNotNone(cursor)
        with self.assertRaisesRegex(LabToolError, 'complete raw operation history'):
            self.reflection.report(self.report)
        while cursor is not None:
            cursor = self.read_page(cursor)
        ready = self.reflection.project_context(context)
        self.assertIn('reflect_report', [s.name for s in ready.operations])
        receipt = self.reflection.report(self.report).data
        self.assertEqual(receipt['archive_evidence']['decision-1']['record_id'], 'decision-1:operation')
        self.assertIs(self.reflection.project_context(context), context)
        self.now = 4600
        self.assertEqual([s.name for s in self.reflection.project_context(context).operations], ['research_archive'])
        with self.assertRaisesRegex(LabToolError, 'complete raw operation history'):
            self.reflection.report({**self.report, 'window_id': '2'})

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
        self.read_all()
        results = []
        for i in range(12):
            result = OperationResult('Recorded program failure', (), False,
                status=ExecutionStatus.FAILED, error=f'failure-{i} ' + 'original diagnostic ' * 800)
            results.append(result)
            self.memory.append(HistoryEntry(f'failed-{i}', 'operation', None, result.summary, False,
                details={'operation': {'name': 'lab_poll', 'arguments': {}}, 'result': result.payload()}))
        for i in range(8):
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
        self.assertEqual({s.name for s in due.operations}, {'research_archive', 'reflect_report'})
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
