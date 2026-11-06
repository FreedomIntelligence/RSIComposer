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
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import ModelRef, GenerationResponse
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.state import DecisionContext
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

    def read_page(self, cursor=None):
        args = {'collection': 'history', 'record_id': 'decision-1:operation', 'cursor': cursor,
                'evidence_path': None, 'limit': 8}
        result = self.tools._archive(args)
        ident = f'archive-{len(self.memory.history())}'
        self.memory.append(HistoryEntry(ident, 'operation', None, result.summary, False,
            details={'operation': {'name': 'research_archive', 'arguments': args}, 'result': result.payload()}))
        return result.data['next_cursor']

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


if __name__ == '__main__':
    unittest.main()
