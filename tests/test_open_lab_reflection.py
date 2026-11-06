from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from examples.medical.open_lab_reflection import PeriodicReflection
from examples.medical.open_lab_evolution import bind_evolution, EvolutionLabSlot
from rsisys.loops.contracts.candidate import SystemVersion
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import ModelRef
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
        self.harness = SimpleNamespace(state=SimpleNamespace(decision_records={
            'decision-1': {'status': 'completed', 'proposal': {'reason': 'raw failure investigation'}}}))
        self.registry = SimpleNamespace(current=SystemVersion('initial', {}))
        self.reflection.bind(self.harness, self.registry)
        self.report = {'window_id': '1', 'outcome': 'no_change', 'analysis': 'Keep evidence windows intact.',
                       'decision_ids': ['decision-1'], 'system_version': 'initial'}

    def test_cadence_noop_persistence_and_exact_idempotent_replay(self):
        with self.assertRaises(LabToolError):
            self.reflection.report(self.report)
        self.now = 2800
        self.assertTrue(self.reflection.snapshot()['due'])
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
        self.reflection.report(self.report)
        self.assertEqual(self.reflection.snapshot()['next_due_at'], 11800)
        self.assertFalse(self.reflection.snapshot()['due'])
        with self.assertRaises(RecoveryRequired):
            self.reflection.restore(saved)
        with self.assertRaises(ValueError):
            PeriodicReflection(self.reflection.path, 900)

    def test_report_requires_existing_evidence_and_actual_component_change(self):
        self.now = 2800
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
