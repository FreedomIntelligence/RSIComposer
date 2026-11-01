import unittest
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.training import ModelActivationPending
from rsisys.runtime.systems import SystemRegistry, ReferenceSystemActivator


class SystemVersionTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelRef('service', 'weights', 'base')
        self.ref = ArtifactRef('prompt', 'prompt', 'file:///prompt.json')
        self.initial = SystemVersion('system-0', {}, self.model)
        self.registry = SystemRegistry(self.initial, ReferenceSystemActivator(self.model))

    def candidate(self, ident='prompt-1'):
        return Candidate(ident, 'base', {'prompt': self.ref}, ('actor',), 'system-0')

    def test_prompt_only_application_preserves_weights_and_stale_sibling_rejected(self):
        self.registry.register(self.candidate())
        self.registry.register(self.candidate('prompt-2'))
        new = self.registry.activate('prompt-1', idempotency_key='apply-1')
        self.assertEqual(new.model, self.model)
        self.assertEqual(new.components, {'prompt': self.ref})
        self.assertEqual(self.registry.activate('prompt-1', idempotency_key='apply-1'), new)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.registry.activate('prompt-2', idempotency_key='apply-2')
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.activate('prompt-2', idempotency_key='apply-1')

    def test_pending_activation_retries_same_intent_only(self):
        class Delayed:
            def __init__(self): self.calls = []
            def activate(self, system, *, idempotency_key):
                self.calls.append(idempotency_key)
                if len(self.calls) == 1: raise ModelActivationPending('loading')
                return system
        provider = Delayed()
        registry = SystemRegistry(self.initial, provider)
        registry.register(self.candidate())
        with self.assertRaises(ModelActivationPending):
            registry.activate('prompt-1', idempotency_key='same')
        self.assertEqual(registry.current, self.initial)
        with self.assertRaisesRegex(RuntimeError, 'unconfirmed'):
            registry.activate('prompt-1', idempotency_key='different')
        registry.activate('prompt-1', idempotency_key='same')
        self.assertEqual(provider.calls, ['same', 'same'])

    def test_reference_activation_cannot_claim_weight_deployment(self):
        self.registry.register(self.candidate(), model=ModelRef('other', 'weights', 'new'))
        with self.assertRaisesRegex(ValueError, 'deployment'):
            self.registry.activate('prompt-1', idempotency_key='k')
        self.assertEqual(self.registry.current, self.initial)
        self.registry.register(self.candidate('valid-prompt'))
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.activate('valid-prompt', idempotency_key='k')
        applied = self.registry.activate('valid-prompt', idempotency_key='valid-key')
        self.assertEqual(applied.model, self.model)

    def test_defensive_copy_and_identity_conflicts(self):
        candidate = self.candidate()
        self.registry.register(candidate)
        candidate.changes.clear()
        self.assertEqual(self.registry.get('prompt-1').components, {'prompt': self.ref})
        self.registry.get('prompt-1').components.clear()
        self.assertEqual(self.registry.get('prompt-1').components, {'prompt': self.ref})
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.registry.register(candidate)

    def test_parent_is_system_identity_not_training_base(self):
        version = self.registry.register(self.candidate())
        self.assertEqual(version.parent_version, 'system-0')
        self.assertEqual(version.model.revision, 'base')
        with self.assertRaisesRegex(ValueError, 'parent_system'):
            self.registry.register(Candidate('legacy', 'base', {}, ('actor',)))

    def test_system_candidate_comparison_does_not_require_a_training_job(self):
        from dataclasses import replace
        from unittest.mock import Mock
        from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
        from rsisys.loops.decision_making.harness.state import CandidateEvaluation
        from rsisys.loops.contracts.lifecycle import ExecutionStatus
        harness = DefaultDecisionHarness(Mock(), Mock(), Mock())
        for ident, score in [('p1', .6), ('p2', .7)]:
            harness.state.candidates[ident] = self.candidate(ident)
            harness.state.candidate_evaluations[ident] = CandidateEvaluation(
                ident, 'eval-' + ident, 'fixed-dev', (), ExecutionStatus.COMPLETED,
                {'accuracy': score}, False)
        rows = harness.compare_candidates(('p1', 'p2'), metric='accuracy')
        self.assertEqual(rows[0].candidate_id, 'p2')
        harness.state.candidates['p2'] = replace(harness.state.candidates['p2'], parent_system='other')
        with self.assertRaisesRegex(ValueError, 'parent system'):
            harness.compare_candidates(('p1', 'p2'), metric='accuracy')
        harness.state.candidates['p2'] = self.candidate('p2')
        harness.state.candidate_evaluations['p2'] = replace(
            harness.state.candidate_evaluations['p2'], protocol_ref='different')
        with self.assertRaisesRegex(ValueError, 'protocols'):
            harness.compare_candidates(('p1', 'p2'), metric='accuracy')
