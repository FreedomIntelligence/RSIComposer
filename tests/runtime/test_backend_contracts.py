"""Bounded opt-in backend contracts, with no concrete backend dependencies."""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import unittest
from typing import get_type_hints

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.backend import (
    BackendCapabilities, BackendCapabilityProvider, DeploymentBackend,
    DeploymentRef, DeploymentState, DeploymentStatus, ReattachableDeploymentBackend,
    require_checkpoint, require_reattach, validate_training_capabilities,
    validate_update_capabilities,
)
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.training import ResumeMode, TrainingOrigin, TrainingResult


MODEL = ModelRef('trainer', 'initial', 'weights-v2')
CHECKPOINT = ArtifactRef('checkpoint', 'training_state', 's3://external/run/checkpoint')


class TrainingOriginTests(unittest.TestCase):
    def test_ancestry_is_independent_of_initial_weights(self):
        origin = TrainingOrigin('experiment-v19', MODEL)
        self.assertNotEqual(origin.parent_system, origin.initial_model.revision)
        self.assertEqual(origin.resume_mode, ResumeMode.RESTART)
        self.assertIsNone(origin.checkpoint)
        self.assertEqual(json.loads(json.dumps(asdict(origin)))['resume_mode'], 'restart')
        with self.assertRaises(FrozenInstanceError):
            origin.parent_system = 'different'

    def test_resume_modes_have_distinct_checkpoint_requirements(self):
        with self.assertRaisesRegex(ValueError, 'restart'):
            TrainingOrigin('parent', MODEL, checkpoint=CHECKPOINT)
        with self.assertRaisesRegex(ValueError, 'full_state'):
            TrainingOrigin('parent', MODEL, ResumeMode.FULL_STATE)
        for mode, checkpoint in ((ResumeMode.WEIGHTS_ONLY, None),
                                 (ResumeMode.WEIGHTS_ONLY, CHECKPOINT),
                                 (ResumeMode.FULL_STATE, CHECKPOINT)):
            with self.subTest(mode=mode, checkpoint=checkpoint):
                origin = TrainingOrigin('parent', MODEL, mode, checkpoint)
                self.assertEqual(origin.resume_mode, mode)
                self.assertEqual(origin.checkpoint, checkpoint)

    def test_invalid_origin_cannot_look_like_a_supported_resume(self):
        origin = TrainingOrigin('parent', MODEL)
        invalid = {
            'parent_system': (None, '', '  ', 7, MODEL),
            'initial_model': (None, 'model', replace(MODEL, revision='')),
            'resume_mode': ('restart', 'full_state', None, True),
            'checkpoint': ('uri', {}, MODEL),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                    replace(origin, **{name: value})

    def test_existing_result_positional_construction_is_preserved(self):
        candidate = Candidate('candidate', MODEL.revision, {'weights': CHECKPOINT}, ('actor',))
        result = TrainingResult(MODEL, replace(MODEL, revision='weights-v3'), candidate, {'loss': .1}, True)
        self.assertTrue(result.simulated)
        self.assertEqual(result.candidate.base_version, MODEL.revision)


class BackendCapabilityTests(unittest.TestCase):
    def test_opt_in_does_not_change_old_learners(self):
        class LegacyLearner:
            pass

        class DeclaredLearner:
            def capabilities(self):
                return BackendCapabilities(methods=('custom-algorithm',))

        self.assertNotIsInstance(LegacyLearner(), BackendCapabilityProvider)
        self.assertIsInstance(DeclaredLearner(), BackendCapabilityProvider)
        validate_training_capabilities(DeclaredLearner().capabilities(), method='custom-algorithm')

    def test_unsupported_operations_fail_explicitly(self):
        capabilities = BackendCapabilities(methods=('sft',), update_fields=('learning_rate',))
        validate_training_capabilities(capabilities, method='sft')
        validate_update_capabilities(capabilities, ('learning_rate',))
        unsupported = (
            lambda: validate_training_capabilities(capabilities, method='rl'),
            lambda: validate_training_capabilities(capabilities, method='sft', origin=TrainingOrigin(
                'parent', MODEL, ResumeMode.WEIGHTS_ONLY)),
            lambda: validate_update_capabilities(capabilities, ('learning_rate', 'optimizer')),
            lambda: require_checkpoint(capabilities),
            lambda: require_reattach(capabilities),
        )
        for operation in unsupported:
            with self.subTest(operation=operation), self.assertRaises(NotImplementedError):
                operation()

    def test_checkpoint_input_is_independent_of_checkpoint_export(self):
        capabilities = BackendCapabilities(methods=('rl',), resume_modes=(ResumeMode.FULL_STATE,),
                                           supports_reattach=True)
        validate_training_capabilities(capabilities, method='rl', origin=TrainingOrigin(
            'parent', MODEL, ResumeMode.FULL_STATE, CHECKPOINT))
        require_reattach(capabilities)
        with self.assertRaises(NotImplementedError):
            require_checkpoint(capabilities)
        with self.assertRaises(NotImplementedError):
            validate_training_capabilities(capabilities, method='rl')
        require_checkpoint(replace(capabilities, supports_checkpoint=True))

    def test_empty_declarations_are_not_wildcards(self):
        with self.assertRaises(NotImplementedError):
            validate_training_capabilities(BackendCapabilities(), method='sft')
        with self.assertRaises(NotImplementedError):
            validate_training_capabilities(BackendCapabilities(methods=('sft',), resume_modes=()), method='sft')

    def test_declarations_are_immutable_and_strictly_typed(self):
        capabilities = BackendCapabilities()
        invalid = {
            'methods': (None, [], 'sft', ('sft', 'sft'), ('',), ([],), (False,)),
            'update_fields': (None, ['lr'], (' ',), ('lr', 'lr'), ({},)),
            'resume_modes': (None, [], ('restart',), (ResumeMode.RESTART, ResumeMode.RESTART), ([],)),
            'supports_checkpoint': (None, 1, 'yes'),
            'supports_reattach': (None, 0, 'no'),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                    replace(capabilities, **{name: value})
        with self.assertRaises(FrozenInstanceError):
            capabilities.methods = ('rl',)

    def test_helpers_reject_malformed_requests_before_backend_calls(self):
        capabilities = BackendCapabilities(methods=('sft',))
        for method in (None, '', ' ', 7):
            with self.subTest(method=method), self.assertRaises(ValueError):
                validate_training_capabilities(capabilities, method=method)
        with self.assertRaises(TypeError):
            validate_training_capabilities(capabilities, method='sft', origin={})
        for fields in ((), [], ('',), ('lr', 'lr'), ({},)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_update_capabilities(capabilities, fields)
        with self.assertRaises(TypeError):
            require_checkpoint(None)


class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        self.ref = DeploymentRef('deployment-7', 'serving')
        self.job = JobRef('prepare-7', 'serving-jobs')
        self.state = DeploymentState(self.ref, MODEL, DeploymentStatus.PREPARING, self.job)

    def test_readiness_is_explicit_and_preserves_external_identity(self):
        for status in DeploymentStatus:
            with self.subTest(status=status):
                state = replace(self.state, status=status,
                                error='failed to load' if status == DeploymentStatus.FAILED else None)
                self.assertEqual(state.ready, status == DeploymentStatus.READY)
                self.assertEqual(state.deployment, self.ref)
                self.assertEqual(state.job, self.job)
                self.assertEqual(state.model, MODEL)
        unknown = replace(self.state, status=DeploymentStatus.UNKNOWN, error='transport unavailable')
        self.assertFalse(unknown.ready)
        with self.assertRaises(FrozenInstanceError):
            self.state.status = DeploymentStatus.READY

    def test_bad_identity_status_and_success_errors_are_rejected(self):
        invalid = {
            'deployment': (None, self.job),
            'model': (None, 'model', replace(MODEL, model_id=' ')),
            'status': ('ready', None, True),
            'job': ('job', {}, JobRef('', 'backend'), JobRef('job', '')),
            'error': ('', ' ', 1, 'not valid while preparing'),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                    replace(self.state, **{name: value})
        with self.assertRaises(ValueError):
            replace(self.state, status=DeploymentStatus.FAILED)
        with self.assertRaises(ValueError):
            replace(self.state, status=DeploymentStatus.READY, error='load failed')
        for name in ('deployment_id', 'backend'):
            for value in (None, '', ' ', 1):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.ref, **{name: value})

    def test_protocols_expose_typed_external_refs_without_harness_dependencies(self):
        for method in ('prepare', 'query', 'release'):
            self.assertIs(get_type_hints(getattr(DeploymentBackend, method))['return'], DeploymentState)
        hints = get_type_hints(ReattachableDeploymentBackend.reattach)
        self.assertIs(hints['deployment'], DeploymentRef)
        self.assertIs(hints['return'], DeploymentState)


if __name__ == '__main__':
    unittest.main()
