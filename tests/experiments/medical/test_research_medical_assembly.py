"""Domain launches use the same research products, with explicit inference."""
from dataclasses import replace
from importlib.util import find_spec
from types import SimpleNamespace
import json
import re
import unittest

from experiments.medical.experiment import ExperimentSettings, parser, run
from rsisys.loops.contracts.model_service import ModelRef
from tests.experiments.medical import test_medical_experiment as fixture


class MedicalResearchAssemblyTests(unittest.TestCase):
    setUp = fixture.MedicalExperimentIntegrationTests.setUp
    execute = fixture.MedicalExperimentIntegrationTests.execute
    read = fixture.MedicalExperimentIntegrationTests.read

    def test_default_research_installs_explicit_memory_and_archives_without_feedback_calls(self):
        result = self.execute([
            {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'CPU fixture',
             'arguments': {'instructions': 'Return one legal label.'}}, fixture.FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(result['trainings_reserved'], 0)
        assembly = self.read('research_assembly.json')
        self.assertEqual(assembly['entrypoint'], 'rsisys.research.assemble_research')
        self.assertEqual(assembly['memory_provider'], 'agent_managed')
        self.assertIsNotNone(assembly['trajectory_archive'])
        self.assertIsNone(assembly['harness_proposals'])
        journal = self.read('feedback_learning.json')
        self.assertEqual(journal['assembly'], assembly)
        self.assertEqual(journal['attempted'], 0)
        jobs = list(journal['jobs'].values())
        self.assertEqual([job['status'] for job in jobs], ['completed'])
        self.assertEqual(jobs[0]['products']['context_memory']['charged_calls'], 0)
        self.assertFalse(self.process.requests)

    def test_archive_can_be_disabled_without_disabling_required_memory(self):
        result = self.execute([fixture.FINISH], archive_trajectories=False)
        self.assertEqual(result['status'], 'completed', result)
        config = self.read('research_assembly.json')
        self.assertIsNone(config['trajectory_archive'])
        self.assertIsNotNone(config['context_memory'])
        self.assertEqual(self.read('feedback_learning.json')['jobs'], {})

    @unittest.skipUnless(find_spec('reme') is not None, 'Optional ReMe runtime is not installed')
    def test_reme_launch_is_single_meta_without_feedback_model_calls(self):
        result = self.execute([fixture.FINISH], memory_provider='reme')
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 1)
        config = self.read('research_assembly.json')
        self.assertEqual(config['memory_provider'], 'reme')
        self.assertEqual(config['agent_topology'], 'single_acting_meta_agent')
        self.assertEqual(config['context_memory']['configuration']['retrieval_backend']['model_calls'], 0)
        self.assertFalse(self.process.requests)

    def test_langmem_requires_native_transport_before_claiming_the_run(self):
        settings = ExperimentSettings(self.task, self.root / 'blocked', self.root / 'model',
                                      memory_provider='langmem')
        with self.assertRaisesRegex(ValueError, 'native tool calling'):
            run(settings, decision_backend=SimpleNamespace(generate=lambda request: None))
        self.assertFalse(settings.run_dir.exists())
        self.assertFalse(self.process.requests)

    def test_feedback_identity_mismatch_is_rejected_before_claiming_the_run(self):
        settings = ExperimentSettings(self.task, self.root / 'blocked', self.root / 'model')
        model = ModelRef('fixture', 'meta-agent', 'v1')
        with self.assertRaisesRegex(ValueError, 'active meta-agent model'):
            run(settings, decision_model=model, feedback_model=replace(model, revision='old'))
        self.assertFalse(settings.run_dir.exists())

    def test_cli_selects_memory_archive_and_proposals_independently(self):
        args = vars(parser().parse_args(['--task', str(self.task), '--run-dir', 'new-run', '--model-path', 'model',
            '--memory-provider', 'journal', '--no-trajectory-archive', '--harness-proposals']))
        args['private_exclusions'] = tuple(args['private_exclusions'])
        settings = ExperimentSettings(**args)
        self.assertEqual(settings.effective_memory_provider, 'journal')
        self.assertFalse(settings.archive_trajectories)
        self.assertTrue(settings.harness_proposals)
        reme = vars(parser().parse_args(['--task', str(self.task), '--run-dir', 'new-run',
            '--model-path', 'model', '--memory-provider', 'reme']))
        reme['private_exclusions'] = tuple(reme['private_exclusions'])
        self.assertEqual(ExperimentSettings(**reme).effective_memory_provider, 'reme')

    def test_invalid_product_configuration_is_rejected(self):
        for fields in ({'memory_provider': 'disabled'}, {'memory_provider': 'legacy'},
                       {'archive_trajectories': 0}, {'harness_proposals': 1}):
            with self.subTest(fields=fields), self.assertRaises((TypeError, ValueError)):
                ExperimentSettings(self.task, self.root / 'blocked', self.root / 'model', **fields)
        self.assertFalse((self.root / 'blocked').exists())

    @unittest.skipUnless(find_spec('langmem') is not None, 'Optional LangMem dependencies are not installed')
    def test_langmem_native_tools_are_routed_through_medical_host_accounting(self):
        from tests.memory.test_langmem_feedback import NativeFixture
        class NativeFeedback(NativeFixture):
            def generate(self, request):
                sources = re.findall(r'feedback:[a-f0-9]{64}', json.dumps(request.chat_messages()))
                if not sources:
                    raise AssertionError('Native memory request lost its execution evidence sources')
                self.value = {'content': 'A recorded CPU prompt configuration.',
                              'source_refs': [sources[0]], 'evaluation_claims': []}
                return super().generate(request)
        model = ModelRef('fixture-independent', 'meta-agent', 'pinned')
        backend = fixture.FixtureResearcher(model, self.process, [
            {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'CPU fixture',
             'arguments': {'instructions': 'Return a legal symbol.'}}, fixture.FINISH])
        feedback = NativeFeedback()
        settings = ExperimentSettings(self.task, self.root / 'run', self.root / 'fixture-model',
                                      memory_provider='langmem')
        def serve(path, **options):
            return fixture.FixtureServing(path, process=self.process, script=[], **options)
        result = run(settings, serving_factory=serve, converter_factory=lambda *a, **kw: fixture.Converter(),
            process_backend=self.process, decision_model=model, decision_backend=backend, feedback_backend=feedback,
            identity_factory=lambda path: ('fixture-pin', {'model_id': 'fixture-actor', 'actual_weight_sha256': 'fixture'}))
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 4)  # Two decisions + extraction + consolidation.
        self.assertEqual(len(feedback.requests), 2)
        self.assertTrue(all(request.tools and request.model == model for request in feedback.requests))
        journal = self.read('feedback_learning.json')
        self.assertEqual(journal['attempted'], 2)
        row = next(iter(journal['jobs'].values()))
        self.assertEqual(row['products']['context_memory']['charged_calls'], 2)
        self.assertEqual(self.read('research_assembly.json')['memory_provider'], 'langmem')
        self.assertEqual(result['trainings_reserved'], 0)
        self.assertFalse(self.process.requests)
