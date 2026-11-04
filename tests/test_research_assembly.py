"""Research composition and launch boundaries; no live model or training."""
from dataclasses import asdict, replace
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.application import Application, ApplicationConfig
from rsisys.config import build_demo
from rsisys.loops.decision_making.feedback import JournalContextMemory, ModelHarnessProposer
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.snapshot.local import LocalSnapshotCapture
from tests.test_feedback_learning_integration import Backend, ImmediateExecutor
from tests.test_feedback_required_outputs import CONFIG, observation


class ResearchAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.backend = Backend()

    def install(self, assembly=None, **options):
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        service = assemble_research(runner.decision_making,
            assembly=ResearchAssembly() if assembly is None else assembly,
            backend=self.backend, path=self.root / 'feedback.json', owner=runner.assembly,
            executor=ImmediateExecutor(), **options)
        return runner, service

    def test_memory_cannot_be_disabled_and_components_are_validated_before_calls(self):
        for provider in (None, object()):
            with self.subTest(provider=provider), self.assertRaisesRegex(TypeError, 'active context/memory'):
                ResearchAssembly(context_memory=provider)
        with self.assertRaisesRegex(TypeError, '[Tt]rajectory archive'):
            ResearchAssembly(trajectory_archive=object())
        with self.assertRaisesRegex(TypeError, 'process\\(event, model\\)'):
            ResearchAssembly(harness_proposals=object())
        self.assertFalse(self.backend.requests)
        self.assertFalse((self.root / 'feedback.json').exists())

    def test_journal_memory_and_default_extraction_need_no_model_inference(self):
        runner, service = self.install()
        ident = service.observe(observation('one'))
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed', row)
        self.assertEqual(row['products']['context_memory']['charged_calls'], 0)
        self.assertEqual(service.learning.state['attempted'], 0)
        self.assertFalse(self.backend.requests)
        self.assertTrue(service.working_memory())
        trace = service.extract_trajectory(ident)
        self.assertFalse(trace['training_ready'])
        self.assertFalse(trace['complete_conversation'])
        self.assertEqual(len(service.trajectories()), 1)
        config = json.loads((self.root / 'research_assembly.json').read_text())
        self.assertEqual(config, runner.decision_making.research_assembly)
        self.assertEqual(config, json.loads((self.root / 'feedback.json').read_text())['assembly'])
        self.assertIsNone(config['harness_proposals'])

    def test_config_is_persisted_before_first_receipt(self):
        runner, service = self.install()
        saved = json.loads((self.root / 'feedback.json').read_text())
        self.assertEqual(saved['jobs'], {})
        self.assertEqual(saved['assembly']['budget_owner'], 'host')
        runner.close()
        self.assertTrue(service.learning._closed)

    def test_disabled_archive_preserves_memory_and_original_receipt(self):
        _, service = self.install(ResearchAssembly(trajectory_archive=None))
        ident = service.observe(observation('no-trace'))
        service.refresh()
        self.assertEqual(service.trajectories(), [])
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'completed')
        self.assertIn('receipt', service.learning.state['jobs'][ident]['raw_input'])
        self.assertIsNone(service.learning.trajectory_archive)

    def test_optional_proposals_run_in_pipeline_and_remain_unapplied(self):
        _, service = self.install(ResearchAssembly(
            harness_proposals=ModelHarnessProposer(self.root / 'proposals')))
        ident = service.observe(observation('proposal'))
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed', row)
        self.assertEqual(row['products']['context_memory']['charged_calls'], 0)
        self.assertEqual(row['products']['harness_proposals']['charged_calls'], 1)
        self.assertEqual(len(self.backend.requests), 1)
        self.assertNotIn('memory/training_data/harness_change/system_change', self.backend.requests[0].system)

    def test_duplicate_install_and_product_override_fail_before_calls(self):
        runner, _ = self.install()
        with self.assertRaisesRegex(ValueError, 'already assembled'):
            assemble_research(runner.decision_making, assembly=ResearchAssembly())
        fresh, _ = build_demo(CONFIG)
        self.addCleanup(fresh.close)
        with self.assertRaisesRegex(ValueError, 'through ResearchAssembly'):
            assemble_research(fresh.decision_making, assembly=ResearchAssembly(), processor=object())
        self.assertFalse(self.backend.requests)

    def test_component_changes_cannot_reopen_an_existing_research_journal(self):
        runner, service = self.install()
        service.observe(observation('existing'))
        service.refresh()
        runner.close()
        original = (self.root / 'feedback.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            self.install(ResearchAssembly(trajectory_archive=None))
        self.assertEqual((self.root / 'feedback.json').read_bytes(), original)
        self.assertFalse(self.backend.requests)

    def test_unchanged_assembly_reopens_without_replaying_completed_work(self):
        runner, service = self.install()
        ident = service.observe(observation('same'))
        service.refresh()
        runner.close()
        _, restored = self.install()
        self.assertEqual(restored.observe(observation('same')), ident)
        restored.refresh()
        self.assertEqual(len(restored.trajectories()), 1)
        self.assertFalse(self.backend.requests)

    def test_old_journal_is_not_silently_migrated(self):
        runner, _ = self.install()
        runner.close()
        path = self.root / 'feedback.json'
        saved = json.loads(path.read_text())
        del saved['assembly']
        path.write_text(json.dumps(saved))
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'changed limits'):
            self.install()
        self.assertEqual(path.read_bytes(), original)

    def test_application_and_demo_share_the_entry_and_release_their_service(self):
        config = replace(ApplicationConfig.load(Path('configs/local.json')), data_dir=self.root / 'app')
        with Application(config, research=ResearchAssembly()) as app:
            service = app.runner.decision_making.feedback_service
            self.assertEqual(app.runner.decision_making.research_assembly['entrypoint'],
                             'rsisys.research.assemble_research')
            with self.assertRaisesRegex(TypeError, 'explicit snapshot/recovery integration'):
                LocalSnapshotCapture(app).capture('unsupported')
        self.assertTrue(service.learning._closed)
        with build_demo(CONFIG, research=ResearchAssembly())[0] as runner:
            service = runner.decision_making.feedback_service
            self.assertEqual(runner.decision_making.research_assembly['entrypoint'],
                             'rsisys.research.assemble_research')
        self.assertTrue(service.learning._closed)

    def test_research_durable_requires_explicit_contract_before_creating_workspace(self):
        config = replace(ApplicationConfig.load(Path('configs/local.json')), data_dir=self.root / 'blocked')
        with self.assertRaisesRegex(ValueError, 'explicit recovery_contract'):
            Application(config, research=ResearchAssembly(), durable=True)
        self.assertFalse(config.data_dir.exists())

    def test_restore_cannot_drop_research_configuration(self):
        config = replace(ApplicationConfig.load(Path('configs/local.json')), data_dir=self.root / 'restore')
        with Application(config, research=ResearchAssembly()):
            pass
        with self.assertRaisesRegex(ValueError, 'original research assembly'):
            Application(config, _restore=True)

    def test_langmem_missing_dependencies_do_not_select_journal_memory(self):
        original_import = __import__
        def missing(name, *args, **kwargs):
            if name == 'rsisys.loops.decision_making.feedback.langmem':
                raise ModuleNotFoundError('fixture missing optional dependencies')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=missing), self.assertRaisesRegex(RuntimeError, 'rsisys\\[langmem\\]'):
            ResearchAssembly.langmem(memory_path=self.root / 'memory.sqlite', namespace=('fixture', 'run'))
        self.assertFalse((self.root / 'memory.sqlite').exists())

    @unittest.skipUnless(find_spec('langmem') is not None, 'Optional LangMem dependencies are not installed')
    def test_upstream_langmem_uses_unified_pipeline_and_native_tool_calls(self):
        from tests.test_langmem_feedback import NativeFixture
        from rsisys.runtime.json_tools import result_id
        backend = NativeFixture()
        receipt = observation('native')
        backend.value = {'content': 'A recorded observation.',
            'source_refs': ['feedback:' + result_id(receipt.payload())], 'evaluation_claims': []}
        assembly = ResearchAssembly.langmem(memory_path=self.root / 'memory.sqlite', namespace=('research', 'run'))
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        service = assemble_research(runner.decision_making, assembly=assembly, backend=backend,
            path=self.root / 'native-feedback.json', owner=runner.assembly, executor=ImmediateExecutor())
        ident = service.observe(receipt)
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed', row)
        self.assertEqual(service.working_memory()['provider'], 'langmem')
        self.assertEqual(len(backend.requests), 2)
        self.assertTrue(all(request.tools for request in backend.requests))
        self.assertEqual(row['products']['context_memory']['charged_calls'], 2)
        self.assertEqual(row['model'], asdict(runner.decision_making.harness.policy.model))

    def test_required_tool_capability_is_checked_before_installation(self):
        class NativeMemory:
            max_calls_per_job, requires_tool_calling = 1, True
            configuration = lambda self: {'provider': 'fixture-native'}
            process = lambda *args: None
            recover = lambda *args: None
        fresh, _ = build_demo(CONFIG)
        self.addCleanup(fresh.close)
        with self.assertRaisesRegex(ValueError, 'native tool calling'):
            assemble_research(fresh.decision_making, backend=self.backend,
                assembly=ResearchAssembly(memory_processor=NativeMemory()))
        self.assertIsNone(getattr(fresh.decision_making, 'feedback_service', None))
        self.assertFalse(self.backend.requests)

    def test_external_memory_and_archive_remain_replaceable_at_the_unified_entry(self):
        from tests.test_feedback_components import ExternalContext, ExternalArchive
        memory, archive = ExternalContext(), ExternalArchive()
        _, service = self.install(ResearchAssembly(context_memory=memory, trajectory_archive=archive))
        service.observe(observation('external'))
        service.refresh()
        self.assertEqual(service.working_memory()['provider'], 'external')
        self.assertEqual(service.trajectories()[0]['provider'], 'external-archive')
        self.assertIs(service.learning.context_memory, memory)
        self.assertFalse(self.backend.requests)

    @unittest.skipUnless(find_spec('langmem') is not None, 'Optional LangMem dependencies are not installed')
    def test_langmem_configuration_does_not_create_a_workspace(self):
        path = self.root / 'unclaimed' / 'memory.sqlite'
        research = ResearchAssembly.langmem(memory_path=path, namespace=('research', 'run'))
        self.assertEqual(research.memory_provider, 'langmem')
        research.validate()
        self.assertFalse(path.parent.exists())


if __name__ == '__main__':
    unittest.main()
