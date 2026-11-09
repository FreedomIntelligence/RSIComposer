"""Real ReMe components, revision/evidence boundaries and harness delivery.

Optional runtime tests never invoke a language model, embedding or network.
Scripted harness consumers check wiring only, not autonomous research benefit.
"""
import asyncio
from dataclasses import asdict
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rsisys.config import build_demo
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.decision import OperationArgumentError, OperationCall
from rsisys.loops.contracts.feedback_context import ContextQuery, FeedbackContextEvent
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.harness.agent_memory import AgentMemory, AgentMemoryState
from rsisys.loops.decision_making.harness.reme_memory import ReMeBM25Retrieval
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.json_tools import result_id
from rsisys.runtime.recovery import RecoveryRequired
from tests.feedback.test_feedback_learning_integration import Backend, ImmediateExecutor
from tests.feedback.test_feedback_required_outputs import CONFIG
from tests.runtime import test_attempt_continuation as continuation_tests


HAS_REME = find_spec('reme') is not None


class ReMeConfigurationTests(unittest.TestCase):
    def test_opt_in_configuration_and_clone_are_side_effect_free(self):
        with patch.object(ReMeBM25Retrieval, 'prepare', side_effect=AssertionError('I/O during configuration')):
            assembly = ResearchAssembly.reme(require_memory_review=True)
            config = assembly.configuration()
            clone = assembly.context_memory.clone()
        self.assertEqual(config['memory_provider'], 'reme')
        self.assertEqual(config['agent_topology'], 'single_acting_meta_agent')
        self.assertEqual(clone.configuration(), assembly.context_memory.configuration())
        self.assertIsNot(clone._retrieval, assembly.context_memory._retrieval)
        self.assertFalse(assembly.requires_model)
        self.assertEqual(ResearchAssembly.reme(harness_proposals=None).memory_provider, 'reme')
        with self.assertRaisesRegex(ValueError, 'independent memory organizer'):
            ResearchAssembly.reme(memory_processor=None)
        with self.assertRaises(ValueError):
            AgentMemory(retrieval_backend='unknown')

    def test_missing_or_wrong_dependency_fails_before_journal_creation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'new' / 'feedback.json'
            runner, _ = build_demo(CONFIG)
            self.addCleanup(runner.close)
            with patch('rsisys.loops.decision_making.harness.reme_memory.metadata.version', return_value='wrong'):
                with self.assertRaisesRegex(RuntimeError, 'requires reme-ai'):
                    assemble_research(runner.decision_making, assembly=ResearchAssembly.reme(), path=path)
            self.assertFalse(path.parent.exists())
            from importlib.metadata import PackageNotFoundError
            with patch('rsisys.loops.decision_making.harness.reme_memory.metadata.version',
                       side_effect=PackageNotFoundError('reme-ai')):
                with self.assertRaisesRegex(RuntimeError, r'requirements/reme\.txt'):
                    ResearchAssembly.reme().prepare()

    def test_open_lab_preflight_precedes_claim_and_rejects_legacy_mixture(self):
        from experiments.medical.open_lab_experiment import main, single_meta_memory
        self.assertEqual(single_meta_memory(SimpleNamespace()).memory_provider, 'agent_managed')
        with self.assertRaisesRegex(ValueError, 'requires feedback_routing'):
            single_meta_memory(SimpleNamespace(memory_backend='reme', feedback_routing='legacy'))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'new'
            with patch.object(ReMeBM25Retrieval, 'prepare', side_effect=RuntimeError('missing optional runtime')):
                with self.assertRaisesRegex(RuntimeError, 'missing optional runtime'):
                    main(['--root', str(path), '--memory-backend', 'reme'])
            self.assertFalse(path.exists())


@unittest.skipUnless(HAS_REME, 'Optional pinned ReMe runtime not installed')
class ReMeMemoryTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runner, _ = build_demo(CONFIG)
        self.addCleanup(self.runner.close)
        self.loop = self.runner.decision_making
        self.harness = self.loop.harness
        self.backend = Backend()
        self.assembly = ResearchAssembly.reme()
        self.service = assemble_research(self.loop, assembly=self.assembly,
            path=self.root / 'feedback.json', backend=self.backend, executor=ImmediateExecutor())
        self.memory = self.harness.agent_memory
        result = OperationResult('Native adapter failure', (), False, ExecutionStatus.FAILED,
            'candidate_adapter missing', data={'stderr': 'ModuleNotFoundError: candidate_adapter'})
        self.harness.memory.append(HistoryEntry('adapter-error', 'operation', None, result.summary, False,
            details={'operation': {'name': 'lab_result', 'arguments': {}}, 'result': result.payload()}))
        # Imports are already checked. All actual retrieval/CRUD/context below
        # must work without a network client or service.
        blocker = patch('socket.socket.connect', side_effect=AssertionError('Unexpected network call'))
        blocker.start()
        self.addCleanup(blocker.stop)

    def mutate(self, name, **arguments):
        return self.memory.execute(OperationCall(name, {
            'expected_revision': self.memory.export()['revision'], **arguments}))

    def write(self, ident='adapter-paths-and-bundle-checklist', content=None):
        self.mutate('memory_write', memory_id=ident,
            content=content or 'Include candidate_adapter.py in the bundle; check adapter weights subdirectory.',
            applicability='Before loading a LoRA adapter for evaluation.',
            source_refs=['history:decision_maker:adapter-error'], status='supported')

    def ids(self, query):
        return [row['memory_id'] for row in self.memory.search(query=query)['items']]

    def test_historical_query_forms_ids_and_negative_queries(self):
        self.write()
        self.write('1epoch-incumbent', '1epoch incumbent accuracy 0.874 versus 2epoch 0.842; protocol cf51d3b7. '
                   'This is an illustrative retrieval fixture, not an evaluation claim.')
        literal = AgentMemory()
        literal.bind(self.harness)
        literal.seed(self.memory.export())
        self.assertEqual(literal.search(query='adapter paths bundle checklist weights subdir')['items'], [])
        for query in ('adapter paths bundle checklist weights subdir',
                      'adapter paths bundle checklist weights', 'adapter-paths-and-bundle-checklist'):
            self.assertEqual(self.ids(query)[0], 'adapter-paths-and-bundle-checklist')
        for query in ('1-epoch incumbent cf51d3b7 0.874', '1epoch same protocol cf51d3b7 0.874',
                      '1-epoch 0.874 2-epoch 0.842 cf51d3b7 incumbent'):
            self.assertEqual(self.ids(query)[0], '1epoch-incumbent')
        self.assertEqual(self.ids('unrelatedxyz quasarxyz'), [])
        self.assertEqual(self.ids(''), ['1epoch-incumbent', 'adapter-paths-and-bundle-checklist'])
        # BM25 is lexical: a paraphrase with no common tokens is not promised.
        self.assertEqual(self.ids('concealed parameter location'), [])

    def test_updates_deletes_recovery_and_source_digests(self):
        self.write(content='olduniquetoken path is invalid')
        original = self.memory.export()
        before_files = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertTrue(self.ids('olduniquetoken'))
        read = self.memory.execute(OperationCall('memory_read', {'memory_id': 'adapter-paths-and-bundle-checklist'}))
        note = read['memory']
        self.assertEqual(note['provenance']['history:decision_maker:adapter-error']['sha256'],
                         result_id(asdict(self.harness.memory.history()[0])))
        self.assertEqual(before_files, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.write(content='newuniquetoken location replaces the invalid path')
        self.assertEqual(self.ids('olduniquetoken'), [])
        self.assertTrue(self.ids('newuniquetoken'))
        self.mutate('memory_delete', memory_id='adapter-paths-and-bundle-checklist')
        self.assertEqual(self.ids('newuniquetoken'), [])
        historical = self.memory.execute(OperationCall('memory_read', {
            'memory_id': 'adapter-paths-and-bundle-checklist', 'revision': 1}))
        self.assertEqual(historical['memory']['content'], 'olduniquetoken path is invalid')
        with self.assertRaises(RecoveryRequired):
            self.memory.restore(original)
        reopened = self.memory.clone()
        reopened.bind(self.harness, self.root / 'feedback.json')
        self.assertEqual(reopened.export(), self.memory.export())
        self.assertEqual(reopened.search(query='newuniquetoken')['items'], [])

    def test_revision_bound_paging_and_rebuild_on_seed(self):
        for i in range(3):
            self.write('adapter-' + str(i))
        page = self.memory.search(query='adapter bundle', limit=1)
        with self.assertRaises(OperationArgumentError):
            self.memory.search(query='adapter bundle', offset=1)
        next_page = self.memory.search(query='adapter bundle', limit=1, offset=1,
                                       expected_revision=page['revision'])
        self.assertNotEqual(next_page['items'], page['items'])
        target = AgentMemory(retrieval_backend='reme')
        target.bind(self.harness)
        target.seed(self.memory.export())
        self.assertEqual(target.search(query='adapter bundle'), self.memory.search(query='adapter bundle'))
        self.write('new-adapter')
        with self.assertRaises(OperationArgumentError):
            self.memory.search(query='adapter bundle', offset=1, expected_revision=page['revision'])
        self.assertNotEqual(self.memory.search(query='adapter')['retrieval']['corpus_sha256'],
                            page['retrieval']['corpus_sha256'])
        self.assertEqual(len(target.search(query='adapter')['items']), 3)

    def test_feedback_and_tasks_drive_automatic_context_without_recency_fill(self):
        self.write()
        for i in range(8):
            self.write('other-' + str(i), 'Bookkeeping only, unrelated entry.')
        feedback = FeedbackContextEvent('event', 'job', 'failed', {'stderr': 'candidate_adapter import failed'})
        context = json.loads(self.memory.context(ContextQuery('Improve accuracy', feedback)))
        self.assertEqual(context['items'][0]['memory_id'], 'adapter-paths-and-bundle-checklist')
        self.assertFalse(context['retrieval']['usage_verified'])
        self.assertEqual(json.loads(self.memory.context(ContextQuery('quasarxyz')))['items'], [])
        self.mutate('research_task', action='open', task_id='adapter', title='Check candidate_adapter bundle',
                    next_step='Inspect weights directory', acceptance_criteria='Actual loaded weights')
        self.assertEqual(json.loads(self.memory.context(ContextQuery('Improve accuracy')))['items'][0]['memory_id'],
                         'adapter-paths-and-bundle-checklist')
        self.assertFalse(self.backend.requests)

    def test_scope_isolation_and_running_event_loop(self):
        self.write()
        other = AgentMemory(retrieval_backend='reme')
        other.bind(SimpleNamespace(memory=SimpleNamespace(history=lambda: ())))
        self.assertEqual(other.search(query='adapter')['items'], [])
        async def retrieve():
            return self.memory.search(query='adapter weights')
        self.assertTrue(asyncio.run(retrieve())['items'])
        self.assertEqual(other.export()['revision'], 0)

    def test_chinese_keywords_use_upstream_tokenizer(self):
        self.write('加载检查', '评测前核验适配器权重路径，并确认实际加载。')
        self.assertEqual(self.ids('权重 加载 路径')[0], '加载检查')

    def test_scripted_consumer_sees_recall_and_tool_result_in_next_decision(self):
        self.write()
        requests = []
        def generate(request):
            context = json.loads(request.prompt)
            requests.append(context)
            memory = json.loads(context['memory_context'])
            self.assertEqual(memory['provider'], 'reme')
            self.assertEqual(memory['items'][0]['memory_id'], 'adapter-paths-and-bundle-checklist')
            if len(requests) == 1:
                operation, args = 'memory_search', {'query': 'adapter paths bundle checklist weights subdir'}
            elif len(requests) == 2:
                hits = self.harness.state.operation_results[-1].data['items']
                operation, args = 'memory_read', {'memory_id': hits[0]['memory_id']}
            else:
                self.assertIn('candidate_adapter.py', self.harness.state.operation_results[-1].data['memory']['content'])
                return GenerationResponse('{"kind":"finish"}', request.model, True)
            return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': operation, 'arguments': args}),
                                      request.model, True)
        with patch.object(self.harness.policy.backend, 'generate', side_effect=generate):
            results = [self.harness.step('Repair adapter bundle', self.loop.action_space) for _ in range(3)]
        self.assertEqual(results[-1].kind, 'finished')
        self.assertEqual(len(requests), 3)
        spec = next(spec for spec in self.loop.action_space.describe() if spec.name == 'memory_search')
        self.assertIn('ReMe BM25', spec.description)
        self.assertTrue(spec.read_only)
        self.assertEqual(self.memory.export()['revision'], 1)

    def test_backend_failure_does_not_mutate_notes_or_return_stale_index(self):
        self.write()
        self.assertTrue(self.ids('adapter'))
        before = self.memory.export()
        with patch.object(self.memory._retrieval._index, 'retrieve', side_effect=RuntimeError('index fault')):
            with self.assertRaisesRegex(RuntimeError, 'index fault'):
                self.memory.search(query='adapter')
        self.assertEqual(self.memory.export(), before)
        port = AgentMemoryState(self.memory)
        self.assertEqual(port.capture(), before)

    def test_resolved_task_still_requires_native_receipt_acceptance(self):
        self.write()
        self.mutate('research_task', action='open', task_id='load', title='Verify adapter loading',
            acceptance_checks=[{'id': 'loaded', 'operation': 'lab_result',
                                'result_path': '/data/adapter_loaded', 'expected': True}])
        before = self.memory.export()
        with self.assertRaises(OperationArgumentError):
            self.mutate('research_task', action='close', task_id='load', resolution='resolved',
                outcome='A memory note says it works', acceptance_evidence={'loaded': 'adapter-error'})
        self.assertEqual(before, self.memory.export())
        success = OperationResult('Native fixture load', (), False, data={'adapter_loaded': True})
        self.harness.memory.append(HistoryEntry('loaded', 'operation', None, success.summary, False,
            details={'operation': {'name': 'lab_result', 'arguments': {}}, 'result': success.payload()}))
        self.mutate('research_task', action='close', task_id='load', resolution='resolved',
            outcome='Native fixture loaded; no medical gain claimed', acceptance_evidence={'loaded': 'loaded'})
        restored = self.memory.clone()
        restored.bind(self.harness, self.root / 'feedback.json')
        self.assertEqual(restored.export()['tasks']['load']['acceptance_receipt']['loaded']['record_id'], 'loaded')

    def test_dsh_receives_same_recall_and_executes_same_memory_tools(self):
        import time
        from tests.runtime.test_dsh_harness import Runtime, DSH
        from rsisys.loops.decision_making.harness.dsh import dsh_decision_factory
        runtime = Runtime([
            '{"kind":"invoke","operation":"memory_search","arguments":{"query":"adapter paths weights"}}',
            '{"kind":"invoke","operation":"memory_read","arguments":{"memory_id":"adapter"}}',
            '{"kind":"finish","reason":"fixture finished"}'])
        with patch('rsisys.loops.decision_making.harness.dsh._sdk_runtime', return_value=runtime):
            runner, _ = build_demo(CONFIG, decision_factory=dsh_decision_factory(DSH))
        self.addCleanup(runner.close)
        loop = runner.decision_making
        assemble_research(loop, assembly=ResearchAssembly.reme(), path=self.root / 'dsh' / 'feedback.json',
                          backend=Backend(), executor=ImmediateExecutor())
        memory = loop.harness.agent_memory
        memory.execute(OperationCall('memory_write', {'expected_revision': 0, 'memory_id': 'adapter',
            'content': 'Inspect adapter weights before evaluation.', 'status': 'unverified', 'source_refs': []}))
        for _ in range(3):
            deadline = time.monotonic() + 3
            while True:
                result = loop.harness.step('Repair adapter weights', loop.action_space)
                if result.kind != 'waiting':
                    break
                if time.monotonic() >= deadline:
                    self.fail('DSH fixture did not finish')
                time.sleep(.002)
        self.assertEqual(result.kind, 'finished')
        self.assertEqual(len(runtime.prompts), 3)
        self.assertIn('ReMe BM25', runtime.prompts[0])
        self.assertIn('Inspect adapter weights', runtime.prompts[0])
        self.assertEqual(loop.harness.state.operation_results[-1].data['memory']['memory_id'], 'adapter')


@unittest.skipUnless(HAS_REME, 'Optional pinned ReMe runtime not installed')
class ReMeContinuationTests(continuation_tests.ContinuationTests):
    """Run the existing evidence/artifact handoff contract with ReMe on both ends."""
    def make_harness(self, name):
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        assemble_research(runner.decision_making, assembly=ResearchAssembly.reme(),
            path=self.root / name / 'feedback.json', backend=Backend(), executor=ImmediateExecutor())
        return runner.decision_making.harness

    def test_handoff_rebuilds_retrieval_and_preserves_backend(self):
        self.freeze()
        self.install()
        page = self.new.agent_memory.search(query='tokenizer input_ids keys')
        self.assertEqual(page['items'][0]['memory_id'], 'tokenizer')
        self.assertEqual(page['retrieval']['provider'], 'reme')
        self.assertEqual(self.new.agent_memory.configuration()['provider'], 'reme')
        self.assertEqual(self.new.agent_memory.export()['tasks']['debug']['status'], 'open')


if __name__ == '__main__':
    unittest.main()
