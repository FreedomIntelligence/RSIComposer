"""Read-only researcher queries and artifact excerpts; no network or real training."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rsisys.assembly import build_default_decision
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import ExplorationProgress
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobProgress
from rsisys.loops.contracts.queries import QueryRequest, QueryResult, QuerySpec, MAX_QUERY_BYTES
from rsisys.loops.contracts.requests import SFTSpec
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions, builtin_directives
from rsisys.loops.decision_making.harness.tools.queries import QueryTools, ArtifactReader
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import InMemoryArtifactStore, FileArtifactStore


CONFIG = Path(__file__).resolve().parents[1] / 'configs/demo.json'


class QueryContractTests(unittest.TestCase):
    def test_invalid_limits_cursors_filters_and_large_results_fail_closed(self):
        for kwargs in ({'limit': 0}, {'limit': 101}, {'limit': True}, {'cursor': '-1'},
                       {'cursor': '10001'}, {'cursor': '../file'}, {'filters': {'x': []}},
                       {'filters': {'x': 'x' * 257}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QueryRequest('tasks', **kwargs)
        with self.assertRaises(ValueError):
            QueryRequest('tasks', filters={'private': 'yes'}).validate(QuerySpec('tasks', 'Task list'))
        for data in ({'n': float('nan')}, {'huge': 'x' * MAX_QUERY_BYTES}):
            with self.assertRaises(ValueError):
                QueryResult('tasks', 'external', 'now', 'v1', data)

    def test_adapter_uses_only_public_provider_and_validates_result(self):
        class Provider:
            def describe_queries(self):
                return (QuerySpec('status', 'External status'),)
            def query(self, request):
                self.request = request
                return QueryResult('status', 'external', 'now', 'v1', {'items': [{'state': 'running'}]})
        provider = Provider()
        tools = QueryTools(provider)
        definition, = tools.definitions
        self.assertTrue(definition.spec.read_only)
        result = tools.bindings[definition.spec.name]({})
        self.assertEqual(result.data['source'], 'external')
        self.assertEqual(result.data['version'], 'v1')
        self.assertEqual(result.data['observed_at'], 'now')
        self.assertIn('refs', result.data)
        self.assertEqual(result.data['data']['items'], [{'state': 'running'}])
        with self.assertRaises(ValueError):
            tools.query(QueryRequest('private'))
        provider.query = lambda request: QueryResult('other', 'external', 'now', 'v1')
        with self.assertRaisesRegex(ValueError, 'another query'):
            tools.bindings[definition.spec.name]({})


class ResearchQueriesTests(unittest.TestCase):
    def setUp(self):
        self.runner, self.store = build_demo(DemoConfig.load(CONFIG), with_learning=True)
        self.addCleanup(self.runner.close)
        self.actor = self.runner.exploration
        self.decision = self.runner.decision_making

    def operation(self, name, arguments=None):
        call = self.decision.action_space.validate(OperationCall('query_exploration_' + name, arguments or {}))
        return self.decision.harness.tools.execute(call)

    def test_default_assembly_exposes_queries_without_replacing_tool_type(self):
        from rsisys.loops.decision_making.harness.tools import DemoDecisionTools
        tools = self.decision.harness.tools
        self.assertIs(type(tools), DemoDecisionTools)
        self.assertIs(tools.query_tools.provider, self.actor)
        self.assertIs(tools.query_tools.artifacts, self.runner.assembly.artifacts)
        specs = self.decision.action_space.describe()
        self.assertEqual(len([spec for spec in specs if spec.read_only]), 9)
        self.assertEqual(self.operation('model').data['data']['items'][0]['model'],
                         asdict(self.actor.harness.policy.model))
        self.assertFalse(self.operation('logs').data['data']['available'])
        self.assertIn('remote backend logs', self.operation('logs').data['data']['unavailable_reason'])

    def test_external_harness_without_queries_advertises_no_phantom_operations(self):
        loop = ExplorationLoop(object(), EchoActionSpace())
        self.assertEqual(loop.describe_queries(), ())
        with self.assertRaisesRegex(NotImplementedError, 'does not support'):
            loop.query(QueryRequest('status'))
        self.runner.assembly.exploration_directives = DirectiveDefinitions(
            definition for definition in builtin_directives() if definition.spec.mode == 'execute')
        decision = build_default_decision(self.runner.assembly, loop)
        self.assertFalse(any(spec.read_only for spec in decision.action_space.describe()))
        self.assertIsNone(decision.harness.tools.query_tools)

    def test_task_pagination_logs_and_refs_are_detached_without_new_side_effects(self):
        for ident in ('one', 'two', 'three'):
            directive = Directive(ident, 'task', 'execute')
            self.actor.accept(directive)
            self.actor.advance(ident)
            self.actor.acknowledge(ident)
        before = deepcopy(self.actor.harness.state)
        count = len(self.store.records())
        first = self.actor.query(QueryRequest('tasks', limit=2))
        self.assertEqual([row['directive_id'] for row in first.data['items']], ['three', 'two'])
        self.assertEqual(first.next_cursor, '2')
        second = self.actor.query(QueryRequest('tasks', limit=2, cursor=first.next_cursor))
        self.assertEqual([row['directive_id'] for row in second.data['items']], ['one'])
        self.assertIsNone(second.next_cursor)
        logs = self.operation('logs', {'filters': {'directive_id': 'two'}})
        self.assertTrue(logs.data['data']['available'])
        self.assertNotIn('details', logs.data['data']['items'][0])
        artifacts = self.operation('artifacts', {'filters': {'directive_id': 'two'}})
        self.assertTrue(artifacts.artifacts)
        self.assertEqual(artifacts.data['refs'], tuple(asdict(ref) for ref in artifacts.artifacts))
        first.data['items'][0]['phase'] = 'mutated'
        self.assertEqual(self.actor.harness.state, before)
        self.assertEqual(len(self.store.records()), count)

    def test_metrics_reads_observed_job_sample_without_polling_or_fabricated_time(self):
        ref = self.runner.assembly.artifacts.put(b'data', kind='training_data')
        directive = Directive('train', 'task', 'train', mode='sft', payload=SFTSpec('mock'), input_refs=(ref,))
        self.actor.accept(directive)
        self.actor.advance('train')
        progress = self.actor.inspect('train')
        sampled = replace(progress, job_progress=JobProgress(step=3, total_steps=10,
                          metrics={'loss': 0.4}, observed_at=1234.5), metrics={'loss': 0.4})
        with patch.object(self.actor.harness, 'inspect', return_value=sampled), \
                patch.object(self.actor.harness.learner, 'poll', side_effect=AssertionError('No poll')):
            result = self.operation('metrics', {'filters': {'directive_id': 'train'}})
        row = result.data['data']['items'][0]
        self.assertEqual(row['job_progress']['step'], 3)
        self.assertEqual(row['job_progress']['total_steps'], 10)
        self.assertEqual(row['job_progress']['observed_at'], 1234.5)
        self.assertNotEqual(result.data['observed_at'], 1234.5)
        self.assertFalse(self.actor.inspect('train').terminal)

    def test_candidates_and_experience_are_available_after_training(self):
        ref = self.runner.assembly.artifacts.put(b'data', kind='training_data')
        directive = Directive('train', 'task', 'train', mode='sft', payload=SFTSpec('mock'), input_refs=(ref,))
        self.actor.accept(directive)
        for _ in range(5):
            progress = self.actor.advance('train')
            if progress.terminal:
                break
        self.assertTrue(progress.terminal)
        candidates = self.operation('candidates')
        self.assertEqual(len(candidates.data['data']['items']), 1)
        self.assertTrue(candidates.artifacts)
        self.assertTrue(candidates.simulated)
        self.assertTrue(self.operation('experience', {'filters': {'kind': 'training_run'}}).data['data']['available'])


class ArtifactReaderTests(unittest.TestCase):
    def query(self, store, ref, **kwargs):
        return ArtifactReader(store).query(QueryRequest('artifact_read',
            filters={key: value for key, value in asdict(ref).items() if value is not None}, limit=1, **kwargs))

    def test_registered_text_pages_binary_and_forged_references(self):
        store = InMemoryArtifactStore()
        ref = store.put(('A' * 1000 + 'B' * 5).encode(), kind='text')
        first = self.query(store, ref)
        self.assertEqual(first.data['items'][0]['text'], 'A' * 1000)
        self.assertEqual(first.next_cursor, '1')
        second = self.query(store, ref, cursor=first.next_cursor)
        self.assertEqual(second.data['items'][0]['text'], 'B' * 5)
        self.assertEqual(first.version, second.version)
        binary = store.put(b'\x00\xff', kind='model')
        result = self.query(store, binary)
        self.assertEqual(result.data['items'], [])
        self.assertTrue(result.data['binary'])
        self.assertEqual(result.refs, (binary,))
        with self.assertRaises(ValueError):
            self.query(store, replace(ref, uri='https://example.invalid/private'))

    def test_file_artifacts_refuse_large_input_without_get_or_full_memory_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileArtifactStore(Path(directory))
            text = store.put(b'local text', kind='text')
            large = store.put(b'x' * (ArtifactReader.max_input_bytes + 1), kind='model')
            with patch.object(store, 'get', side_effect=AssertionError('No unbounded get')):
                self.assertEqual(self.query(store, text).data['items'][0]['text'], 'local text')
                result = self.query(store, large)
            self.assertFalse(result.data['available'])
            self.assertEqual(result.refs, (large,))
            self.assertIn('1 MiB', result.data['unavailable_reason'])


if __name__ == '__main__':
    unittest.main()
