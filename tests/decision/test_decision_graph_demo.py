"""End-to-end fork/join provenance, using real overlapping CPU artifact workers."""
from tests import ROOT as REPO_ROOT
from dataclasses import asdict
import json
import os
from experiments.qwen_toy.monitor import enrich_completed_run
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID

from examples.components.decision_graph import (
    BASE, CANDIDATE_IDS, DATA, OBJECTIVE, PROMPT, VERSION,
    ConsumeCurrentSystem, build_demo, run_demo,
)
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.exploration.harness.commands import CommandState
from rsisys.runtime.files import json_bytes
from rsimonitor import project_snapshot


class DecisionGraphDemoTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'decision-graph'
        run_demo(self.root)
        self.report = json.loads((self.root / 'result.json').read_text())
        self.monitor = json.loads((self.root / 'monitor.json').read_text())
        self.history = json.loads((self.root / 'decision_memory.json').read_text())['history']
        self.steps = [json.loads(line) for line in (self.root / 'steps.jsonl').read_text().splitlines()]
        self.graph = {row['decision_id']: row for row in self.report['decision_graph']}
        self.producers = {row['directive_id']: row for row in self.graph.values()
                          if (row.get('directive_id') or '').startswith('produce-')}
        self.adoption = next(row for row in self.graph.values() if row.get('operation') == 'system_adopt')

    def test_attempt_membership_flows_from_harness_through_live_and_offline_monitor(self):
        attempts = {row['research_attempt_id'] for row in self.graph.values()}
        self.assertEqual(len(attempts), 3)
        producers = list(self.producers.values())
        self.assertNotEqual(producers[0]['research_attempt_id'], producers[1]['research_attempt_id'])
        join_attempt = self.adoption['research_attempt_id']
        other = [r for r in self.graph.values() if r not in producers]
        self.assertTrue(all(r['research_attempt_id'] == join_attempt for r in other))
        for snapshot in (self.monitor, enrich_completed_run(self.root)):
            for row in snapshot['decisions']:
                actual = self.graph[row['decision_id']]
                self.assertEqual(row['research_attempt_id'], actual['research_attempt_id'])
                self.assertEqual(row['research_attempt_title'], actual['research_attempt_title'])
        self.assertEqual(set(self.adoption['adopted_decision_ids']), {r['decision_id'] for r in producers})

    def test_workers_really_overlap_and_produce_independent_same_parent_artifacts(self):
        self.assertEqual(self.report['max_in_flight'], 2)
        intervals = self.report['worker_intervals']
        self.assertEqual(set(intervals), {'prompt', 'data'})
        self.assertLess(max(row['started_ns'] for row in intervals.values()),
                        min(row['finished_ns'] for row in intervals.values()))
        self.assertEqual({row['parent_system'] for row in intervals.values()}, {'initial'})
        self.assertEqual(self.report['completion_order'], ['produce-data', 'produce-prompt', 'consume-current'])
        candidates = [candidate for row in self.history if row['kind'] == 'feedback'
                      for candidate in row['details']['feedback']['candidates']]
        self.assertEqual({candidate['candidate_id'] for candidate in candidates}, set(CANDIDATE_IDS))
        self.assertEqual({candidate['parent_system'] for candidate in candidates}, {'initial'})
        self.assertEqual({candidate['base_version'] for candidate in candidates}, {BASE.revision})
        self.assertEqual({tuple(candidate['changes']) for candidate in candidates}, {('prompt',), ('data',)})
        self.assertTrue(all(not row['depends_on_decision_ids'] for row in self.producers.values()))
        self.assertTrue(all(not row['source_decision_ids'] for row in self.producers.values()))

    def test_join_adopts_both_origins_once_only_after_both_feedbacks(self):
        self.assertEqual(self.report['activations'], 1)
        calls = [row for row in self.history if row['kind'] == 'operation'
                 and row['details']['operation']['name'] == 'system_adopt']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['details']['result']['adopted_candidate_ids'], list(CANDIDATE_IDS))
        origins = tuple(self.report['candidate_origins'][ident] for ident in CANDIDATE_IDS)
        for component in ('prompt', 'data'):
            self.assertEqual(self.report['candidate_origins'][f'{component}-candidate'],
                             self.producers[f'produce-{component}']['decision_id'])
        self.assertEqual(tuple(self.adoption['depends_on_decision_ids']), origins)
        self.assertEqual(tuple(self.adoption['source_decision_ids']), origins)
        self.assertEqual(tuple(self.adoption['adopted_decision_ids']), origins)
        self.assertEqual(tuple(self.adoption['adopted_candidate_ids']), CANDIDATE_IDS)
        self.assertEqual(self.adoption['status'], 'completed')
        join_tick = next(i for i, step in enumerate(self.steps)
                         if (step['event'].get('decision', {}).get('operation_result') or {}).get('adopted_candidate_ids'))
        feedback_ticks = {step['event']['feedback']['directive_id']: i for i, step in enumerate(self.steps)
                          if step['event'].get('feedback')}
        self.assertLess(feedback_ticks['produce-prompt'], join_tick)
        self.assertLess(feedback_ticks['produce-data'], join_tick)
        confirmed = {(edge['from'], edge['to']) for edge in self.adoption['confirmed_edges']
                     if edge['kind'] == 'adopted' and edge['confirmed']}
        self.assertEqual(confirmed, {(ident, self.adoption['decision_id']) for ident in origins})

    def test_immediate_next_decision_executes_both_current_registry_components(self):
        rows = list(self.graph.values())
        join_index = rows.index(self.adoption)
        execution = rows[join_index + 1]
        self.assertEqual(execution['directive_id'], 'consume-current')
        self.assertEqual(execution['depends_on_decision_ids'], [self.adoption['decision_id']])
        self.assertEqual(execution['status'], 'completed')
        directive = execution['proposal']['directive']
        self.assertIsNone(directive['payload'])
        self.assertEqual(directive['input_refs'], [])
        consumed, = self.report['consumed']
        self.assertEqual(consumed['version'], VERSION)
        self.assertEqual(consumed['components'], self.report['current']['components'])
        self.assertEqual(consumed['prompt'], PROMPT)
        self.assertEqual(consumed['data'], DATA)
        self.assertEqual(consumed['output'], 'SIMULATED adopted prompt: total=16; count=3')
        for component, ref in consumed['components'].items():
            actual = (self.root / 'artifacts' / ref['artifact_id'] / 'data').read_bytes()
            expected = PROMPT.encode() if component == 'prompt' else json_bytes(DATA)
            self.assertEqual(actual, expected)

    def test_monitor_records_join_via_generic_monitor_path_and_survives_web_projection(self):
        # Read the actual file, not a demo graph projection or a patched Monitor.
        projected = project_snapshot(self.monitor, self.root.name)
        for rows in (self.monitor['decisions'], projected['decisions']):
            with self.subTest(projected=rows is projected['decisions']):
                joins = [row for row in rows if row.get('operation') == 'system_adopt']
                self.assertEqual(len(joins), 1)
                join = joins[0]
                self.assertEqual(join['decision_id'], self.adoption['decision_id'])
                self.assertEqual(join['status'], 'completed')
                self.assertEqual(set(join['adopted_candidate_ids']), set(CANDIDATE_IDS))
                self.assertEqual(set(join['adopted_decision_ids']), set(self.report['candidate_origins'].values()))
                self.assertEqual(join['adopted_decision_ids'], self.adoption['adopted_decision_ids'])
                self.assertIn('Adopted 2 candidates', join['feedback_summary'])
                self.assertTrue(join['reason'])
                self.assertEqual(join['changes'], [])  # No demo-generated change summaries.
        self.assertEqual(self.monitor['status'], 'completed')
        self.assertTrue(self.monitor['simulated'])
        self.assertIn('Simulated', projected['experiment_name'])
        self.assertEqual(self.monitor['workflow']['completed_directives'], 3)

    def test_entire_chain_has_only_explicit_edges_and_no_chronological_parent(self):
        monitor_rows = {row['decision_id']: row for row in self.monitor['decisions']}
        self.assertEqual(set(monitor_rows), set(self.graph))
        self.assertEqual(len(monitor_rows), len(self.monitor['decisions']))
        waits = [row for row in self.graph.values() if row['kind'] == 'wait']
        self.assertEqual(len(waits), 1)
        self.assertEqual(set(waits[0]['source_decision_ids']), set(self.report['candidate_origins'].values()))
        self.assertFalse(waits[0]['depends_on_decision_ids'])
        self.assertFalse(waits[0]['adopted_decision_ids'])
        finished = next(row for row in self.graph.values() if row['kind'] == 'finish')
        consumer = next(row for row in self.graph.values() if row.get('directive_id') == 'consume-current')
        expected_dependencies = {
            **{ident: [] for ident in self.graph},
            self.adoption['decision_id']: self.adoption['source_decision_ids'],
            consumer['decision_id']: [self.adoption['decision_id']],
            finished['decision_id']: [consumer['decision_id']],
        }
        for ident, record in self.graph.items():
            with self.subTest(decision_id=ident):
                self.assertEqual(str(UUID(ident)), ident)
                self.assertEqual(record['depends_on_decision_ids'], expected_dependencies[ident])
                self.assertEqual(monitor_rows[ident]['depends_on_decision_ids'], expected_dependencies[ident])
                self.assertEqual(monitor_rows[ident]['source_decision_ids'], record['source_decision_ids'])
                self.assertNotIn('parent_decision_id', monitor_rows[ident])
                if ident != self.adoption['decision_id']:
                    self.assertFalse(record['adopted_decision_ids'])
                    self.assertFalse(monitor_rows[ident].get('adopted_decision_ids'))
                for edge in record['confirmed_edges']:
                    allowed = {'depends_on': record['depends_on_decision_ids'],
                               'source': record['source_decision_ids'], 'adopted': record['adopted_decision_ids']}
                    self.assertTrue(edge['confirmed'])
                    self.assertEqual(edge['to'], ident)
                    self.assertIn(edge['from'], allowed[edge['kind']])

    def test_consumer_resolves_registry_at_execution_not_at_prepare(self):
        demo = build_demo(self.root.parent / 'live-read')
        self.addCleanup(demo.close)
        handler = ConsumeCurrentSystem(demo.registry, demo.artifacts, demo.consumed)
        state = CommandState(Directive('live-read', OBJECTIVE, 'Read current', mode='consume_current', simulated=True))
        handler.prepare(None, state)
        prompt = demo.artifacts.put(b'Changed after prepare: {total}/{count}', kind='prompt')
        data = demo.artifacts.put(json_bytes([20, 22]), kind='data')
        for component, ref in (('prompt', prompt), ('data', data)):
            demo.registry.register(Candidate(component, BASE.revision, {component: ref}, ('actor',),
                                             parent_system=demo.registry.current.version))
        demo.registry.adopt(('prompt', 'data'), version='later-current', idempotency_key='later')
        feedback = handler.advance(None, state, None)
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        consumed, = demo.consumed
        self.assertEqual(consumed['version'], 'later-current')
        self.assertEqual(consumed['output'], 'Changed after prepare: 42/2')
        self.assertEqual(consumed['components'], asdict(demo.registry.current)['components'])

    def test_documented_cli_exports_a_run_and_refuses_to_overwrite_it(self):
        repo = REPO_ROOT
        root = self.root.parent / 'cli-run'
        env = {**os.environ, 'PYTHONPATH': str(repo / 'src')}
        command = [sys.executable, 'examples/components/decision_graph.py', '--run-dir', str(root)]
        process = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        self.assertTrue(report['simulated'])
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['activations'], 1)
        before = (root / 'monitor.json').read_bytes()
        repeated = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn('Choose a fresh run directory', repeated.stderr)
        self.assertEqual((root / 'monitor.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
