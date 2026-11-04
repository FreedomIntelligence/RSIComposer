"""Long finish-only tails must not make a fixed ledger consume the context budget."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.decision_making.harness.context import (
    BasicContextCompactor, ContextBudgetError, encoded, working_decision_graph,
)
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.contracts.model_service import ModelRef


def ledger(n=206):
    return tuple({'decision_id': f'decision-{i:04}', 'status': 'completed'} for i in range(n))


class LedgerWindowTests(unittest.TestCase):
    def test_finish_growth_compacts_and_archive_recovers_original_ids(self):
        tools = ResearchTools(ResearchAgenda(), DecisionMemory())
        graph = ledger(1000)
        tools.decision_graph = lambda: graph
        history = tuple(HistoryEntry(str(i), 'decision', None, 'finish', False,
                        details={'reason': 'unchanged negative findings ' * 60}) for i in range(94))
        raw = DecisionContext('Investigate', (), tuple(d.spec for d in tools.definitions), (), history=history,
                              decision_graph=graph, candidate_origins={'candidate': graph[0]['decision_id']})
        before = deepcopy(raw)
        compactor = BasicContextCompactor(trigger_tokens=80000, target_tokens=24000, summary_tokens=4096)
        view = compactor.compact(raw, measure=lambda c: len(encoded(asdict(c))),
                                summarize=lambda *args: 'Source-qualified fixture summary, not real inference.')
        self.assertLessEqual(len(encoded(asdict(view))), 24000)
        self.assertEqual(view.history[-2:], raw.history[-2:])
        self.assertEqual(len(view.decision_graph), 33)
        self.assertEqual(raw, before)
        self.assertEqual(view.research_agenda['decision_graph_archive']['total'], 1000)
        receipt = tools._archive({'collection': 'decision_graph', 'record_id': 'decision-0005'})
        recovered = json.loads(''.join(chunk['text'] for chunk in receipt.data['items']))
        self.assertEqual(recovered, graph[5])
        # A source retrieved from an archive still validates against the full ledger.
        self.assertEqual(ModelDecisionPolicy._parse_lineage(
            {'source_decision_ids': ['decision-0005']}, raw)['source_decision_ids'], ('decision-0005',))

    def test_retains_active_producers_attempts_and_transitive_relations(self):
        graph = list(ledger())
        graph[1].update(status='pending')
        graph[4].update(source_decision_ids=['decision-0003'])
        graph[3].update(depends_on_decision_ids=['decision-0002'])
        graph[7].update(research_attempt_id='idea-1', research_attempt_title='Investigate data')
        graph[1].update(source_decision_ids=['decision-0004'])
        view = working_decision_graph(graph, {'candidate': 'decision-0000'})
        ids = {row['decision_id'] for row in view}
        self.assertTrue({'decision-0000', 'decision-0001', 'decision-0002',
                         'decision-0003', 'decision-0004', 'decision-0007'} <= ids)
        self.assertNotIn('decision-0006', ids)
        view[0]['status'] = 'changed'
        self.assertEqual(graph[0]['status'], 'completed')

    def test_no_archive_capability_keeps_graph_and_fails_explicitly(self):
        raw = DecisionContext('Research', (), (), (), decision_graph=ledger(2000))
        with self.assertRaises(ContextBudgetError):
            BasicContextCompactor().compact(raw)

    def test_oversized_active_graph_still_fails_instead_of_dropping_live_state(self):
        tools = ResearchTools(ResearchAgenda(), DecisionMemory())
        graph = tuple({**row, 'status': 'pending'} for row in ledger(2000))
        raw = DecisionContext('Research', (), tuple(d.spec for d in tools.definitions), (), decision_graph=graph)
        with self.assertRaises(ContextBudgetError):
            BasicContextCompactor().compact(raw)


class ProjectPrefixTests(unittest.TestCase):
    def test_project_is_stable_owned_configuration_not_learned_system_instructions(self):
        from rsisys.loops.decision_making.harness.learning_policy import MemoryPrefixDecisionPolicy
        from rsisys.loops._memory import Memory
        project = {'version': 'v1', 'resources': {'physical_devices': [13], 'available_memory': None}}
        backend = Mock(spec=['generate'])
        policy = MemoryPrefixDecisionPolicy(backend, ModelRef('fixture', 'decision', 'v1'), project_context=project)
        project['resources']['physical_devices'].append(0)
        copy = policy.project_context
        copy['version'] = 'mutated'
        context = DecisionContext('Improve target and decision policy', (), (), (),
            memories=(Memory('m', 'episodic', 'UNTRUSTED_MEMORY', (), status='unverified'),))
        first = policy._decision_request(context)
        second = policy._decision_request(replace(context, run_control={'seconds_remaining': 5}))
        self.assertEqual(first.prompt.split(',"run_control":')[0], second.prompt.split(',"run_control":')[0])
        payload = json.loads(first.prompt)
        self.assertEqual(list(payload)[:2], ['objective', 'project_context'])
        self.assertEqual(payload['project_context']['resources']['physical_devices'], [13])
        self.assertEqual(payload['project_context']['version'], 'v1')
        self.assertNotIn('UNTRUSTED_MEMORY', first.system)
        self.assertEqual(json.loads(policy._review_request(context).prompt)['project_context'], policy.project_context)
        self.assertIn('proposal-only', first.system)

    def test_rejects_non_json_or_unbounded_project(self):
        backend = Mock(spec=['generate'])
        for project in ({'nan': float('nan')}, {'huge': 'x'*33000}):
            with self.assertRaises(ValueError):
                ModelDecisionPolicy(backend, ModelRef('fixture', 'decision', 'v1'), project_context=project)
