"""Only prompt residency changes; raw facts and validation use the full ledger."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock

from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.harness.context import IdentityContextCompactor, with_working_decision_graph
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.ledger_context import working_ledger_context
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.state import CandidateEvaluation, DecisionContext, DirectiveProvenance
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.runtime.json_tools import encoded
from tests.test_context_ledger_window import ledger


MODEL = ModelRef('fixture', 'model', 'one')


def context(size):
    tools = ResearchTools(ResearchAgenda(), DecisionMemory())
    graph = tuple({**r, 'research_attempt_id': 'attempt-' + str(i),
                   'research_attempt_title': 'Separate idea',
                   'source_decision_ids': [] if i == 0 else [f'decision-{i-1:04}']}
                  for i, r in enumerate(ledger(size)))
    return DecisionContext('Explore', (), tuple(d.spec for d in tools.definitions), (),
        candidates=tuple(Candidate('c-' + str(i), 'base', {}, ('actor',)) for i in range(size)),
        candidate_evaluations=tuple(CandidateEvaluation('c-' + str(i), 'd-' + str(i), 'protocol', (),
            ExecutionStatus.COMPLETED, {'accuracy': 0.0 if i == 1 else 0.5}, False,
            ('receipt-' + str(i),)) for i in range(size)),
        candidate_origins={'c-' + str(i): r['decision_id'] for i, r in enumerate(graph)},
        directive_provenance=tuple(DirectiveProvenance('d-' + str(i), MODEL, MODEL) for i in range(size)),
        decision_graph=graph)


class LedgerContextTests(unittest.TestCase):
    def test_completed_history_and_attempt_growth_stays_bounded_and_exact(self):
        lengths = []
        for size in (100, 1000):
            raw = context(size)
            saved = deepcopy(raw)
            view = with_working_decision_graph(working_ledger_context(raw))
            lengths.append(len(encoded(asdict(view))))
            self.assertLessEqual(len(view.candidates), 10)
            self.assertLessEqual(len(view.directive_provenance), 10)
            self.assertLessEqual(len(view.decision_graph), 42)
            self.assertIn(raw.candidate_evaluations[1], view.candidate_evaluations)
            self.assertEqual(raw, saved)
            self.assertEqual(view.research_agenda['ledger_archives']['candidates']['total'], size)
            self.assertEqual(working_ledger_context(view), view)
        self.assertLess(lengths[1], lengths[0] + 3000)

    def test_live_dependencies_adoption_and_candidate_provenance_survive(self):
        raw = context(100)
        graph = list(raw.decision_graph)
        graph[-1].update(status='pending', source_decision_ids=['decision-0000'])
        graph[2].update(adopted_candidate_ids=['c-2'])
        raw = replace(raw, decision_graph=tuple(graph),
            outstanding=(Directive('d-3', 'Explore', 'Evaluate', payload={'candidate_id': 'c-3'}),))
        view = with_working_decision_graph(working_ledger_context(raw))
        self.assertTrue({'c-0', 'c-1', 'c-2', 'c-3'} <= {c.candidate_id for c in view.candidates})
        self.assertIn(raw.directive_provenance[3], view.directive_provenance)
        self.assertIn(raw.candidate_evaluations[2], view.candidate_evaluations)
        self.assertIn('decision-0000', {r['decision_id'] for r in view.decision_graph})

    def test_without_archive_capabilities_all_state_stays_verbatim(self):
        raw = replace(context(100), operations=())
        self.assertEqual(working_ledger_context(raw), raw)
        self.assertEqual(with_working_decision_graph(raw), raw)

    def test_archive_provider_follows_restored_state_and_pages_exact_records(self):
        provider = ResearchTools(ResearchAgenda(), DecisionMemory())
        tools = Mock(spec=['execute', 'research_tools'])
        tools.research_tools = provider
        harness = DefaultDecisionHarness(Mock(), tools, Mock())
        raw = context(100)
        # Replacing state must not leave archive callbacks pointing at old maps.
        state = deepcopy(harness.state)
        for name, key in (('candidates', 'candidate_id'), ('candidate_evaluations', 'candidate_id'),
                          ('directive_provenance', 'directive_id')):
            setattr(state, name, {getattr(row, key): row for row in getattr(raw, name)})
        state.candidate_origins = raw.candidate_origins
        harness.state = state
        for collection, record_id, expected in (
            ('candidates', 'c-0', asdict(raw.candidates[0])),
            ('candidate_evaluations', 'c-0', asdict(raw.candidate_evaluations[0])),
            ('directive_provenance', 'd-0', asdict(raw.directive_provenance[0])),
            ('candidate_origins', 'c-0', {'candidate_id': 'c-0', 'decision_id': 'decision-0000'})):
            page = provider._archive({'collection': collection, 'record_id': record_id})
            recovered = json.loads(''.join(c['text'] for c in page.data['items']))
            self.assertEqual(recovered, json.loads(encoded(expected)))
            self.assertTrue(page.data['found'])
        page = provider._archive({'collection': 'candidate_evaluations', 'record_id': 'c-0',
                                  'evidence_path': '/quality_metrics'})
        self.assertEqual(json.loads(page.data['items'][0]['text']), {'accuracy': 0.5})

    def test_native_compactor_gets_working_view_but_old_ids_validate_against_full_context(self):
        raw = replace(context(100), directive_modes=('evaluate',))
        backend = Mock(spec=['generate'])
        backend.generate.return_value = GenerationResponse(json.dumps({'kind': 'explore',
            'mode': 'evaluate', 'strategy': 'Recheck an archived candidate',
            'arguments': {'candidate_id': 'c-0', 'protocol_ref': 'protocol'}}), MODEL, True)
        policy = ModelDecisionPolicy(backend, MODEL)
        policy.context_compactor = IdentityContextCompactor()
        decision = policy.decide(raw)
        payload = json.loads(backend.generate.call_args.args[0].prompt)
        self.assertNotIn('c-0', [c['candidate_id'] for c in payload['candidates']])
        self.assertEqual(ModelDecisionPolicy._parse_lineage(
            {'source_decision_ids': ['decision-0000']}, raw)['source_decision_ids'], ('decision-0000',))
        self.assertEqual(len(raw.candidates), 100)
        self.assertEqual(decision.directive.payload.candidate, raw.candidates[0])
