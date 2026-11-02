"""Bounded factual evidence with real local archive round trips, no services."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import OperationResult, OperationSpec
from rsisys.loops.decision_making.harness.context import (
    BasicContextCompactor, ContextBudgetError, IdentityContextCompactor, encoded, result_id,
)
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.research_state import evidence_working_view, research_working_view
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.harness.tools.research import ResearchTools


MODEL = {'backend_id': 'fixture', 'model_id': 'evaluated-model', 'revision': 'frozen-v1'}
METRICS = {'accuracy': 0.0, 'delta': -0.125, 'invalid_format': 1.0}


def measured(index):
    return HistoryEntry(f'evidence-{index}', 'feedback', f'directive-{index}',
        'Measured result; qualification prose. ' * 100, False, (f'source-{index}',), details={
            'decision_id': f'decision-{index}',
            'directive': {'directive_id': f'directive-{index}', 'mode': 'evaluate',
                'objective': 'Repeated objective. ' * 700,
                'strategy': 'Unverified hypothesis. ' * 200,
                'payload': {'candidate': {'candidate_id': f'candidate-{index}',
                                        'base_version': 'frozen-base'}, 'protocol_ref': 'frozen-protocol'}},
            'feedback': {'directive_id': f'directive-{index}', 'status': 'completed',
                'quality_metrics': deepcopy(METRICS), 'simulated': False,
                'evidence_refs': [f'source-{index}'], 'summary': 'not a success interpretation',
                'raw_duplicate': 'Raw duplicate. ' * 600},
            'provenance': {'execution_model': deepcopy(MODEL)}})


def fixture(memory=None, experience=None, count=8):
    memory = memory if memory is not None else DecisionMemory()
    experience = experience if experience is not None else BasicExperienceSystem(InMemoryExperienceStore())
    tools = ResearchTools(ResearchAgenda(), memory, experience=experience)
    old_tool = OperationResult('Old measured tool', (), False, data={
        'quality_metrics': deepcopy(METRICS), 'protocol_ref': 'tool-protocol',
        'candidate_id': 'tool-candidate', 'model': deepcopy(MODEL),
        'evidence_refs': ['tool-source'], 'content': 'duplicate raw tool data ' * 1000})
    latest_tool = OperationResult('Latest full tool', (), False, data={'content': 'latest raw payload ' * 40})
    latest_feedback = Feedback('latest', ExecutionStatus.COMPLETED, 'Full latest qualification ' * 20,
                               False, quality_metrics={'accuracy': 0.25}, evidence_refs=('latest-source',))
    old_history = tuple(measured(i) for i in range(count))
    old_operation = HistoryEntry('old-tool', 'operation', None, old_tool.summary, False,
                                details={'result': asdict(old_tool)})
    latest = (
        HistoryEntry('latest-feedback', 'feedback', 'latest', latest_feedback.summary, False,
                     details={'feedback': asdict(latest_feedback), 'directive': {'strategy': 'latest full strategy'}}),
        HistoryEntry('latest-tool', 'operation', None, latest_tool.summary, False,
                     details={'result': asdict(latest_tool)}),
    )
    for row in (*old_history, old_operation, *latest):
        memory.append(row)
    observation = ExperienceRecord('experience-negative', 'evaluation', 'actor', None,
        'Full evaluation qualification ' * 200, False, ('experience-source',), details={
            'request': {'protocol_ref': 'experience-protocol', 'model': deepcopy(MODEL)},
            'outcome': {'status': 'succeeded', 'result': {'metrics': deepcopy(METRICS),
                        'evidence_refs': ['experience-source']}, 'raw': 'raw evaluation payload ' * 1000}})
    experience.record(observation)
    context = DecisionContext('Fixture objective', experience.retrieve(limit=None),
        tuple(d.spec for d in tools.definitions), (old_tool, latest_tool),
        history=memory.history(), last_feedback=latest_feedback)
    return research_working_view(context, [], []), tools, memory, experience


def measure(context):
    # Conservative complete-request fixture meter, not a provider tokenizer.
    return 800 + len(encoded(asdict(context)).encode('utf-8'))


def read_original(tools, card):
    arguments = dict(card['read_path'])
    arguments.pop('operation')
    chunks, cursor = [], None
    while True:
        receipt = tools._archive({**arguments, 'limit': 32, 'cursor': cursor})
        chunks.extend(item['text'] for item in receipt.data['items'])
        cursor = receipt.data['next_cursor']
        if cursor is None:
            return json.loads(''.join(chunks))


class EvidenceProjectionTests(unittest.TestCase):
    def test_large_measured_feedback_is_bounded_before_compact_with_exact_facts_and_unchanged_archives(self):
        with TemporaryDirectory() as directory:
            memory = DecisionMemory(Path(directory) / 'memory.json')
            raw, tools, memory, experience = fixture(memory)
            saved, archive_bytes, experience_rows = deepcopy(raw), memory.path.read_bytes(), experience.retrieve(limit=None)
            compactor = BasicContextCompactor()
            projected = compactor.compact(raw, measure=measure, summarize=lambda *args: 'Archive navigation only')
            self.assertGreater(measure(raw), 80000)
            self.assertLess(measure(projected), 40000)
            self.assertEqual(compactor.compactions, 0)
            self.assertEqual(projected.history[-2:], raw.history[-2:])
            self.assertEqual(projected.last_feedback, raw.last_feedback)
            self.assertEqual(projected.operation_results[-1], raw.operation_results[-1])
            for old, visible in zip(raw.history[:8], projected.history[:8]):
                self.assertEqual(visible.record_id, old.record_id)
                facts = visible.details['facts']
                self.assertEqual(facts['/details/feedback/quality_metrics'], METRICS)
                self.assertEqual(facts['/details/feedback/status'], 'completed')
                self.assertEqual(facts['/details/feedback/simulated'], False)
                self.assertEqual(facts['/details/directive/payload/protocol_ref'], 'frozen-protocol')
                self.assertEqual(facts['/details/directive/payload/candidate/candidate_id'], f'candidate-{old.record_id.split("-")[-1]}')
                self.assertEqual(facts['/details/provenance/execution_model'], MODEL)
                self.assertEqual(facts['/details/feedback/evidence_refs'], list(old.source_refs))
                self.assertNotIn('Repeated objective.', encoded(asdict(visible)))
                card = visible.details['evidence_card']
                self.assertEqual(card['sha256'], result_id(asdict(old)))
                self.assertEqual(read_original(tools, card), json.loads(encoded(asdict(old))))
            operation = projected.operation_results[0]
            self.assertEqual(operation.data['facts']['/data/quality_metrics'], METRICS)
            recovered = read_original(tools, operation.data['evidence_card'])
            self.assertEqual(recovered, json.loads(encoded(asdict(raw.operation_results[0]))))
            observation = projected.experience[0]
            self.assertEqual(observation.details['facts']['/details/outcome/result/metrics'], METRICS)
            self.assertEqual(read_original(tools, observation.details['evidence_card']),
                             json.loads(encoded(asdict(raw.experience[0]))))
            self.assertEqual(raw, saved)
            self.assertEqual(memory.path.read_bytes(), archive_bytes)
            self.assertEqual(experience.retrieve(limit=None), experience_rows)

    def test_no_archive_or_wrong_collection_or_mutable_archive_cannot_drop_original_evidence(self):
        raw, _, _, _ = fixture(count=1)
        archive = next(spec for spec in raw.operations if spec.name == 'research_archive')
        wrong_schema = deepcopy(archive.input_schema)
        wrong_schema['properties']['collection']['enum'] = ['memories', 'decision_graph']
        for specs in ((), (replace(archive, read_only=False),), (replace(archive, input_schema=wrong_schema),)):
            with self.subTest(specs=specs):
                unprivileged = replace(raw, operations=specs)
                saved = deepcopy(unprivileged)
                self.assertEqual(evidence_working_view(unprivileged), unprivileged)
                compactor = BasicContextCompactor(trigger_tokens=40000, target_tokens=16000, summary_tokens=200)
                state = compactor.export_state()
                writer = Mock(return_value='Cannot authorize dropping evidence')
                with self.assertRaisesRegex(ContextBudgetError, 'Irreducible factual evidence'):
                    compactor.compact(unprivileged, measure=measure, summarize=writer)
                writer.assert_not_called()
                self.assertEqual(compactor.export_state(), state)
                self.assertEqual(unprivileged, saved)

    def test_projection_state_roundtrip_does_not_reattach_old_directives(self):
        raw, _, _, _ = fixture()
        compactor = BasicContextCompactor()
        first = compactor.compact(raw, measure=measure)
        restored = BasicContextCompactor()
        restored.restore_state(json.loads(json.dumps(compactor.export_state())))
        self.assertEqual(restored.compact(raw, measure=measure), first)
        self.assertEqual(restored.compact(first, measure=measure), first)
        fresh = HistoryEntry('fresh', 'decision', None, 'New observation', False)
        next_view = restored.compact(replace(raw, history=(*raw.history, fresh)), measure=measure)
        self.assertEqual(next_view.history[-2:], (raw.history[-1], fresh))
        self.assertEqual(next_view.history[0], first.history[0])
        self.assertEqual(restored.compactions, 0)
        legacy = compactor.export_state()
        legacy.pop('project_evidence')
        restored.restore_state(legacy)  # Previous snapshots remain valid.

    def test_small_measured_receipts_remain_verbatim_until_pressure(self):
        raw, _, _, _ = fixture(count=1)
        raw = replace(raw, history=raw.history[-2:], experience=(), operation_results=raw.operation_results[-1:])
        self.assertEqual(BasicContextCompactor().compact(raw, measure=measure), raw)

    def test_dsh_identity_compactor_and_feedback_projection_leave_large_raw_receipts_complete(self):
        raw, _, _, _ = fixture()
        self.assertGreater(measure(raw), 80000)
        projected = research_working_view(raw, [], [])
        self.assertEqual(projected.history, raw.history)
        self.assertEqual(projected.experience, raw.experience)
        self.assertEqual(projected.operation_results, raw.operation_results)
        self.assertEqual(IdentityContextCompactor().compact(projected), projected)

    def test_transitive_active_dependencies_keep_facts_but_can_project_large_payloads(self):
        raw, _, _, _ = fixture(count=3)
        graph = ({'decision_id': 'active', 'status': 'waiting', 'depends_on_decision_ids': ['decision-1']},
                 {'decision_id': 'decision-1', 'status': 'completed', 'source_decision_ids': ['decision-0']},
                 {'decision_id': 'decision-0', 'status': 'completed', 'adopted_decision_ids': ['decision-2']},
                 {'decision_id': 'decision-2', 'status': 'completed'})
        raw = replace(raw, decision_graph=graph)
        self.assertEqual(evidence_working_view(raw).history[:3], raw.history[:3])
        compactor = BasicContextCompactor(trigger_tokens=40000, target_tokens=24000, summary_tokens=200)
        projected = compactor.compact(raw, measure=measure)
        for row in projected.history[:3]:
            self.assertIn('evidence_card', row.details)
            self.assertEqual(row.details['facts']['/details/feedback/quality_metrics'], METRICS)
        self.assertEqual(projected.decision_graph, raw.decision_graph)
        # The latest feedback's matching history remains full even outside the newest two entries.
        later = tuple(HistoryEntry(f'wait-{i}', 'decision', None, 'Wait', False) for i in range(3))
        raw = replace(raw, decision_graph=(), history=(*raw.history, *later))
        view = evidence_working_view(raw)
        self.assertEqual(next(h for h in view.history if h.record_id == 'latest-feedback'), raw.history[-5])

    def test_outstanding_directive_keeps_its_completed_dependencies_and_tool_receipts_full(self):
        raw, _, _, _ = fixture(count=2)
        operation = replace(raw.history[2], details={**raw.history[2].details, 'decision_id': 'decision-0'})
        raw = replace(raw, history=(*raw.history[:2], operation, *raw.history[3:]),
            outstanding=(Directive('directive-1', 'objective', 'Still owned by the executor'),),
            decision_graph=({'decision_id': 'decision-1', 'status': 'completed',
                             'depends_on_decision_ids': ['decision-0']},
                            {'decision_id': 'decision-0', 'status': 'completed'}))
        view = evidence_working_view(raw)
        self.assertEqual(view.history, raw.history)
        self.assertEqual(view.operation_results, raw.operation_results)
        working = research_working_view(raw, [], [])
        self.assertTrue({'evidence-0', 'evidence-1', 'old-tool'} <=
                        set(working.research_state['protected_history_ids']))

    def test_irreducible_aggregate_metric_maps_fail_atomically(self):
        raw, _, _, _ = fixture(count=1)
        huge_metrics = {f'metric-{i}': -0.25 for i in range(5000)}
        huge = replace(raw.history[0], details={'feedback': {'quality_metrics': huge_metrics}})
        context = replace(raw, history=(huge, *raw.history[1:]),
                          outstanding=(Directive('directive-0', 'objective', 'Live metric dependency'),))
        saved = deepcopy(context)
        compactor = BasicContextCompactor()
        state = compactor.export_state()
        with self.assertRaisesRegex(ContextBudgetError, 'Irreducible factual evidence'):
            compactor.compact(context, measure=measure)
        self.assertEqual(compactor.export_state(), state)
        self.assertEqual(context, saved)

    def test_card_never_infers_a_result_from_prose(self):
        raw, _, _, _ = fixture(count=1)
        old = replace(raw.history[0], summary='accuracy 0.99; probably improved! ' * 1000)
        # The archive in this fixture has the original receipt; no generated claim
        # enters structured facts, even when prose conflicts with those facts.
        view = evidence_working_view(replace(raw, history=(old, *raw.history[1:])))
        facts = view.history[0].details['facts']
        self.assertEqual(facts['/details/feedback/quality_metrics'], METRICS)
        self.assertNotIn('0.99', encoded(facts))
        self.assertNotIn('probably improved', encoded(facts))

    def test_local_frozen_run_receipts_project_without_sdk_or_archive_changes(self):
        root = Path(__file__).resolve().parents[1] / 'runs' / 'medical-researchcompact30-20261102T080724Z'
        requests = root / 'model_requests.jsonl'
        if not requests.exists():
            self.skipTest('Optional frozen run receipts are not checked into this checkout')
        archive = root / 'decision_memory.json'
        original_bytes = archive.read_bytes()
        # Inspect the last emitted request before the budget failure. This checks
        # real receipt structure, not provider token counts or a rerun of research.
        last = json.loads(requests.read_text().splitlines()[-1])
        payload = json.loads(last['request']['prompt'])
        feedback = payload['last_feedback']
        feedback = Feedback(**{**feedback, 'status': ExecutionStatus(feedback['status']),
            **{key: tuple(feedback[key]) for key in ('artifact_refs', 'evidence_refs', 'candidates',
                                                   'jobs', 'adopted_candidate_ids')}})
        raw = DecisionContext(payload['objective'],
            tuple(ExperienceRecord(**row) for row in payload['experience']),
            tuple(OperationSpec(**spec) for spec in payload['operations']), (),
            history=tuple(HistoryEntry(**{**row, 'source_refs': tuple(row['source_refs'])})
                          for row in payload['history']),
            last_feedback=feedback, decision_graph=tuple(payload['decision_graph']))
        view = evidence_working_view(raw)
        tools = ResearchTools(ResearchAgenda(), DecisionMemory(archive))
        cards = 0
        for source, visible in zip(raw.history, view.history):
            card = visible.details.get('evidence_card')
            if card is None:
                self.assertEqual(visible, source)
                continue
            cards += 1
            self.assertEqual(card['sha256'], result_id(asdict(source)))
            self.assertEqual(read_original(tools, card), json.loads(encoded(asdict(source))))
            feedback = source.details.get('feedback')
            if feedback:
                self.assertEqual(visible.details['facts']['/details/feedback/quality_metrics'],
                                 feedback['quality_metrics'])
                self.assertEqual(visible.details['facts']['/details/feedback/evidence_refs'],
                                 feedback['evidence_refs'])
        self.assertGreater(cards, 0)
        self.assertLess(len(encoded(asdict(view))), len(encoded(asdict(raw))))
        self.assertEqual(view.history[-2:], raw.history[-2:])
        self.assertEqual(view.last_feedback, raw.last_feedback)
        self.assertEqual(archive.read_bytes(), original_bytes)
