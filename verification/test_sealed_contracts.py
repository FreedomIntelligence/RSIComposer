"""Offline regressions for sealed final plans and Host-only aggregate evidence."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsicomposer.application import Application, ApplicationConfig
from rsicomposer.methods import ControllerDefinition
from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.controller_version import EvidenceScope
from rsicomposer.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.model_service import BackendBinding, ModelRef
from rsicomposer.loops.contracts.profiles import DEV, SEALED, Invariant
from rsicomposer.registry import Registry
from rsicomposer.research import MemoryAssembly, ResearchAssembly
from rsicomposer.runtime.evidence.records import ExperienceRecord
from rsicomposer.runtime.evidence.retrieval import controller_experience
from rsicomposer.runtime.orchestration.lab_group import LabGroup
from rsicomposer.runtime.orchestration.run_policy import RunDecision, TimeRunPolicy
from rsicomposer.runtime.storage.artifacts import InMemoryArtifactStore


OBJECTIVE = 'Check Host-only evaluation with synthetic fixture scores'
FINAL_METRIC = 'fixture_final_accuracy'
ROW = 'synthetic-per-item-row-not-an-aggregate'


class FinishPolicy:
    def __init__(self, windows):
        self.windows = windows

    def decide(self, window):
        self.windows.append(window)
        return Decision('finish')

    def state(self):
        return {}

    def restore(self, state):
        pass


class SealedLab:
    def __init__(self, effects, plans):
        self.effects, self.plans = effects, plans

    def describe(self):
        return (OperationSpec('evaluate', 'Score a synthetic fixture', {'type': 'object'}),)

    def split_roles(self):
        return {'dev': DEV, 'heldout': SEALED}

    def final_evaluations(self):
        self.plans.append('planned')
        return tuple(OperationCall('evaluate', {'split': 'heldout'}, call_id=ident)
                     for ident in ('host-final-a', 'host-final-b'))

    def execute(self, call):
        self.effects.append(call.call_id)
        return OperationResult(ROW, (), True, data={'metrics': {FINAL_METRIC: 0.375}, 'rows': [ROW]})

    def state(self):
        return {}

    def restore(self, state):
        pass


class RecordingMemory:
    def __init__(self, queries, events):
        self.queries, self.events = queries, events

    def context(self, query):
        self.queries.append(query)
        return ''

    def observe(self, event):
        self.events.append(event)


def modules(windows, effects, plans, *, memory=None):
    registry = Registry(builtins=False)
    registry.register_controller('fixture', ControllerDefinition(
        lambda context, options: FinishPolicy(windows), memory=memory), version='1')
    registry.register_lab('fixture', lambda context, options: SealedLab(effects, plans), version='1')
    return registry.build({'controller': {'name': 'fixture'}, 'labs': {'fixture': {'name': 'fixture'}}})


class SealedEvidenceTest(unittest.TestCase):
    def test_time_policy_restarts_without_planning_finals_and_filters_existing_host_rows(self):
        windows, effects, plans, queries, observations = [], [], [], [], []
        memory = lambda options: MemoryAssembly(ResearchAssembly(
            context_memory=RecordingMemory(queries, observations)))
        with TemporaryDirectory() as directory, Application(
            ApplicationConfig(OBJECTIVE, Path(directory), {'offline': BackendBinding('offline', 'mock')},
                              {'decision_maker': ModelRef('offline', 'fixture', 'v1')}),
            modules=modules(windows, effects, plans, memory=memory),
            run_policy=TimeRunPolicy(100, restart_delay_seconds=2),
        ) as app:
            clock = [0.]
            runner = app.runner
            runner._clock = lambda: clock[0]
            first = runner.step(OBJECTIVE)
            self.assertEqual((first.kind, first.event['run_control']), ('waiting', 'researcher_finished'))
            self.assertEqual((plans, effects), ([], []))

            # Simulate rows from a pre-existing evidence source, not new sealing
            # during a run that is about to resume research. Cover both legacy
            # identity detection and an explicit host-finish phase marker.
            app.experience.record(ExperienceRecord('public-fixture', 'evaluation', 'shared', None,
                'Visible validation score', True, details={'metrics': {'validation_accuracy': 0.5}}))
            for ident, extra in [('sealed-legacy-fixture', {}), ('host-final-import', {'phase': 'host_finish'})]:
                app.experience.record(ExperienceRecord(ident, 'evaluation', 'shared', None,
                    'Fixture host aggregate', True, details={**extra, 'metrics': {FINAL_METRIC: 0.375}}))
            recent = controller_experience(app.experience).retrieve(kinds=('evaluation',), limit=1)
            self.assertEqual([row.record_id for row in recent], ['public-fixture'])

            self.assertEqual(runner.step(OBJECTIVE).event['run_control'], 'restart_wait')
            clock[0] = 3.
            self.assertEqual(runner.step(OBJECTIVE).event['run_control'], 'research_restarted')
            runner.step(OBJECTIVE)
            self.assertEqual(len(windows), 2)
            self.assertEqual((plans, effects), ([], []))
            window = json.loads(windows[-1].context)
            self.assertIn('validation_accuracy', repr(window['experience']))
            self.assertNotIn(FINAL_METRIC, repr(window))
            self.assertGreaterEqual(len(queries), 2)
            app.assembly.controller_host.active.context.record('interpretation', {'note': 'visible control'})
            self.assertIn('visible control', repr(observations))
            self.assertNotIn(FINAL_METRIC, repr((queries, observations)))

            harness = runner.decision_making.harness
            harness.observe(EvidenceEvent('sealed-legacy-fixture', 'lab', output={FINAL_METRIC: 0.375}))
            self.assertNotIn(FINAL_METRIC, repr(observations))
            for collection in ('experience', 'history', 'operation_results'):
                archive = harness.execute_tool(OperationCall('research_archive', {'collection': collection}))
                self.assertNotIn(FINAL_METRIC, repr(archive.data))
                if collection == 'experience':
                    self.assertIn('validation_accuracy', repr(archive.data['items']))
            for ident in ('sealed-legacy-fixture', 'host-final-import'):
                archive = harness.execute_tool(OperationCall('research_archive', {
                    'collection': 'experience', 'record_id': ident, 'evidence_path': '/details/metrics'}))
                self.assertFalse(archive.data['found'])
                self.assertEqual(archive.data['items'], [])
                with self.assertRaisesRegex(ValueError, 'Unknown evidence event or checkpoint'):
                    app.assembly.evidence_view(EvidenceScope(since=ident)).events()
            self.assertNotIn(FINAL_METRIC, repr(app.assembly.evidence_view().records()))
            self.assertEqual([row.details['metrics'][FINAL_METRIC]
                for row in app.experience.retrieve(kinds=('evaluation',)) if FINAL_METRIC in row.details['metrics']],
                [0.375, 0.375])

    def test_parked_finish_never_plans_finals(self):
        class Park:
            def evaluate(self, context):
                return RunDecision()

        windows, effects, plans = [], [], []
        with TemporaryDirectory() as directory, Application(
            ApplicationConfig(OBJECTIVE, Path(directory), {}, {}),
            modules=modules(windows, effects, plans), run_policy=Park(),
        ) as app:
            for _ in range(2):
                self.assertEqual(app.runner.step(OBJECTIVE).event['run_control'], 'researcher_finished')
            self.assertEqual((len(windows), effects, plans), (1, [], []))

    def test_terminal_host_report_and_receipts_survive_durable_resume(self):
        windows, effects, plans = [], [], []
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Application(ApplicationConfig(OBJECTIVE, root, {}, {}), durable=True,
                             modules=modules(windows, effects, plans)) as app:
                first = app.runner.step(OBJECTIVE).event
                self.assertEqual(first['phase'], 'host_finish')
                self.assertEqual(first['receipt']['data'], {'metrics': {FINAL_METRIC: 0.375}})
            with Application.resume(root, modules=modules(windows, effects, plans)) as app:
                result = app.run(wait_for_jobs=True)
                self.assertEqual((result.status, result.stop_reason), ('finished', 'researcher_finished'))
                finals = [event for event in result.events if event.get('phase') == 'host_finish']
                self.assertEqual([event['operation']['call_id'] for event in finals], ['host-final-b'])
                self.assertEqual(finals[0]['receipt']['data'], first['receipt']['data'])
                rows = app.experience.retrieve(kinds=('evaluation',))
                self.assertEqual([row.record_id for row in rows], ['sealed-host-final-a', 'sealed-host-final-b'])
                ref = ArtifactRef(**rows[0].details['receipt']['artifacts'][0])
                self.assertIn(ROW.encode(), app.artifacts.read(ref))
                with self.assertRaises(PermissionError):
                    app.assembly.evidence_view().read(ref)
                self.assertNotIn(ROW, repr((first, result, rows)))
                self.assertNotIn(FINAL_METRIC, repr(windows))
            self.assertEqual((effects, plans, len(windows)), (['host-final-a', 'host-final-b'], ['planned'], 1))


class MeasuredLab(SealedLab):
    def __init__(self, plan):
        super().__init__([], [])
        self.plan = plan

    def describe(self):
        return (*super().describe(), OperationSpec('select', 'Select a synthetic candidate', {'type': 'object'}))

    def invariants(self):
        return (Invariant.baseline_before('select'), Invariant.measured_selection())

    def final_evaluations(self):
        return tuple(OperationCall('evaluate', {'split': 'heldout', 'candidate_id': name}, call_id='final-' + name)
                     for name in self.plan)

    def execute(self, call):
        if call.arguments.get('split') == 'heldout':
            return super().execute(call)
        return OperationResult('Fixture search', (), True,
            data={'baseline': call.arguments.get('candidate_id') == 'baseline'})


class SealedPlanTest(unittest.TestCase):
    def prepare(self, plan, selected='selected'):
        lab = MeasuredLab(plan)
        group = LabGroup(artifacts=InMemoryArtifactStore())
        self.addCleanup(group.close)
        group.add('fixture', lab)
        for name in ('baseline', 'selected', 'unrelated'):
            receipt = group.execute(OperationCall('fixture.evaluate', {'split': 'dev', 'candidate_id': name}))
            self.assertEqual(receipt.status, ExecutionStatus.COMPLETED)
        receipt = group.execute(OperationCall('fixture.select', {'candidate_id': selected}))
        self.assertEqual(receipt.status, ExecutionStatus.COMPLETED)
        return group, lab

    def test_plan_accepts_measured_baseline_and_selected_or_baseline_as_selection(self):
        for plan, selection in [(('baseline', 'selected'), 'selected'), (('baseline',), 'baseline')]:
            with self.subTest(plan=plan, selection=selection):
                group, lab = self.prepare(plan, selection)
                for name in plan:
                    self.assertEqual(group.finish_step()['operation']['call_id'], 'final-' + name)
                self.assertIsNone(group.finish_step())
                self.assertEqual(lab.effects, ['final-' + name for name in plan])

    def test_entire_plan_rejects_wrong_candidate_before_any_sealed_effect(self):
        for plan, message in [(('baseline',), 'must include the measured selection'),
                              (('baseline', 'selected', 'unrelated'), 'may only measure the selection')]:
            with self.subTest(plan=plan):
                group, lab = self.prepare(plan)
                with self.assertRaisesRegex(ValueError, message):
                    group.finish_step()
                self.assertEqual(lab.effects, [])
                self.assertIsNone(group.profile_state()['final_calls'])

    def test_restored_legacy_plan_accepts_named_measured_baseline(self):
        group, lab = self.prepare(('baseline', 'selected'))
        group.finish_step()
        saved = group.profile_state()
        del saved['bindings']['fixture']['baseline_candidates']
        group.restore_profile_state(saved)
        self.assertEqual(group.finish_step()['operation']['call_id'], 'final-selected')
        self.assertEqual(lab.effects, ['final-baseline', 'final-selected'])

    def test_restored_plan_must_still_bind_the_measured_selection(self):
        group, lab = self.prepare(('baseline', 'selected'))
        group.finish_step()
        saved = group.profile_state()
        selected = saved['final_calls'][1]
        saved['final_calls'] = (saved['final_calls'][0], replace(selected,
            arguments={**selected.arguments, 'candidate_id': 'unrelated'}))
        with self.assertRaisesRegex(ValueError, 'must include the measured selection'):
            group.restore_profile_state(saved)
        self.assertEqual(lab.effects, ['final-baseline'])


if __name__ == '__main__':
    unittest.main()
