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
from rsicomposer.loops.contracts.execution import LabToolError
from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.model_service import BackendBinding, ModelRef
from rsicomposer.loops.contracts.profiles import DEV, PROGRAM_EXECUTION, SEALED, Invariant, MetricSpec
from rsicomposer.registry import Registry
from rsicomposer.research import MemoryAssembly, ResearchAssembly
from rsicomposer.runtime.evidence.records import ExperienceRecord
from rsicomposer.runtime.evidence.retrieval import controller_experience
from rsicomposer.runtime.orchestration.lab_group import LabGroup
from rsicomposer.runtime.orchestration.run_policy import EvaluationOutcome, GoalRunPolicy, MetricGoal, RunDecision, TimeRunPolicy
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
        return (OperationSpec('score', 'Score a synthetic fixture', {'type': 'object'}),)

    def split_roles(self):
        return {'dev': DEV, 'heldout': SEALED}

    def evaluation_operations(self):
        return ('score',)

    def final_evaluations(self):
        self.plans.append('planned')
        return tuple(OperationCall('score', {'split': 'heldout'}, call_id=ident)
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


def modules(windows, effects, plans, *, memory=None, lab_type=SealedLab):
    registry = Registry(builtins=False)
    registry.register_controller('fixture', ControllerDefinition(
        lambda context, options: FinishPolicy(windows), memory=memory), version='1')
    registry.register_lab('fixture', lambda context, options: lab_type(effects, plans), version='1')
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
                self.assertEqual(first['operation']['name'], 'fixture.score')
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

    def test_public_final_evidence_rebuilds_typed_aggregates_on_success_and_failure(self):
        good = {'quality_metrics': {'accuracy': .75}, 'training_metrics': {'loss': 0},
                'execution_metrics': {'seconds': 1.25}, 'metrics': {'other': -1},
                'valid': False, 'scored': False, 'count': 0}
        row = {'items': [ROW]}
        success, failed = ExecutionStatus.COMPLETED, ExecutionStatus.FAILED
        cases = {
            'good': (OperationResult(ROW, (), False, data={**good, 'rows': [row]}), success, good),
            'bad-count': (OperationResult(ROW, (), False, data={'count': row, 'valid': True}), failed, {'valid': True}),
            'bad-types': (OperationResult(ROW, (), False, data={'count': -1, 'valid': [row], 'scored': 1}), failed, {}),
            'bad-metrics': (OperationResult(ROW, (), False, data={'metrics': {'mean': .25, 'details': [row]},
                'count': 2}), failed, {'metrics': {'mean': .25}, 'count': 2}),
            'already-failed': (OperationResult(ROW, (), True, failed, ROW, data={'metrics': [row], 'scored': False,
                'valid': ROW, 'count': True}), failed, {'scored': False}),
            'bad-mode': (OperationResult(ROW, (), row, data={'count': 1.5}), failed, {}),
        }

        class TypedLab(SealedLab):
            def final_evaluations(self):
                return tuple(OperationCall('score', {'split': 'heldout'}, call_id=name) for name in cases)

            def execute(self, call):
                return cases[call.call_id][0] if call.arguments['split'] == 'heldout' else cases['bad-count'][0]

        with TemporaryDirectory() as directory, Application(ApplicationConfig(OBJECTIVE, Path(directory), {}, {}),
                modules=modules([], [], [], lab_type=TypedLab)) as app:
            group = app.assembly.lab_execution
            dev = group.execute(OperationCall('fixture.score', {'split': 'dev'}))
            self.assertEqual((dev.status, dev.data['count']), (success, row))  # Native dev data is unchanged.
            run = app.run(wait_for_jobs=True)
            finals = {e['operation']['call_id']: e for e in run.events if e.get('phase') == 'host_finish'}
            records = app.experience.retrieve(kinds=('evaluation',))
            self.assertEqual(set(finals), set(cases))
            self.assertEqual(len(records), len(cases))
            self.assertNotIn(ROW, repr((run.events, records)))
            for name, (raw, expected_status, expected_data) in cases.items():
                with self.subTest(case=name):
                    event = finals[name]
                    journal = group.stored_receipt(OperationCall(**event['operation']))
                    self.assertEqual((journal.status, journal.data), (expected_status, expected_data))
                    self.assertIs(type(journal.simulated), bool)
                    self.assertEqual(event['receipt'], journal.payload())
                    record, = (r for r in records if r.record_id == 'sealed-' + name)
                    self.assertEqual(record.details['receipt']['data'], expected_data)
                    ref, = journal.artifacts
                    self.assertEqual(json.loads(app.artifacts.read(ref)),
                                     json.loads(json.dumps(replace(raw, call_id=name).payload())))
                    with self.assertRaises(PermissionError):
                        app.assembly.evidence_view().read(ref)


class MeasuredLab(SealedLab):
    def __init__(self, plan):
        super().__init__([], [])
        self.plan = plan

    def describe(self):
        return (*super().describe(), OperationSpec('select', 'Select a synthetic candidate', {'type': 'object'}))

    def invariants(self):
        return (Invariant.baseline_before('select'), Invariant.measured_selection())

    def candidate_identity_fields(self):
        return {'score': ('candidate_id',), 'select': ('candidate_id',)}

    def final_evaluations(self):
        return tuple(OperationCall('score', {'split': 'heldout', 'candidate_id': name}, call_id='final-' + name)
                     for name in self.plan)

    def execute(self, call):
        if call.arguments.get('split') == 'heldout':
            return super().execute(call)
        return OperationResult('Fixture search', (), True,
            data={'baseline': call.arguments.get('candidate_id') == 'baseline'})


class CandidateIdentityTest(unittest.TestCase):
    class Lab(SealedLab):
        def __init__(self):
            super().__init__([], [])
            self.final = None

        def describe(self):
            return (*super().describe(), OperationSpec('choose', 'Choose a measured target', {}))

        def candidate_identity_fields(self):
            return {'score': ('encoder', 'index.sha256'), 'choose': ('model_part', 'index_part')}

        def invariants(self):
            return (Invariant.measured_selection('choose'), Invariant.min_candidates(2))

        def final_evaluations(self):
            return (OperationCall('score', {**self.final, 'split': 'heldout'}),)

        def execute(self, call):
            self.effects.append(call.call_id)
            if call.name == 'choose':
                return OperationResult('chosen', (), False)
            data = {'quality_metrics': {'accuracy': .9}}
            if call.arguments.get('misreport'):
                data['candidate_id'] = 'fixed-model'
            if call.arguments.get('normalize'):
                call.arguments['index']['sha256'] = 'changed-by-worker'
            return OperationResult('measured', (), False, data=data)

    def group(self, lab):
        group = LabGroup(artifacts=InMemoryArtifactStore())
        self.addCleanup(group.close)
        group.add('search', lab)
        return group

    def test_distinct_composite_targets_count_select_and_seal_with_one_identity(self):
        lab = self.Lab()
        group = self.group(lab)
        a = {'model': 'fixed-model', 'encoder': {'rev': 1, 'seed': 4}, 'index': {'sha256': 'A', 'path': '/one'}}
        b = {**a, 'index': {'sha256': 'B'}}
        def score(target, **extra):
            return group.execute(OperationCall('search.score', {'split': 'dev', **target, **extra}))
        def choose(index):
            return group.execute(OperationCall('search.choose', {'model': 'fixed-model',
                'model_part': {'seed': 4, 'rev': 1}, 'index_part': index}))

        self.assertEqual(score(a, normalize=True).status, ExecutionStatus.COMPLETED)
        same = {**a, 'encoder': {'seed': 4, 'rev': 1}, 'index': {'path': '/two', 'sha256': 'A'}}
        self.assertEqual(score(same).status, ExecutionStatus.COMPLETED)
        self.assertEqual(choose('A').status, ExecutionStatus.COMPLETED)
        self.assertEqual(choose('B').status, ExecutionStatus.FAILED)
        self.assertEqual(score(b, misreport=True).status, ExecutionStatus.FAILED)
        self.assertEqual(choose('B').status, ExecutionStatus.FAILED)
        lab.final = b
        with self.assertRaisesRegex(ValueError, 'at least 2 measured candidates'):
            group.finish_step()
        self.assertEqual(score(b).status, ExecutionStatus.COMPLETED)
        before = list(lab.effects)
        with self.assertRaisesRegex(ValueError, 'must include the measured selection'):
            group.finish_step()  # B was measured, but A was selected.
        self.assertEqual(lab.effects, before)
        self.assertEqual(choose('B').status, ExecutionStatus.COMPLETED)
        self.assertEqual(group.finish_step()['receipt']['status'], ExecutionStatus.COMPLETED)
        state = group.profile_state()['bindings']['search']
        self.assertEqual(len(state['candidates']), 2)
        self.assertIn(state['selected'], state['measured'])

    def test_required_declaration_and_missing_parts_fail_before_effects(self):
        for declared in ({}, {'choose': ('id',)}, {'score': ('id',)},
                         {'score': ('encoder', 'idx'), 'choose': ('encoder',)}):
            with self.subTest(declared=declared):
                lab = self.Lab()
                lab.candidate_identity_fields = lambda: declared
                with self.assertRaisesRegex(ValueError, 'candidate'):
                    self.group(lab)
        lab = self.Lab()
        group = self.group(lab)
        for missing in ({'model': 'fixed'}, {'encoder': 'e', 'index': {}},
                        {'encoder': 'e', 'index': {'sha256': '  '}}, {'encoder': None, 'index': {'sha256': 'A'}}):
            call = OperationCall('search.score', {**missing, 'split': 'dev', 'candidate_id': 'untrusted'})
            self.assertEqual(group.execute(call).status, ExecutionStatus.FAILED)
        self.assertEqual(lab.effects, [])
        self.assertEqual(group.profile_state()['bindings']['search']['measured'], [])

    def test_untyped_or_changed_identity_cannot_recover_measured_history(self):
        original = self.group(self.Lab())
        target = {'model': 'fixed', 'encoder': 'e1', 'index': {'sha256': 'A'}, 'split': 'dev'}
        original.execute(OperationCall('search.score', target))
        saved = original.profile_state()
        changed = self.Lab()
        changed.candidate_identity_fields = lambda: {'score': ('model', 'index.sha256'),
                                                       'choose': ('model', 'index_part')}
        group = self.group(changed)
        with self.assertRaisesRegex(ValueError, 'profile history'):
            group.restore_profile_state(saved)
        del saved['bindings']['search']['candidate_identity_fields']
        with self.assertRaisesRegex(ValueError, 'profile history'):
            original.restore_profile_state(saved)

    def test_composite_replay_cannot_conflate_json_scalar_types(self):
        group = self.group(self.Lab())
        def call(flag, ident):
            return OperationCall('search.score', {'encoder': {'enabled': flag}, 'index': {'sha256': 'A'},
                'split': 'dev'}, call_id=ident)
        self.assertEqual(group.execute(call(True, 'once')).status, ExecutionStatus.COMPLETED)
        for value in (1, 1.0):
            with self.assertRaisesRegex(ValueError, 'different operation'):
                group.execute(call(value, 'once'))
            self.assertEqual(group.execute(call(value, str(value))).status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(group.profile_state()['bindings']['search']['measured']), 3)

    def test_delivered_identity_matches_measurement_history_and_policy_only_when_declared(self):
        class Measure(self.Lab):
            def invariants(self):
                return ()

            def final_evaluations(self):
                return ()

            def primary_metric(self):
                return MetricSpec('accuracy', 'maximize', 'fixture/v1')

            def execute(self, call):
                result = super().execute(call)
                if not declared:
                    # Native data and request claims are not an installed identity declaration.
                    return replace(result, data={**result.data, 'candidate_id': 'fixed-model'})
                return result

        class OneCall:
            def decide(self, window):
                return Decision('finish') if window.feedbacks else Decision.invoke(call)

        for declared in (False, True):
            with self.subTest(declared=declared), TemporaryDirectory() as directory:
                lab = Measure()
                if not declared:
                    lab.candidate_identity_fields = lambda: {}
                call = OperationCall('search.score', {'candidate_id': 'fixed-model', 'model': 'fixed-model',
                    'encoder': {'rev': 2, 'seed': 7}, 'index': {'sha256': 'A'}, 'split': 'dev'})
                registry = Registry(builtins=False)
                registry.register_controller('policy', lambda *_: OneCall(), version='1')
                registry.register_lab('lab', lambda *_: lab, version='1')
                config = ApplicationConfig(OBJECTIVE, Path(directory), {}, {})
                selection = registry.build({'controller': {'name': 'policy'}, 'labs': {'search': {'name': 'lab'}}})
                goal = GoalRunPolicy(MetricGoal('accuracy', .8, 'fixture/v1', input_refs=('split:dev',)))
                with Application(config, modules=selection, run_policy=goal) as app:
                    result = app.run(wait_for_jobs=True, max_decision_steps=1)
                    group = app.assembly.lab_execution
                    native = group.stored_receipt(call)
                    outcome = app.runner._outcomes[call.call_id]
                    measured = group.profile_state()['bindings']['search']['measured']
                    if declared:
                        self.assertEqual(result.stop_reason, 'goal_reached')
                        self.assertNotIn('candidate_id', native.data)
                        self.assertEqual(measured, [outcome.candidate_id])
                        self.assertTrue(outcome.candidate_id.startswith('candidate-json-sha256:'))
                        self.assertEqual(outcome.receipt.data['candidate_id'], outcome.candidate_id)
                    else:
                        self.assertNotEqual(result.stop_reason, 'goal_reached')
                        self.assertEqual(measured, [])
                        self.assertNotIsInstance(outcome, EvaluationOutcome)


class SealedPlanTest(unittest.TestCase):
    def test_only_boolean_evaluation_flags_can_establish_measured_history(self):
        class Lab(MeasuredLab):
            def execute(self, call):
                # No primary_metric: the Lab's successful receipt is still a valid baseline.
                data = call.arguments.get('flags', {}) if call.name == 'score' else {'scored': 'native selection note'}
                return OperationResult('fixture', (), False, data=data)

        group = LabGroup(artifacts=InMemoryArtifactStore())
        self.addCleanup(group.close)
        group.add('fixture', Lab(()))
        def score(ident, **flags):
            return group.execute(OperationCall('fixture.score', {'candidate_id': ident, 'split': 'dev', 'flags': flags}))
        def select(ident):
            return group.execute(OperationCall('fixture.select', {'candidate_id': ident}))

        for flag, value in (('valid', 0), ('valid', None), ('scored', 'false'), ('scored', 1)):
            with self.subTest(flag=flag, value=value):
                bad = score('bad', baseline=True, **{flag: value})
                self.assertEqual(bad.status, ExecutionStatus.FAILED)
                self.assertIn('must be boolean', bad.error)
        self.assertFalse(group.profile_state()['bindings']['fixture']['baseline'])
        self.assertEqual(score('baseline', baseline=True).status, ExecutionStatus.COMPLETED)
        self.assertEqual(score('invalid', valid=False).status, ExecutionStatus.COMPLETED)
        self.assertEqual(score('unscored', scored=False).status, ExecutionStatus.COMPLETED)
        for ident in ('bad', 'invalid', 'unscored'):
            self.assertEqual(select(ident).status, ExecutionStatus.FAILED)
        self.assertEqual(score('ok', valid=True, scored=True).status, ExecutionStatus.COMPLETED)
        self.assertEqual(select('ok').status, ExecutionStatus.COMPLETED)

    def prepare(self, plan, selected='selected'):
        lab = MeasuredLab(plan)
        group = LabGroup(artifacts=InMemoryArtifactStore())
        self.addCleanup(group.close)
        group.add('fixture', lab)
        for name in ('baseline', 'selected', 'unrelated'):
            receipt = group.execute(OperationCall('fixture.score', {'split': 'dev', 'candidate_id': name}))
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

    def test_restored_plan_never_infers_baseline_identity_from_a_candidate_name(self):
        group, lab = self.prepare(('baseline', 'selected'))
        group.finish_step()
        saved = group.profile_state()
        del saved['bindings']['fixture']['baseline_candidates']
        with self.assertRaisesRegex(ValueError, 'Invalid Lab profile history'):
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

    def test_missing_final_prefix_fails_restore_without_changing_current_plan_or_history(self):
        effects = []
        group = LabGroup(artifacts=InMemoryArtifactStore())
        lab = SealedLab(effects, [])
        group.add('fixture', lab)
        before = group.profile_state()
        state = group.profile_state()
        state['bindings']['fixture']['baseline'] = True
        state['final_calls'] = tuple(replace(call, name='fixture.score') for call in lab.final_evaluations())
        state['final_index'] = 1
        with self.assertRaisesRegex(ValueError, 'missing applied receipt'):
            group.restore_profile_state(state)
        self.assertEqual(group.profile_state(), before)
        self.assertEqual(effects, [])

    def test_published_final_ids_cannot_be_taken_by_alias_with_no_final_operations(self):
        effects = []
        group = LabGroup(artifacts=InMemoryArtifactStore())
        other = SealedLab(effects, [])
        other.final_evaluations = lambda: ()
        group.add('fixture', SealedLab(effects, []))
        group.add('other', other)
        self.assertEqual(group.finish_step()['operation']['call_id'], 'host-final-a')
        collision = OperationCall('other.score', {'split': 'dev'}, call_id='host-final-b')
        self.assertEqual(group.execute(collision).status, ExecutionStatus.FAILED)
        self.assertIsNone(group.stored_receipt(collision))
        self.assertEqual(group.finish_step()['operation']['call_id'], 'host-final-b')
        self.assertEqual(effects, ['host-final-a', 'host-final-b'])

    def test_final_call_ids_are_unique_across_all_labs_before_any_effect(self):
        for aliases in (('a',), ('a', 'b')):
            with self.subTest(aliases=aliases), TemporaryDirectory() as root:
                effects = []
                group = LabGroup(receipts=root, artifacts=InMemoryArtifactStore())
                plan = (OperationCall('score', {'split': 'heldout'}, call_id='final-score'),)
                for alias in aliases:
                    lab = SealedLab(effects, [])
                    lab.final_evaluations = lambda: plan * (2 if len(aliases) == 1 else 1)
                    group.add(alias, lab)
                checkpoints = []
                with self.assertRaisesRegex(ValueError, 'duplicate call IDs'):
                    group.finish_step(before_execute=checkpoints.append)
                self.assertEqual((effects, checkpoints), ([], []))
                saved = group.profile_state()
                self.assertIsNone(saved['final_calls'])
                saved['final_calls'] = tuple(replace(call, name=alias + '.score') for alias in aliases
                                             for call in group.labs[alias].final_evaluations())
                with self.assertRaisesRegex(ValueError, 'duplicate call IDs'):
                    group.restore_profile_state(saved)
                self.assertIsNone(group.profile_state()['final_calls'])

    def test_preflight_checks_existing_call_ids_across_mounts_on_build_and_restore(self):
        for location in ('disk', 'memory'):
            with self.subTest(location=location), TemporaryDirectory() as root:
                effects = []
                group = LabGroup(receipts=root if location == 'disk' else None, artifacts=InMemoryArtifactStore())
                for name in ('first', 'second'):
                    lab = SealedLab(effects, [])
                    lab.final_evaluations = lambda name=name: (
                        OperationCall('score', {'split': 'heldout'}, call_id=name),)
                    group.add(name, lab)
                prior = OperationCall('first.score', {'split': 'dev'}, call_id='second')
                self.assertEqual(group.execute(prior).status, ExecutionStatus.COMPLETED)
                with self.assertRaisesRegex(ValueError, 'different operation'):
                    group.finish_step()
                self.assertEqual(effects, ['second'])
                saved = group.profile_state()
                self.assertIsNone(saved['final_calls'])
                saved['final_calls'] = tuple(replace(call, name=alias + '.score') for alias, lab in group.labs.items()
                                             for call in lab.final_evaluations())
                with self.assertRaisesRegex(ValueError, 'different operation'):
                    group.restore_profile_state(saved)
                self.assertIsNone(group.profile_state()['final_calls'])
                self.assertEqual(effects, ['second'])

    def test_restored_final_plan_replays_matching_receipt_after_evidence_write_failed(self):
        from unittest.mock import Mock
        effects = []
        records = Mock()
        records.record.side_effect = OSError('evidence write failed')
        with TemporaryDirectory() as root:
            artifacts = InMemoryArtifactStore()
            group = LabGroup(receipts=root, artifacts=artifacts, experience=records)
            group.add('fixture', SealedLab(effects, []))
            with self.assertRaisesRegex(OSError, 'evidence write failed'):
                group.finish_step()
            saved = group.profile_state()
            self.assertEqual((saved['final_index'], effects), (0, ['host-final-a']))
            restored = LabGroup(receipts=root, artifacts=artifacts)
            restored.add('fixture', SealedLab(effects, []))
            # A fresh plan is not allowed to claim a previous final's ID; the
            # restored plan, however, owns this exact receipt at the cursor.
            with self.assertRaisesRegex(ValueError, 'already recorded'):
                restored.finish_step()
            restored.restore_profile_state(saved)
            self.assertEqual(restored.finish_step()['operation']['call_id'], 'host-final-a')
            self.assertEqual(effects, ['host-final-a'])
            self.assertEqual(restored.finish_step()['operation']['call_id'], 'host-final-b')
            self.assertEqual(effects, ['host-final-a', 'host-final-b'])

    def test_new_and_restored_plans_require_declared_measurement_and_sealed_role(self):
        class Lab(SealedLab):
            def describe(self):
                return (*super().describe(), OperationSpec('evaluate', 'Not a declared measurement', {}))

            def final_evaluations(self):
                return (OperationCall('score', {'split': 'heldout'}), invalid)

        for invalid, message in ((OperationCall('evaluate', {'split': 'heldout'}), 'not a declared evaluation'),
                                 (OperationCall('score', {'split': 'dev'}), 'declared sealed split')):
            with self.subTest(call=invalid):
                group, lab = LabGroup(artifacts=InMemoryArtifactStore()), Lab([], [])
                group.add('fixture', lab)
                with self.assertRaisesRegex(ValueError, message):
                    group.finish_step()
                saved = group.profile_state()
                self.assertIsNone(saved['final_calls'])
                saved['final_calls'] = tuple(replace(c, name='fixture.' + c.name) for c in lab.final_evaluations())
                with self.assertRaisesRegex(ValueError, message):
                    group.restore_profile_state(saved)
                self.assertEqual(lab.effects, [])

    def test_program_profile_allows_heldout_execute_without_requiring_a_quality_score(self):
        class Program:
            def describe(self):
                return PROGRAM_EXECUTION.operations

            def profiles(self):
                return (PROGRAM_EXECUTION.identity,)

            def primary_metric(self):
                return MetricSpec('accuracy', 'maximize', 'fixture/v1')

            def split_roles(self):
                return {'heldout': SEALED}

            def final_evaluations(self):
                return (OperationCall('execute', {'program': 'fixture', 'split': 'heldout'}),)

            def execute(self, call):
                return OperationResult('Unscored program', (), False, data={
                    'term_out': ROW, 'exc_type': 'ProgramError', 'exec_time': 0., 'metrics': {}})

        group = LabGroup(artifacts=InMemoryArtifactStore())
        group.add('program', Program())
        self.assertTrue(group.is_evaluation('program.execute'))
        event = group.finish_step()
        self.assertEqual(event['receipt']['status'], ExecutionStatus.COMPLETED)
        self.assertEqual(event['receipt']['data'], {'metrics': {}})
        self.assertNotIn(ROW, repr(event))

    def test_sealed_domain_rejection_is_redacted_before_journaling(self):
        class Rejected(SealedLab):
            def execute(self, call):
                raise LabToolError(ROW, data={'rows': [ROW]})

        artifacts = InMemoryArtifactStore()
        with TemporaryDirectory() as root:
            group = LabGroup(artifacts=artifacts, receipts=root)
            group.add('fixture', Rejected([], []))
            event = group.finish_step()
            call = OperationCall(**event['operation'])
            saved = group.stored_receipt(call)
            self.assertEqual((saved.status, saved.call_id, saved.data), (ExecutionStatus.FAILED, call.call_id, {}))
            self.assertEqual(event['receipt'], saved.payload())
            self.assertNotIn(ROW, repr(event))
            self.assertIn(ROW.encode(), artifacts.read(saved.artifacts[0]))


if __name__ == '__main__':
    unittest.main()
