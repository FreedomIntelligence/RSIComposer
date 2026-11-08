"""Standalone stdlib tests for the isolated read-only monitor service."""
from tests import ROOT as REPO_ROOT
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rsimonitor import server as monitor


class MonitorProjectionTests(unittest.TestCase):
    """Projection and file-store regressions without opening a listening socket."""

    def project(self, **fields):
        return monitor.project_snapshot({'schema_version': 1, **fields}, 'run-1')

    def test_interpretation_projection_preserves_missing_members_and_public_provenance(self):
        item = {'attempt_id': 'a', 'title': 'Idea', 'summary': 'Interpretation only',
                'member_decision_ids': ['outside-window', 'inside'], 'source_refs': ['artifact:report'],
                'revision': 2, 'simulated': False, 'private': 'PRIVATE',
                'provenance': {'job_id': 'j', 'input_digest': 'digest', 'secret': 'PRIVATE',
                               'model': {'backend_id': 'b', 'model_id': 'm', 'revision': 'r', 'secret': 'PRIVATE'}}}
        view = {'schema_version': 1, 'authority': 'model_interpretation', 'items': [item],
                'revisions': [{**item, 'revision': 1}], 'ungrouped_decision_ids': ['other']}
        projected = self.project(decisions=[{'decision_id': 'inside'}], research_attempts=view)
        self.assertEqual(projected['research_attempts']['items'][0]['member_decision_ids'], ['outside-window', 'inside'])
        self.assertEqual(projected['research_attempts']['items'][0]['provenance']['model']['model_id'], 'm')
        self.assertEqual(projected['research_attempts']['revisions'][0]['revision'], 1)
        self.assertNotIn('research_attempt_id', projected['decisions'][0])
        self.assertNotIn('PRIVATE', json.dumps(projected))
        self.assertEqual(self.project(research_attempts={**view, 'items': []})['research_attempts']['items'], [])
        self.assertNotIn('research_attempts', self.project(research_attempts={**view, 'authority': 'untrusted'}))

    def test_explicit_action_changes_and_outcomes_are_projected_without_raw_payloads(self):
        data = self.project(decisions=[{'reason': 'Improve the answer format',
            'changes': [{'field': 'instructions', 'after': 'Answer concisely.', 'payload': 'PRIVATE'},
                        {'field': 'steps', 'before': '16', 'after': '32'}],
            'feedback_summary': 'Prompt proposed; evaluate before application',
            'error': 'Evaluation rejected', 'payload': {'secret': 'PRIVATE'}, 'reasoning_content': 'PRIVATE'}])
        row = data['decisions'][0]
        self.assertEqual(row['changes'], [{'field': 'instructions', 'after': 'Answer concisely.'},
                                         {'field': 'steps', 'before': '16', 'after': '32'}])
        self.assertEqual(row['feedback_summary'], 'Prompt proposed; evaluate before application')
        self.assertEqual(row['error'], 'Evaluation rejected')
        self.assertNotIn('PRIVATE', json.dumps(data))

    def test_action_change_projection_has_bounded_plain_text_fields(self):
        changes = [{'field': 'x' * 101, 'after': '<script>' + 'x' * 5000, 'before': {'secret': 'hidden'}}] * 30
        rows = self.project(decisions=[{'changes': changes}, {'changes': [None, 3, {},
            {'field': 'valid', 'after': {'nested': 'hidden'}}, {'field': 'instructions', 'after': ''}]}])['decisions']
        self.assertEqual(len(rows[0]['changes']), 20)
        self.assertEqual(len(rows[0]['changes'][0]['field']), 100)
        self.assertEqual(len(rows[0]['changes'][0]['after']), 4000)
        self.assertNotIn('before', rows[0]['changes'][0])
        self.assertEqual(rows[1]['changes'], [{'field': 'instructions', 'after': ''}])

    def test_public_workflow_execution_decisions_and_round_results(self):
        workflow = {'phase': 'training', 'round': 2, 'rounds': 3, 'completed_directives': 5,
                    'elapsed_seconds': 12.5}
        current = {'directive_id': 'd2', 'mode': 'train', 'status': 'running', 'phase': 'training',
                   'strategy': 'Train another candidate', 'decision_revision': 'decision-v1',
                   'actor_revision': 'actor-v2', 'job_id': 'j2', 'elapsed_seconds': 3.5,
                   'review_after_seconds': 2, 'next_review_seconds': 4, 'review_status': 'continue',
                   'step': 3, 'total_steps': 10, 'metrics': {'loss': 0.4, 'eval_samples': 32}}
        decision = {'decision_id': 'review-2', 'timestamp': '2026-11-01T00:00:00Z', 'kind': 'review',
                    'source': 'model', 'round': 2, 'directive_id': 'd2', 'mode': 'train', 'action': 'continue',
                    'strategy': 'Train another candidate', 'reason': 'Loss is still improving', 'status': 'accepted',
                    'decision_revision': 'decision-v1', 'actor_revision': 'actor-v2', 'job_id': 'j2',
                    'candidate_id': 'c2', 'protocol_ref': 'heldout-nll', 'recipe_ref': 'toy-lora', 'method': 'sft',
                    'review_after_seconds': 2, 'next_review_seconds': 4}
        round_result = {'round': 1, 'base_revision': 'base', 'candidate_revision': 'actor-v2',
                        'selection': 'accepted', 'actor_revision': 'actor-v2', 'applied': True,
                        'answer_correct': False, 'training_review_count': 2,
                        'metrics': {'delta_nll': 0.1, 'candidate_nll': 1.5}}
        job = {'job_id': 'j2', 'kind': 'training', 'status': 'running', 'directive_id': 'd2', 'round': 2,
               'step': 3, 'total_steps': 10, 'elapsed_seconds': 3.5, 'metrics': {'loss': 0.4}}
        data = self.project(workflow=workflow, current=current, decisions=[decision],
                            round_results=[round_result], jobs=[job])
        self.assertEqual(data['workflow'], workflow)
        self.assertEqual(data['current'], current)
        self.assertEqual(data['decisions'], [decision])
        self.assertEqual(data['round_results'], [round_result])
        self.assertEqual(data['jobs'], [job])
        self.assertEqual(data['schema_version'], 1)

    def test_old_snapshots_and_malformed_optional_containers_have_empty_states(self):
        legacy = self.project(jobs=[{'job_id': 'j', 'kind': 'training', 'status': 'running'}])
        self.assertEqual(legacy['jobs'], [{'job_id': 'j', 'kind': 'training', 'status': 'running'}])
        for data in (legacy, *(self.project(workflow=value, current=value, decisions=value, round_results=value)
                              for value in (None, 'broken', 42, True))):
            self.assertEqual(data['workflow'], {})
            self.assertEqual(data['current'], {})
            self.assertEqual(data['decisions'], [])
            self.assertEqual(data['round_results'], [])
        data = self.project(workflow=[], current=[], decisions={}, round_results={})
        self.assertEqual((data['workflow'], data['current'], data['decisions'], data['round_results']),
                         ({}, {}, [], []))
        data = self.project(decisions=[None, [], 7, {'action': 'continue'}],
                            round_results=[False, 'broken', {'round': 1, 'metrics': []}],
                            current={'metrics': 'broken'}, jobs=[{'metrics': []}])
        self.assertEqual(data['decisions'], [{'action': 'continue'}])
        self.assertEqual(data['round_results'], [{'round': 1, 'metrics': {}}])
        self.assertEqual(data['current'], {'metrics': {}})
        self.assertEqual(data['jobs'], [{'metrics': {}}])

    def test_new_fields_never_forward_private_attachments_or_non_numeric_metrics(self):
        private = {key: {'contents': 'TOP_SECRET'} for key in
                   ('payload', 'raw_payload', 'prompt', 'response', 'config', 'credentials', 'reasoning')}
        metrics = {'loss': 0.4, 'eval_samples': 32, 'accuracy': {'baseline': 0.5, 'candidate': 0.8},
                   'text': 'TOP_SECRET', 'flag': True, 'values': ['TOP_SECRET'], 'prompt_tokens': 50,
                   'raw_payload': {'loss': 3}, 'response': {'loss': 4}, 'config': {'loss': 5},
                   'credentials': {'loss': 6}, 'NaN': float('nan'), 'inf': float('inf'), 'huge': 10 ** 1000}
        raw = {'schema_version': 1, 'workflow': {'phase': 'training', **private},
               'current': {'strategy': 'Public strategy', 'metrics': metrics, **private},
               'decisions': [{'reason': 'Public decision explanation', **private}],
               'round_results': [{'round': 1, 'metrics': metrics, **private}],
               'jobs': [{'job_id': 'j', 'metrics': metrics, **private}]}
        original = deepcopy(raw)
        data = monitor.project_snapshot(raw, 'run-1')
        safe_metrics = {'loss': 0.4, 'eval_samples': 32, 'accuracy': {'baseline': 0.5, 'candidate': 0.8}}
        self.assertEqual(data['workflow'], {'phase': 'training'})
        self.assertEqual(data['decisions'], [{'reason': 'Public decision explanation'}])
        self.assertEqual(data['current'], {'strategy': 'Public strategy', 'metrics': safe_metrics})
        self.assertEqual(data['round_results'], [{'round': 1, 'metrics': safe_metrics}])
        self.assertEqual(data['jobs'], [{'job_id': 'j', 'metrics': safe_metrics}])
        self.assertNotIn('TOP_SECRET', json.dumps(data, allow_nan=False))
        self.assertEqual(raw, original)

    def test_event_details_accept_public_decision_fields_but_not_arbitrary_objects(self):
        details = {'directive_id': 'd', 'decision_id': 'review', 'round': 2, 'source': 'model',
                   'action': 'cancel', 'mode': 'train', 'reason': 'Public cancellation reason',
                   'strategy': 'Train a candidate', 'decision_revision': 'd-v1', 'actor_revision': 'a-v2',
                   'candidate_revision': 'a-v3', 'review_after_seconds': 2, 'next_review_seconds': None,
                   'elapsed_seconds': 8, 'total_steps': 20, 'selection': 'rejected', 'applied': False,
                   'answer_correct': True, 'training_review_count': 3}
        data = self.project(events=[{'kind': 'review', 'details': {
            **details, 'payload': {'raw': 'TOP_SECRET'}, 'response': 'TOP_SECRET',
            'config': 'TOP_SECRET', 'credentials': 'TOP_SECRET', 'metrics': {'loss': 0.4}}}])
        self.assertEqual(data['events'][0]['details'], details)
        bad = self.project(events=[{'details': {'reason': {'prompt': 'TOP_SECRET'}}}],
                           decisions=[{'strategy': ['TOP_SECRET']}])
        self.assertIsNone(bad['events'][0]['details']['reason'])
        self.assertIsNone(bad['decisions'][0]['strategy'])
        self.assertNotIn('TOP_SECRET', json.dumps(bad))

    def test_job_history_is_bounded_and_decision_operation_metrics_are_projected(self):
        points = [{'step': n, 'elapsed_seconds': n / 2, 'metrics': {'loss': n / 1000}}
                  for n in range(1005)]
        data = self.project(jobs=[{'job_id': 'j', 'metric_history': points}],
            decisions=[{'kind': 'operation', 'operation': 'query_experiments',
                        'payload': 'TOP_SECRET', 'metrics': {'loss': 0.25, 'api_key': 123}}])
        self.assertEqual(data['jobs'][0]['metric_history'], points[-1000:])
        self.assertEqual(data['decisions'], [{'kind': 'operation', 'operation': 'query_experiments',
                                            'metrics': {'loss': 0.25}}])
        for value in (None, {}, False, 'broken'):
            self.assertEqual(self.project(jobs=[{'metric_history': value}])['jobs'], [{'metric_history': []}])
        self.assertNotIn('TOP_SECRET', json.dumps(data))

    def test_explanations_and_recent_collections_are_bounded(self):
        explanation = '公开说明' * 400
        rows = [{'decision_id': str(i), 'reason': explanation} for i in range(monitor.MAX_ROWS + 5)]
        data = self.project(current={'strategy': explanation}, decisions=rows,
                            round_results=[{'round': i} for i in range(monitor.MAX_ROWS + 5)],
                            events=[{'details': {'reason': explanation}}])
        self.assertEqual(len(data['decisions']), monitor.MAX_ROWS)
        self.assertEqual(data['decisions'][0]['decision_id'], '5')
        self.assertEqual(len(data['round_results']), monitor.MAX_ROWS)
        self.assertEqual(data['round_results'][-1]['round'], monitor.MAX_ROWS + 4)
        for text in (data['current']['strategy'], data['decisions'][0]['reason'],
                     data['events'][0]['details']['reason']):
            self.assertEqual(text, explanation[:1000])

    def test_new_metrics_keep_existing_depth_and_entry_limits(self):
        metrics = {'m' + str(i): i for i in range(120)}
        data = self.project(current={'metrics': metrics})
        self.assertEqual(len(data['current']['metrics']), 100)
        deep = {'a': {'b': {'c': {'d': {'e': {'hidden': 1}}}}}}
        data = self.project(round_results=[{'metrics': deep}])
        self.assertNotIn('hidden', json.dumps(data))

    def test_file_store_projects_extended_schema_without_serving_raw_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run-1'
            run.mkdir()
            (run / 'monitor.json').write_text(json.dumps({
                'schema_version': 1, 'workflow': {'round': 2}, 'current': {'mode': 'train'},
                'decisions': [{'reason': 'Public reason', 'response': 'TOP_SECRET'}],
                'round_results': [{'round': 1, 'applied': False, 'metrics': {'delta_nll': -0.1}}],
            }))
            store = monitor.RunStore(root)
            data = store.read('run-1')
            self.assertEqual(data['workflow'], {'round': 2})
            self.assertEqual(data['decisions'], [{'reason': 'Public reason'}])
            self.assertNotIn('TOP_SECRET', json.dumps(data))
            # Run-list summaries retain their existing compact contract.
            self.assertEqual(store.list()[0]['counts'], {'decisions': 1, 'jobs': 0, 'candidates': 0, 'events': 0})


class DashboardProjectionTests(unittest.TestCase):
    def panel(self, **fields):
        return {'id': 'quality', 'title': 'Quality', 'type': 'line', 'metrics': ['accuracy'],
                'x': 'evaluation', 'group_by': 'attempt', **fields}

    def project(self, **fields):
        return monitor.project_snapshot({'schema_version': 1, **fields}, 'run-1')

    def test_dashboard_roundtrip_is_whitelisted_independent_and_does_not_mutate_input(self):
        fields = {
            'metric_definitions': {'accuracy': {'label': 'Accuracy', 'unit': 'percent', 'direction': 'higher'}},
            'metric_points': [{'metric': 'accuracy', 'value': .75, 'step': 0, 'elapsed_seconds': 0,
                               'timestamp': '2026-11-02T00:00:00Z', 'task_id': 't',
                               'decision_id': 'd', 'attempt_id': 'a'},
                              {'metric': 'unknown_score', 'value': 85}],
            'dashboard': {'schema_version': 1, 'panels': [self.panel(layout={'w': 6, 'h': 360}),
                                                        self.panel(id='legacy')]},
        }
        original = deepcopy(fields)
        projected = self.project(**fields)
        for key, value in fields.items():
            self.assertEqual(projected[key], value)
        roundtrip = monitor.project_snapshot(json.loads(json.dumps(projected, allow_nan=False)), 'run-1')
        for key in fields:
            self.assertEqual(roundtrip[key], projected[key])
        projected['dashboard']['panels'][0]['metrics'].append('loss')
        projected['dashboard']['panels'][0]['layout']['w'] = 12
        projected['metric_definitions']['accuracy']['label'] = 'Changed'
        projected['metric_points'][0]['value'] = 0
        self.assertEqual(fields, original)

    def test_forbidden_payloads_are_stripped_at_every_new_protocol_level(self):
        private = {'payload': {'secret': 'PRIVATE'}, 'config': ['PRIVATE'], 'credentials': 'PRIVATE'}
        projected = self.project(
            metric_definitions={'loss': {'label': 'Loss', 'unit': 'number', 'direction': 'lower', **private}},
            metric_points=[{'metric': 'loss', 'value': .5, **private},
                           {'metric': 'loss', 'value': {'payload': 'PRIVATE'}},
                           {'metric': 'loss', 'value': 1, 'task_id': {'payload': 'PRIVATE'}}],
            dashboard={'schema_version': 1, 'panels': [
                self.panel(layout={'w': 6, 'h': 360, **private}, **private)], **private}, **private)
        self.assertEqual(projected['metric_definitions'], {
            'loss': {'label': 'Loss', 'unit': 'number', 'direction': 'lower'}})
        self.assertEqual(projected['metric_points'], [{'metric': 'loss', 'value': .5}])
        self.assertEqual(projected['dashboard'], {
            'schema_version': 1, 'panels': [self.panel(layout={'w': 6, 'h': 360})]})
        self.assertNotIn('PRIVATE', json.dumps(projected, allow_nan=False))

    def test_panel_layout_accepts_inclusive_bounds_and_preserves_absence(self):
        panels = [self.panel(id=f'{w}-{h}', layout={'w': w, 'h': h})
                  for w in (3, 12) for h in (240, 720)] + [self.panel()]
        dashboard = {'schema_version': 1, 'panels': panels}
        projected = self.project(dashboard=dashboard)['dashboard']
        self.assertEqual(projected, dashboard)
        self.assertNotIn('layout', projected['panels'][-1])
        self.assertEqual(monitor.dashboard_only(json.loads(json.dumps(projected))), dashboard)

    def test_invalid_layout_is_omitted_without_losing_panel_or_valid_siblings(self):
        invalid = [None, True, False, 6, '6', [], {}, {'w': 6}, {'h': 360}]
        for key in ('w', 'h'):
            invalid += [{**{'w': 6, 'h': 360}, key: value} for value in
                        (None, True, False, '6', [], {}, 6.0, float('nan'), float('inf'), 10 ** 1000)]
        invalid += [{'w': 2, 'h': 360}, {'w': 13, 'h': 360},
                    {'w': 6, 'h': 239}, {'w': 6, 'h': 721}, {'w': 6, 'h': 360.0}]
        sibling = self.panel(id='sibling', layout={'w': 12, 'h': 720})
        for layout in invalid:
            with self.subTest(layout=layout):
                dashboard = {'schema_version': 1, 'panels': [self.panel(layout=layout), sibling]}
                original = deepcopy(dashboard)
                self.assertEqual(self.project(dashboard=dashboard)['dashboard'], {
                    'schema_version': 1, 'panels': [self.panel(), sibling]})
                self.assertEqual(dashboard, original)

    def test_collections_are_bounded_and_points_keep_recent_observations(self):
        definitions = {f'm{i}': {'label': str(i), 'unit': 'number', 'direction': 'neutral'} for i in range(105)}
        points = [{'metric': 'loss', 'value': i} for i in range(10005)]
        panels = [self.panel(id=f'p{i}', metrics=list(definitions)[:15]) for i in range(30)]
        projected = self.project(metric_definitions=definitions, metric_points=points,
                                 dashboard={'schema_version': 1, 'panels': panels})
        self.assertEqual(list(projected['metric_definitions']), list(definitions)[:100])
        self.assertEqual(projected['metric_points'], points[-10000:])
        self.assertEqual(len(projected['dashboard']['panels']), 24)
        self.assertEqual(projected['dashboard']['panels'][-1]['id'], 'p23')
        self.assertEqual(projected['dashboard']['panels'][0]['metrics'], list(definitions)[:12])

    def test_panel_y_range_roundtrips_optional_bounds_and_normalizes_auto(self):
        ranges = [{}, {'min': 0}, {'max': 0}, {'min': .7, 'max': .8},
                  {'min': -.2, 'max': 1.2}, {'min': -1e308}, {'max': 1e308},
                  {'min': 0, 'max': 5e-324}, {'min': -10 ** 307, 'max': 10 ** 307}]
        panels = [self.panel(id=str(index), y_range=bounds, layout={'w': 6, 'h': 360})
                  for index, bounds in enumerate(ranges)] + [self.panel(id='legacy')]
        dashboard = {'schema_version': 1, 'panels': panels}
        original = deepcopy(dashboard)
        expected = deepcopy(dashboard)
        del expected['panels'][0]['y_range']
        projected = self.project(dashboard=dashboard)['dashboard']
        self.assertEqual(projected, expected)
        self.assertEqual(monitor.dashboard_only(json.loads(json.dumps(projected, allow_nan=False))), expected)
        projected['panels'][1]['y_range']['min'] = -1
        self.assertEqual(dashboard, original)

    def test_panel_y_range_filters_unknown_keys_without_rescaling_percent_bounds(self):
        private = {'payload': {'secret': 'PRIVATE'}, 'padding': True}
        panels = [self.panel(y_range={'min': .7, 'max': .8, **private}),
                  self.panel(id='viewport', y_range={'min': -.2, 'max': 1.2}),
                  self.panel(id='auto', y_range=private)]
        definitions = {'accuracy': {'label': 'Accuracy', 'unit': 'percent', 'direction': 'higher'}}
        projected = self.project(dashboard={'schema_version': 1, 'panels': panels},
                                 metric_definitions=definitions,
                                 metric_points=[{'metric': 'accuracy', 'value': value}
                                                for value in (-.2, 0, .75, 1, 1.2)])
        self.assertEqual(projected['dashboard']['panels'], [
            self.panel(y_range={'min': .7, 'max': .8}),
            self.panel(id='viewport', y_range={'min': -.2, 'max': 1.2}), self.panel(id='auto')])
        self.assertEqual(projected['metric_points'], [{'metric': 'accuracy', 'value': value}
                                                     for value in (0, .75, 1)])
        self.assertNotIn('PRIVATE', json.dumps(projected, allow_nan=False))

    def test_invalid_y_range_omits_whole_range_and_keeps_panel_and_sibling(self):
        invalid = [None, True, False, 0, 'auto', [], [0, 1],
                   {'min': 1, 'max': 1}, {'min': 2, 'max': 1},
                   {'min': -1e308, 'max': 1e308},
                   {'min': -10 ** 308, 'max': 10 ** 308}]
        for key in ('min', 'max'):
            invalid += [{**{'min': 0, 'max': 1}, key: value} for value in
                        (None, True, False, '0', [], {}, float('nan'), float('inf'),
                         -float('inf'), 10 ** 1000)]
            invalid += [{key: value} for value in (None, True, float('nan'), float('inf'))]
        sibling = self.panel(id='sibling', y_range={'max': .8})
        for bounds in invalid:
            with self.subTest(y_range=bounds):
                dashboard = {'schema_version': 1, 'panels': [
                    self.panel(y_range=bounds, layout={'w': 6, 'h': 360}), sibling]}
                original = deepcopy(dashboard)
                self.assertEqual(self.project(dashboard=dashboard)['dashboard'], {
                    'schema_version': 1, 'panels': [self.panel(layout={'w': 6, 'h': 360}), sibling]})
                self.assertEqual(dashboard, original)

    def test_duplicate_panel_ids_keep_first_valid_and_metrics_keep_first_occurrence(self):
        first = self.panel(metrics=['accuracy', 'loss', 'accuracy', 'loss'])
        second = self.panel(id='other', metrics=['loss'] * 11 + ['accuracy', 'beyond-limit'])
        dashboard = {'schema_version': 1, 'panels': [
            self.panel(title=None), self.panel(metrics=[None]), first,
            self.panel(title='Duplicate', metrics=['ignored']), second]}
        original = deepcopy(dashboard)
        expected = {'schema_version': 1, 'panels': [
            {**first, 'metrics': ['accuracy', 'loss']}, {**second, 'metrics': ['loss', 'accuracy']}]}
        self.assertEqual(self.project(dashboard=dashboard)['dashboard'], expected)
        self.assertEqual(monitor.dashboard_only(expected), expected)
        self.assertEqual(dashboard, original)

    def test_invalid_numeric_and_coordinate_fields_reject_whole_points(self):
        invalid = [None, True, False, '0.5', [], {}, float('nan'), float('inf'), -float('inf'), 10 ** 1000]
        base = {'metric': 'score', 'value': .5}
        rows = [None, {}, {'value': 1}, {'metric': 'score'}]
        rows += [{**base, 'value': value} for value in invalid]
        rows += [{**base, 'step': value} for value in invalid + [-1, 1.5, 0.0]]
        rows += [{**base, 'elapsed_seconds': value} for value in invalid + [-.1]]
        for key in ('metric', 'timestamp', 'task_id', 'decision_id', 'attempt_id'):
            rows += [{**base, key: value} for value in (None, 0, {}, [], '', ' ', 'x' * 201)]
        valid = [base, {**base, 'value': -42, 'step': 0, 'elapsed_seconds': 0.0}]
        self.assertEqual(self.project(metric_points=rows + valid)['metric_points'], valid)

    def test_percent_is_an_explicit_ratio_and_unknown_scores_stay_unscaled(self):
        definitions = {'accuracy': {'label': 'Accuracy', 'unit': 'percent', 'direction': 'higher'}}
        points = [{'metric': name, 'value': value} for name in ('accuracy', 'score')
                  for value in (-1, 0, .75, 1, 75)]
        projected = self.project(metric_definitions=definitions, metric_points=points)
        self.assertEqual(projected['metric_points'], [point for point in points
                         if point['metric'] == 'score' or 0 <= point['value'] <= 1])

    def test_optional_old_schema_and_malformed_containers(self):
        for key in ('dashboard', 'metric_definitions', 'metric_points'):
            self.assertNotIn(key, self.project())
        for value in (None, False, 1, 'broken', {}):
            with self.subTest(value=value):
                projected = self.project(dashboard=value, metric_definitions=value, metric_points=value)
                self.assertNotIn('dashboard', projected)
                self.assertEqual(projected['metric_definitions'], {})
                self.assertEqual(projected['metric_points'], [])
        for version in (None, True, 1.0, 2):
            self.assertIsNone(monitor.dashboard_only({'schema_version': version, 'panels': []}))
        self.assertEqual(monitor.dashboard_only({'schema_version': 1, 'panels': []}),
                         {'schema_version': 1, 'panels': []})

    def test_invalid_definitions_and_panels_are_omitted_without_losing_valid_siblings(self):
        good = {'label': 'Loss', 'unit': 'number', 'direction': 'lower'}
        definitions = {'loss': good, '': good, 'x' * 101: good, 'wrong_type': [],
                       'missing': {'unit': 'number', 'direction': 'lower'}}
        for key, values in {'label': (None, [], '', 'x' * 201),
                            'unit': (None, [], 'percentage'), 'direction': (None, {}, 'up')}.items():
            for index, value in enumerate(values):
                definitions[f'{key}{index}'] = {**good, key: value}
        panels = [None, {}, self.panel()]
        for key, values in {'id': ('', None, 'x' * 201), 'title': ([], ''),
                            'type': ('bar', []), 'x': ('index', {}), 'group_by': ('job', []),
                            'metrics': ('loss', None, [None], ['x' * 101])}.items():
            panels += [self.panel(**{key: value}) for value in values]
        projected = self.project(metric_definitions=definitions, dashboard={'schema_version': 1, 'panels': panels})
        self.assertEqual(projected['metric_definitions'], {'loss': good})
        self.assertEqual(projected['dashboard']['panels'], [self.panel()])

    def test_all_supported_panel_modes_and_definition_units_survive(self):
        panels = [self.panel(id=str(i), type=kind, x=x, group_by=group)
                  for i, (kind, x, group) in enumerate(zip(
                      ('line', 'scalar', 'table'), ('step', 'elapsed_seconds', 'evaluation'),
                      ('task', 'attempt', 'none')))]
        definitions = {unit: {'label': unit, 'unit': unit, 'direction': direction}
                       for unit, direction in zip(('number', 'percent', 'seconds', 'tokens_per_second'),
                                                  ('neutral', 'higher', 'lower', 'higher'))}
        self.assertEqual(monitor.metric_definitions_only(definitions), definitions)
        self.assertEqual(monitor.dashboard_only({'schema_version': 1, 'panels': panels})['panels'], panels)

    def test_host_metrics_protocol_works_without_rsisys_installed(self):
        code = '''
import importlib.abc
import json
import sys
sys.path.insert(0, sys.argv[1])
class BlockRSISys(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'rsisys':
            raise ImportError('rsisys is not installed')
sys.meta_path.insert(0, BlockRSISys())
from rsimonitor.server import project_snapshot
snapshot = project_snapshot({'schema_version': 1, 'metric_points': [{'metric': 'score', 'value': 75}],
                             'dashboard': {'schema_version': 1, 'panels': []}}, 'run')
assert snapshot['metric_points'] == [{'metric': 'score', 'value': 75}]
json.dumps(snapshot, allow_nan=False)
'''
        result = subprocess.run([sys.executable, '-I', '-c', code,
                                 str(REPO_ROOT / 'src')],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs"
        self.root.mkdir()
        self.server = monitor.MonitorServer(("127.0.0.1", 0), self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def write(self, run_id="run-1", **fields):
        directory = self.root / run_id
        directory.mkdir(exist_ok=True)
        data = dict(schema_version=1, run_id=run_id, status="running", objective="A real test run", **fields)
        (directory / "monitor.json").write_text(json.dumps(data), encoding="utf-8")
        return directory

    def get(self, path, method="GET", headers=None):
        request = Request(self.url + path, method=method, headers=headers or {})
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read()

    def test_empty_and_missing_root(self):
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])
        self.assertEqual(json.loads(self.get("/api/projects")[2])["projects"], [])
        self.root.rmdir()
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])
        self.assertEqual(json.loads(self.get("/api/projects")[2])["projects"], [])
        self.assertFalse(self.root.exists())

    def test_dashboard_api_rejects_nonfinite_points_without_rewriting_snapshot(self):
        directory = self.write()
        path = directory / 'monitor.json'
        contents = ('{"schema_version":1,"metric_definitions":{"accuracy":{"label":"Accuracy",'
                    '"unit":"percent","direction":"higher"}},"metric_points":['
                    '{"metric":"loss","value":NaN},{"metric":"loss","value":Infinity},'
                    '{"metric":"loss","value":-Infinity},{"metric":"loss","value":1e999},'
                    '{"metric":"loss","value":1,"step":NaN},'
                    '{"metric":"loss","value":1,"elapsed_seconds":Infinity},'
                    '{"metric":"accuracy","value":75},{"metric":"accuracy","value":0.75},'
                    '{"metric":"score","value":75,"payload":"PRIVATE"}],'
                    '"dashboard":{"schema_version":1,"panels":[]}}')
        path.write_text(contents)
        status, _, body = self.get('/api/runs/run-1')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data['metric_points'], [{'metric': 'accuracy', 'value': .75},
                                                {'metric': 'score', 'value': 75}])
        self.assertEqual(data['dashboard'], {'schema_version': 1, 'panels': []})
        json.dumps(data, allow_nan=False)
        self.assertNotIn('PRIVATE', body.decode())
        self.assertEqual(path.read_text(), contents)
        summary = json.loads(self.get('/api/runs')[2])['runs'][0]
        for key in ('dashboard', 'metric_definitions', 'metric_points'):
            self.assertNotIn(key, summary)

    def test_projects_api_groups_explicit_membership_and_preserves_experiment_names_and_provenance(self):
        legacy = {'project_id': 'shared', 'name': 'Research', 'group': 'qwen'}
        self.write('parent', experiment_name='Parent experiment', display_name='Lower priority', project=legacy)
        self.write('child', name='Child experiment', project=legacy, parent_run_id='parent', source_run_ids=['parent'])
        status, _, body = self.get('/api/projects')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data['schema_version'], 1)
        self.assertEqual([p['project_id'] for p in data['projects']], ['shared'])
        self.assertEqual(data['projects'][0]['name'], 'Research')
        rows = json.loads(self.get('/api/runs')[2])['runs']
        self.assertEqual(data['projects'][0]['runs'], rows)
        self.assertEqual([r['experiment_name'] for r in rows], ['Child experiment', 'Parent experiment'])
        for row in rows:
            self.assertEqual(row['project']['project_id'], 'shared')
            detail = json.loads(self.get('/api/runs/' + row['run_id'])[2])
            self.assertEqual(detail['project'], row['project'])
            self.assertEqual(detail['experiment_name'], row['experiment_name'])
        self.assertEqual(rows[0]['parent_run_id'], 'parent')
        self.assertEqual(rows[0]['source_run_ids'], ['parent'])
        (self.root / 'child' / 'monitor.json').write_text('{')
        projects = json.loads(self.get('/api/projects')[2])['projects']
        self.assertEqual([p['project_id'] for p in projects], ['shared', 'unassigned'])
        self.assertEqual(projects[1]['runs'][0]['run_id'], 'child')
        self.assertEqual(projects[1]['runs'][0]['status'], 'unavailable')
        self.assertEqual(projects[1]['runs'][0]['experiment_name'], 'child')
        self.write('child', display_name='Recovered', project=legacy)
        recovered = json.loads(self.get('/api/projects')[2])['projects']
        self.assertEqual([p['project_id'] for p in recovered], ['shared'])
        self.assertEqual(recovered[0]['runs'][0]['experiment_name'], 'Recovered')

    def test_unrelated_preview_directory_is_not_a_run(self):
        (self.root / "monitor-preview").mkdir()
        (self.root / "monitor-preview" / "wide.jpg").write_bytes(b"preview")
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])

    def test_worker_flat_metrics_preserve_values_and_direction(self):
        metrics = {"base_nll": 1.942, "candidate_nll": 1.781,
                   "delta_nll": -0.161, "eval_samples": 128}
        self.write(candidates=[{"candidate_id": "c", "metrics": metrics}])
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["candidates"][0]["metrics"], metrics)

    def test_huge_numeric_value_does_not_crash(self):
        self.write(candidates=[{"metrics": {"score": 10 ** 1000}}])
        self.assertEqual(self.get("/api/runs/run-1")[0], 200)

    def test_summary_and_snapshot(self):
        self.write(jobs=[{"job_id": "j-1", "kind": "training", "status": "running"}],
                   loops={"decision_making": {"status": "running", "summary": "Planning"}},
                   models={"actor": {"model_id": "actor", "revision": "r2"}})
        summary = json.loads(self.get("/api/runs")[2])["runs"][0]
        self.assertEqual(summary["counts"], {"decisions": 0, "jobs": 1, "candidates": 0, "events": 0})
        self.assertNotIn("jobs", summary)
        status, headers, body = self.get("/api/runs/run-1")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["loops"]["decision_making"]["summary"], "Planning")
        self.assertEqual(data["models"]["actor"]["revision"], "r2")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))

    def test_optional_fields_and_wrong_types(self):
        self.write(loops=None, models=[], jobs="wrong", candidates=[None, 2, {}], events=None)
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["jobs"], [])
        self.assertEqual(data["loops"], {})
        self.assertEqual(data["candidates"], [{"metrics": {}}])

    def test_only_monitor_is_read_and_private_payloads_are_dropped(self):
        directory = self.write(config={"api_key": "TOP_SECRET"}, training_samples=["TOP_SECRET"],
            jobs=[{"job_id": "j", "config": {"password": "TOP_SECRET"}}],
            candidates=[{"candidate_id": "c", "metrics": {"score": {"baseline": .4, "candidate": .6},
                        "api_key": "TOP_SECRET", "samples": {"private": 42}, "config": {"private": 42}}}],
            events=[{"kind": "trained", "details": {"job_id": "j", "credentials": "TOP_SECRET", "samples": ["TOP_SECRET"], "config": "TOP_SECRET"}}])
        (directory / "config.json").write_text('"TOP_SECRET"')
        status, _, body = self.get("/api/runs/run-1")
        self.assertEqual(status, 200)
        self.assertNotIn(b"TOP_SECRET", body)
        self.assertNotIn(b"private", body)
        self.assertEqual(json.loads(body)["events"][0]["details"], {"job_id": "j"})
        self.assertEqual(self.get("/api/runs/run-1/config.json")[0], 404)
        self.assertEqual(self.get("/runs/run-1/config.json")[0], 404)

    def test_paths_cannot_escape(self):
        self.write()
        for path in ["/api/runs/..", "/api/runs/%2e%2e", "/api/runs/%2e%2e%2fsecret",
                     "/api/runs/run-1%2fmonitor.json", "/api/runs/run-1%5cmonitor.json",
                     "/api/runs/%252e%252e", "/api/runs/%00", "/../server.py", "/server.py", "/README.md"]:
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_symlink_run_and_snapshot_are_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "monitor.json").write_text('{"schema_version":1,"objective":"TOP_SECRET"}')
        (self.root / "linked-run").symlink_to(outside, target_is_directory=True)
        local = self.root / "linked-file"
        local.mkdir()
        (local / "monitor.json").symlink_to(outside / "monitor.json")
        self.assertEqual(self.get("/api/runs/linked-run")[0], 404)
        self.assertEqual(self.get("/api/runs/linked-file")[0], 404)
        data = json.loads(self.get("/api/runs")[2])
        self.assertNotIn("linked-run", [r["run_id"] for r in data["runs"]])
        self.assertNotIn("TOP_SECRET", json.dumps(data))

    def test_directory_swap_to_symlink_is_rejected(self):
        directory = self.write()
        directory.rename(self.root / "old-run")
        (self.root / "run-1").symlink_to(self.root / "old-run", target_is_directory=True)
        self.assertEqual(self.get("/api/runs/run-1")[0], 404)

    def test_non_regular_snapshot_rejected_without_blocking(self):
        import os
        directory = self.root / "fifo-run"
        directory.mkdir()
        os.mkfifo(directory / "monitor.json")
        self.assertEqual(self.get("/api/runs/fifo-run")[0], 404)

    def test_partial_json_is_retryable_and_recovers(self):
        directory = self.write()
        (directory / "monitor.json").write_text('{"schema_version":')
        status, headers, _ = self.get("/api/runs/run-1")
        self.assertEqual(status, 503)
        self.assertEqual(headers["Retry-After"], "2")
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"][0]["status"], "unavailable")
        self.write()
        self.assertEqual(self.get("/api/runs/run-1")[0], 200)

    def test_snapshot_missing_and_schema_mismatch(self):
        directory = self.root / "run-1"
        directory.mkdir()
        self.assertEqual(self.get("/api/runs/run-1")[0], 404)
        for raw in [[], {"schema_version": 2}, {"schema_version": True}, {}]:
            (directory / "monitor.json").write_text(json.dumps(raw))
            self.assertEqual(self.get("/api/runs/run-1")[0], 422)

    def test_large_snapshot_rejected(self):
        directory = self.write()
        (directory / "monitor.json").write_bytes(b" " * (monitor.MAX_BYTES + 1))
        self.assertEqual(self.get("/api/runs/run-1")[0], 413)

    def test_no_write_methods(self):
        directory = self.write()
        original = (directory / "monitor.json").read_bytes()
        for method in ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"]:
            self.assertEqual(self.get("/api/runs/run-1", method=method)[0], 405)
        self.assertEqual((directory / "monitor.json").read_bytes(), original)

    def test_host_and_cross_site_guard(self):
        self.assertEqual(self.get("/api/runs", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.get("/api/runs", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)

    def test_static_allowlist_head_and_no_development_files(self):
        for path in ["/", "/style.css", "/i18n.js", "/decisions.js", "/app.js"]:
            self.assertEqual(self.get(path)[0], 200)
        self.assertEqual(self.get("/", method="HEAD")[2], b"")
        for path in ("/demo.json", "/project.test.js", "/decisions.test.js", "/i18n.test.js", "/attempts.test.js"):
            self.assertEqual(self.get(path)[0], 404)
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])

    def test_run_directory_name_is_authoritative(self):
        directory = self.write()
        (directory / "monitor.json").write_text('{"schema_version":1,"run_id":"../elsewhere"}')
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["run_id"], "run-1")

    def test_non_finite_numbers_do_not_break_json(self):
        directory = self.write()
        (directory / "monitor.json").write_text('{"schema_version":1,"candidates":[{"metrics":{"score":NaN,"loss":Infinity}}]}')
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["candidates"][0]["metrics"], {})


if __name__ == "__main__":
    unittest.main()
