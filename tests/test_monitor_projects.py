"""Snapshot-declared projects group experiments without rewriting history."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from web.server import MAX_BYTES, RunStore, project_snapshot


class ProjectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = RunStore(self.root)

    def write(self, ident, **extra):
        directory = self.root / ident
        directory.mkdir(exist_ok=True)
        path = directory / 'monitor.json'
        path.write_text(json.dumps({'schema_version': 1, **extra}))
        return path

    def test_shared_legacy_project_groups_runs_without_modifying_files(self):
        paths = [self.write(ident, run_id='wrong', objective='same', project={
            'project_id': 'qwen', 'name': 'Shared name', 'group': 'medical',
            'system_id': 'shared-system'}) for ident in ('a', 'b')]
        paths.append(self.write('c'))
        # A leftover registry is unrelated data, never a membership source.
        registry = self.root / 'projects.json'
        registry.write_text(json.dumps({'projects': [{'project_id': 'qwen', 'runs': ['a', 'b', 'c']}]}))
        paths.append(registry)
        original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        projects = self.store.projects()
        self.assertEqual([p['project_id'] for p in projects], ['qwen', 'unassigned'])
        self.assertEqual([p['name'] for p in projects], ['Shared name', '未归属'])
        self.assertEqual([[r['run_id'] for r in p['runs']] for p in projects], [['a', 'b'], ['c']])
        for project in projects:
            for run in project['runs']:
                self.assertEqual(project['project_id'], run['project']['project_id'])
                self.assertEqual(run['experiment_name'], run['run_id'])
                self.assertEqual(run['project'], self.store.read(run['run_id'])['project'])
                self.assertNotIn('group', run['project'])
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, original)

    def test_experiment_names_are_separate_bounded_and_directory_identity_wins(self):
        cases = [({}, 'run'), ({'display_name': '  Explicit  '}, 'Explicit'),
            ({'experiment_name': '  Experiment  ', 'display_name': 'Display', 'name': 'Name'}, 'Experiment'),
            ({'experiment_name': [], 'display_name': 'Display'}, 'Display'),
            ({'experiment_name': ' ', 'display_name': ' ', 'name': 'Name'}, 'Name'),
            ({'name': 'Run name'}, 'Run name'), ({'project_name': 'New project'}, 'run'),
            ({'project': {'name': 'Snapshot name'}}, 'run'),
            ({'project': {'project_id': 'run', 'name': '医疗 RSI · run'}}, '医疗 RSI · run'),
            ({'project': {'project_id': 'large-project', 'name': 'Project name'}}, 'run'),
            ({'project': {'project_id': '../invalid', 'name': 'Old name'}}, 'run'),
            ({'display_name': 'Primary', 'name': 'Secondary', 'project': {'name': 'Legacy'}}, 'Primary'),
            ({'display_name': ' ', 'name': 'Secondary'}, 'Secondary'),
            ({'display_name': [], 'name': ' ', 'project_name': 5, 'project': {'name': None}}, 'run'),
            ({'name': 'run'}, 'run')]
        for extra, expected in cases:
            with self.subTest(extra=extra):
                raw = {'schema_version': 1, 'run_id': 'untrusted', 'objective': 'Never a name', **extra}
                original = deepcopy(raw)
                snapshot = project_snapshot(raw, 'run')
                self.assertEqual(snapshot['experiment_name'], expected)
                self.assertEqual(snapshot['run_id'], 'run')
                self.assertEqual(raw, original)
        for project in (None, [], 'group', 42):
            self.assertEqual(project_snapshot({'schema_version': 1, 'project': project}, 'run')['project'],
                             {'project_id': 'unassigned', 'name': '未归属'})
        snapshot = project_snapshot({'schema_version': 1, 'display_name': 'x' * 2000,
                                     'project': {'project_id': 'p', 'name': 'y' * 2000}}, 'run')
        self.assertEqual(snapshot['experiment_name'], 'x' * 1000)
        self.assertEqual(snapshot['project'], {'project_id': 'p', 'name': 'y' * 1000})
        self.assertEqual(project_snapshot({'schema_version': 1, 'experiment_name': 'z' * 2000}, 'run')
                         ['experiment_name'], 'z' * 1000)
        for project, expected in [({'project_id': 'p'}, 'Compatible project'),
                                  ({'project_id': 'p', 'name': 'Canonical'}, 'Canonical')]:
            snapshot = project_snapshot({'schema_version': 1, 'project': project,
                                         'project_name': 'Compatible project'}, 'run')
            self.assertEqual(snapshot['project']['name'], expected)
            self.assertEqual(snapshot['experiment_name'], 'run')

    def test_only_valid_explicit_project_ids_define_membership(self):
        for project_id in (None, '', ' ', '../invalid', '/absolute', 'with space', 'a' * 129, 7, True, [], {}):
            with self.subTest(project_id=project_id):
                raw = {'schema_version': 1, 'system_id': 'shared', 'project_id': 'guessed',
                       'project_name': 'Guessed', 'project': {'project_id': project_id,
                       'name': 'Ignore', 'system_id': 'same', 'description': 'Ignore'}}
                self.assertEqual(project_snapshot(raw, 'run')['project'],
                                 {'project_id': 'unassigned', 'name': '未归属'})
        for project_id in ('p', 'qwen-medical.v1', 'a' * 128, 'run'):
            self.assertEqual(project_snapshot({'schema_version': 1, 'project': {
                'project_id': project_id}}, 'run')['project'], {'project_id': project_id, 'name': project_id})

    def test_distinct_ids_with_same_names_and_systems_keep_separate_multi_run_projects(self):
        for ident, project_id in [('d', 'p2'), ('c', 'p1'), ('b', 'p2'), ('a', 'p1')]:
            self.write(ident, display_name='Same experiment', project={
                'project_id': project_id, 'name': 'Same project', 'system_id': 'same'})
        self.write('solo', project={'project_id': 'solo', 'name': 'Same project', 'system_id': 'same'})
        projects = self.store.projects()
        self.assertEqual([p['project_id'] for p in projects], ['p1', 'p2', 'solo'])
        self.assertEqual([[r['run_id'] for r in p['runs']] for p in projects], [['a', 'c'], ['b', 'd'], ['solo']])
        self.assertEqual([p['name'] for p in projects], ['Same project'] * 3)
        self.assertEqual([r['experiment_name'] for r in self.store.list()], ['Same experiment'] * 4 + ['Same project'])

    def test_reserved_unassigned_id_cannot_override_its_name_or_metadata(self):
        self.write('a', project={'project_id': 'unassigned', 'name': 'Misleading title',
                                'description': 'Misleading description', 'system_id': 'guessed'},
                   project_name='Another misleading title', experiment_name='Named experiment')
        self.write('b')
        projects = self.store.projects()
        self.assertEqual(len(projects), 1)
        self.assertEqual({k: v for k, v in projects[0].items() if k != 'runs'},
                         {'project_id': 'unassigned', 'name': '未归属'})
        self.assertEqual([r['run_id'] for r in projects[0]['runs']], ['a', 'b'])
        self.assertEqual(self.store.read('a')['project'], {'project_id': 'unassigned', 'name': '未归属'})
        self.assertEqual(self.store.read('a')['experiment_name'], 'Named experiment')

    def test_project_name_conflicts_use_first_named_run_by_directory_id(self):
        self.write('z', project={'project_id': 'p', 'name': 'Last name'})
        self.write('b', project={'project_id': 'p', 'name': 'First name'})
        self.write('a', project={'project_id': 'p', 'description': 'First metadata'})
        expected = self.store.projects()
        self.assertEqual(expected[0]['name'], 'First name')
        self.assertEqual(expected[0]['description'], 'First metadata')
        self.assertEqual([r['run_id'] for r in expected[0]['runs']], ['a', 'b', 'z'])
        self.assertEqual(self.store.projects(), expected)
        self.assertEqual(self.store.read('a')['project']['name'], 'p')
        self.assertEqual(self.store.read('z')['project']['name'], 'Last name')

    def test_unavailable_snapshots_are_unassigned_and_recover_into_declared_project(self):
        for ident, contents in [('broken', '{'), ('unsupported', '{"schema_version": 2}'),
                                ('oversized', ' ' * (MAX_BYTES + 1))]:
            self.write(ident).write_text(contents)
        (self.root / 'no-snapshot').mkdir()
        projects = self.store.projects()
        self.assertEqual([p['project_id'] for p in projects], ['unassigned'])
        self.assertEqual(projects[0]['name'], '未归属')
        self.assertEqual([r['run_id'] for r in projects[0]['runs']], ['broken', 'oversized', 'unsupported'])
        for run in projects[0]['runs']:
            self.assertEqual(run['status'], 'unavailable')
            self.assertEqual(run['experiment_name'], run['run_id'])
            self.assertIn('error', run)
            self.assertEqual(run['project']['project_id'], 'unassigned')
        self.write('broken', display_name='Recovered experiment', project={'project_id': 'p', 'name': 'Recovered'})
        recovered = self.store.projects()
        self.assertEqual([p['project_id'] for p in recovered], ['p', 'unassigned'])
        self.assertEqual(recovered[0]['name'], 'Recovered')
        self.assertEqual(recovered[0]['runs'][0]['experiment_name'], 'Recovered experiment')

    def test_missing_root_has_no_projects_and_is_not_created(self):
        root = self.root / 'missing'
        self.assertEqual(RunStore(root).projects(), [])
        self.assertFalse(root.exists())

    def test_provenance_is_explicit_bounded_and_private_data_excluded(self):
        result = project_snapshot({'schema_version': 1, 'project': {
            'project_id': 'p', 'name': 'Project', 'secret': 'hidden'}, 'decisions': [
            {'decision_id': 'a'}, {'decision_id': 'b', 'parent_decision_id': 'a',
             'source_decision_ids': ['a', {}, 'c'], 'private': 'hidden'}]}, 'run')
        self.assertEqual(result['project'], {'project_id': 'p', 'name': 'Project'})
        self.assertEqual(result['experiment_name'], 'run')
        self.assertNotIn('hidden', json.dumps(result))
        self.assertNotIn('parent_decision_id', result['decisions'][0])
        self.assertEqual(result['decisions'][1]['source_decision_ids'], ['a', 'c'])
        self.assertNotIn('private', result['decisions'][1])

    def test_attempt_membership_is_public_only_as_valid_paired_metadata(self):
        valid = {'decision_id': 'd', 'research_attempt_id': 'attempt-1',
                 'research_attempt_title': 'Compare output constraints'}
        invalid = [dict(valid, research_attempt_id='x' * 201),
                   dict(valid, research_attempt_id=42), dict(valid, research_attempt_title=None),
                   {'decision_id': 'legacy'}]
        rows = project_snapshot({'schema_version': 1, 'decisions': [valid, *invalid]}, 'run')['decisions']
        self.assertEqual(rows[0]['research_attempt_id'], 'attempt-1')
        self.assertEqual(rows[0]['research_attempt_title'], valid['research_attempt_title'])
        self.assertTrue(all('research_attempt_id' not in row for row in rows[1:]))

    def test_cross_project_references_preserve_provenance_without_merging(self):
        self.write('a', project={'project_id': 'p1'})
        self.write('b', project={'project_id': 'p2'}, parent_run_id='a', source_run_ids=['a', 'absent', {}])
        projects = self.store.projects()
        self.assertEqual([p['project_id'] for p in projects], ['p1', 'p2'])
        rows = {r['run_id']: r for r in self.store.list()}
        self.assertNotIn('parent_run_id', rows['a'])
        self.assertEqual(rows['b']['parent_run_id'], 'a')
        self.assertEqual(rows['b']['source_run_ids'], ['a', 'absent'])
        self.assertEqual(self.store.read('b')['parent_run_id'], 'a')
        self.assertEqual(projects[1]['runs'][0]['parent_run_id'], 'a')


if __name__ == '__main__':
    unittest.main()
