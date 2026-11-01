"""A monitor project is exactly one RSI startup, identified by its run ID."""
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

    def test_shared_legacy_groups_never_merge_runs_or_modify_files(self):
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
        self.assertEqual([p['project_id'] for p in projects], ['a', 'b', 'c'])
        self.assertEqual([p['name'] for p in projects], ['Shared name · a', 'Shared name · b', 'c'])
        for project in projects:
            self.assertEqual(len(project['runs']), 1)
            run = project['runs'][0]
            self.assertEqual(project['project_id'], run['run_id'])
            self.assertEqual(run['project'], self.store.read(run['run_id'])['project'])
            self.assertEqual(run['project'], {k: v for k, v in project.items() if k != 'runs'})
            self.assertNotIn('group', run['project'])
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, original)

    def test_explicit_names_are_bounded_and_directory_identity_wins(self):
        cases = [({}, 'run'), ({'display_name': '  Explicit  '}, 'Explicit · run'),
            ({'name': 'Run name'}, 'Run name · run'), ({'project_name': 'New project'}, 'New project · run'),
            ({'project': {'name': 'Snapshot name'}}, 'Snapshot name · run'),
            ({'project': {'project_id': 'run', 'name': '医疗 RSI · run'}}, '医疗 RSI · run'),
            ({'project': {'project_id': '../invalid', 'name': 'Old name'}}, 'Old name · run'),
            ({'display_name': 'Primary', 'name': 'Secondary', 'project': {'name': 'Legacy'}}, 'Primary · run'),
            ({'display_name': [], 'name': ' ', 'project_name': 5, 'project': {'name': None}}, 'run'),
            ({'name': 'run'}, 'run')]
        for extra, expected in cases:
            with self.subTest(extra=extra):
                raw = {'schema_version': 1, 'run_id': 'untrusted', 'objective': 'Never a name', **extra}
                original = deepcopy(raw)
                project = project_snapshot(raw, 'run')['project']
                self.assertEqual(project, {'project_id': 'run', 'name': expected})
                self.assertEqual(raw, original)
        for project in (None, [], 'group', 42):
            self.assertEqual(project_snapshot({'schema_version': 1, 'project': project}, 'run')['project'],
                             {'project_id': 'run', 'name': 'run'})
        name = project_snapshot({'schema_version': 1, 'display_name': 'x' * 2000}, 'run')['project']['name']
        self.assertEqual(len(name), 1000)
        self.assertTrue(name.endswith(' · run'))

    def test_unavailable_snapshots_remain_separate_projects_and_recover(self):
        for ident, contents in [('broken', '{'), ('unsupported', '{"schema_version": 2}'),
                                ('oversized', ' ' * (MAX_BYTES + 1))]:
            self.write(ident).write_text(contents)
        (self.root / 'no-snapshot').mkdir()
        projects = self.store.projects()
        self.assertEqual([p['project_id'] for p in projects], ['broken', 'oversized', 'unsupported'])
        for project in projects:
            self.assertEqual(project['name'], project['project_id'])
            self.assertEqual(len(project['runs']), 1)
            self.assertEqual(project['runs'][0]['status'], 'unavailable')
            self.assertIn('error', project['runs'][0])
            self.assertEqual(project['runs'][0]['project']['project_id'], project['project_id'])
        self.write('broken', project={'name': 'Recovered'})
        self.assertEqual(self.store.projects()[0]['name'], 'Recovered · broken')

    def test_missing_root_has_no_projects_and_is_not_created(self):
        root = self.root / 'missing'
        self.assertEqual(RunStore(root).projects(), [])
        self.assertFalse(root.exists())

    def test_provenance_is_explicit_bounded_and_private_data_excluded(self):
        result = project_snapshot({'schema_version': 1, 'project': {
            'project_id': 'p', 'name': 'Project', 'secret': 'hidden'}, 'decisions': [
            {'decision_id': 'a'}, {'decision_id': 'b', 'parent_decision_id': 'a',
             'source_decision_ids': ['a', {}, 'c'], 'private': 'hidden'}]}, 'run')
        self.assertEqual(result['project'], {'project_id': 'run', 'name': 'Project · run'})
        self.assertNotIn('parent_decision_id', result['decisions'][0])
        self.assertEqual(result['decisions'][1]['source_decision_ids'], ['a', 'c'])
        self.assertNotIn('private', result['decisions'][1])

    def test_cross_project_references_preserve_provenance_without_merging(self):
        self.write('a')
        self.write('b', parent_run_id='a', source_run_ids=['a', 'absent', {}])
        projects = self.store.projects()
        self.assertEqual([p['project_id'] for p in projects], ['a', 'b'])
        rows = {r['run_id']: r for r in self.store.list()}
        self.assertNotIn('parent_run_id', rows['a'])
        self.assertEqual(rows['b']['parent_run_id'], 'a')
        self.assertEqual(rows['b']['source_run_ids'], ['a', 'absent'])
        self.assertEqual(self.store.read('b')['parent_run_id'], 'a')
        self.assertEqual(projects[1]['runs'][0]['parent_run_id'], 'a')


if __name__ == '__main__':
    unittest.main()
