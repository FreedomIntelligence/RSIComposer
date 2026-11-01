import json
from pathlib import Path
import tempfile
import unittest
from web.server import RunStore, project_snapshot


class ProjectTests(unittest.TestCase):
    def test_explicit_membership_and_unassigned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ident in ('a', 'b'):
                (root / ident).mkdir()
                (root / ident / 'monitor.json').write_text(json.dumps({'schema_version': 1, 'objective': 'same'}))
            config = root / 'projects.json'
            config.write_text(json.dumps({'projects': [{'project_id': 'p', 'system_id': 'sys', 'name': 'Project', 'runs': ['a']}]}))
            store = RunStore(root, config)
            self.assertEqual(store.read('a')['project']['system_id'], 'sys')
            self.assertEqual(store.read('b')['project']['project_id'], 'unassigned')
            self.assertEqual({p['project_id']: len(p['runs']) for p in store.projects()}, {'p': 1, 'unassigned': 1})

    def test_duplicate_membership_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'projects.json'
            config.write_text(json.dumps({'projects': [{'project_id': 'p', 'runs': ['a']}, {'project_id': 'q', 'runs': ['a']}]}))
            with self.assertRaises(ValueError):
                RunStore(directory, config)

    def test_provenance_is_explicit_bounded_and_private_data_excluded(self):
        result = project_snapshot({'schema_version': 1, 'project': {'project_id': 'p', 'name': 'Project', 'secret': 'hidden'}, 'decisions': [
            {'decision_id': 'a'}, {'decision_id': 'b', 'parent_decision_id': 'a', 'source_decision_ids': ['a', {}, 'c'], 'private': 'hidden'}]}, 'run')
        self.assertNotIn('secret', result['project'])
        self.assertNotIn('parent_decision_id', result['decisions'][0])
        self.assertEqual(result['decisions'][1]['source_decision_ids'], ['a', 'c'])
        self.assertNotIn('private', result['decisions'][1])

    def test_snapshot_project_metadata_works_without_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run').mkdir()
            (root / 'run' / 'monitor.json').write_text(json.dumps({'schema_version': 1,
                'project': {'project_id': 'system-a', 'name': 'A', 'system_id': 'a'}}))
            projects = RunStore(root).projects()
            self.assertEqual(projects[0]['project_id'], 'system-a')
            self.assertEqual(projects[0]['runs'][0]['run_id'], 'run')

    def test_unreadable_snapshot_keeps_explicit_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run').mkdir()
            (root / 'run' / 'monitor.json').write_text('{')
            config = root / 'registry.json'
            config.write_text(json.dumps({'projects': [{'project_id': 'p', 'runs': ['run']}]}))
            project = RunStore(root, config).projects()[0]
            self.assertEqual(project['runs'][0]['status'], 'unavailable')
            self.assertEqual(project['runs'][0]['project']['project_id'], 'p')
