"""Workspace observations use existing capabilities and permission boundaries."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.action_space.capabilities import Capabilities
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools, execute_operation
from rsisys.loops.decision_making.harness.tools.workspace import WorkspaceTools


class WorkspaceToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.run = self.base / 'run'
        self.code = self.base / 'code'
        self.run.mkdir()
        self.code.mkdir()
        self.workspace = WorkspaceTools({'run': self.run, 'code': self.code})

    def call(self, kind, *, workspace=None, **arguments):
        workspace = workspace or self.workspace
        capabilities = Capabilities.from_provider(workspace)
        space = ModularDecisionActionSpace(capabilities.definitions)
        call = space.validate(OperationCall('workspace_' + kind, {'root': 'run', **arguments}))
        return BoundDecisionTools(capabilities.bindings).execute(call).data

    def test_capabilities_are_read_only_and_use_existing_permissions(self):
        caps = self.workspace.capabilities
        self.assertIsInstance(caps, Capabilities)
        self.assertEqual(set(caps.bindings), {'workspace_list', 'workspace_read', 'workspace_search'})
        self.assertTrue(all(definition.spec.read_only for definition in caps.definitions))
        denied = ModularDecisionActionSpace(caps.definitions, permissions=ActionPermissions(operations=()))
        outcome = execute_operation(BoundDecisionTools(caps.bindings), authorized_space(denied),
                                    OperationCall('workspace_list', {'root': 'run'}))
        self.assertIsInstance(outcome.error, PermissionError)
        allowed = ModularDecisionActionSpace(caps.definitions)
        self.assertEqual(len(authorized_space(allowed, 'review').describe()), 3)
        schema = caps.definitions[0].spec.input_schema
        self.assertEqual(schema['properties']['root']['enum'], ['run', 'code'])
        self.assertNotIn(str(self.run), json.dumps(schema))

    def test_read_pages_facts_and_multiple_roots(self):
        (self.run / 'log.txt').write_text('alpha\nbeta\ngamma\n')
        (self.code / 'config.py').write_text('answer = 42')
        first = self.call('read', path='log.txt', max_bytes=6)
        self.assertEqual(first['text'], 'alpha\n')
        self.assertEqual(first['next_offset'], 6)
        self.assertTrue(first['truncated'])
        self.assertEqual(first['source'], 'workspace:run')
        self.assertEqual(first['path'], 'log.txt')
        self.assertTrue(first['observed_at'].endswith('+00:00'))
        self.assertTrue(first['read_only'])
        rest = self.call('read', path='log.txt', offset=first['next_offset'])
        self.assertEqual(rest['text'], 'beta\ngamma\n')
        self.assertFalse(rest['truncated'])
        self.assertIsNone(rest['next_offset'])
        self.assertEqual(self.call('read', root='code', path='config.py')['text'], 'answer = 42')
        self.assertEqual(self.call('read', path='log.txt', offset=999)['text'], '')

    def test_list_pagination_does_not_materialize_or_sort_whole_directory(self):
        for number in range(7):
            (self.run / f'{number}.txt').write_text(str(number))
        paths, offset = [], 0
        for _ in range(5):
            page = self.call('list', limit=2, offset=offset)
            paths.extend(item['path'] for item in page['items'])
            if page['next_offset'] is None:
                break
            self.assertGreater(page['next_offset'], offset)
            offset = page['next_offset']
        self.assertEqual(set(paths), {f'{number}.txt' for number in range(7)})
        self.assertEqual(len(paths), 7)
        self.assertEqual(self.call('list', offset=99)['items'], [])

    def test_literal_search_file_directory_and_result_pages(self):
        (self.run / 'logs').mkdir()
        (self.run / 'logs' / 'worker.log').write_text('loss=3\nloss=2\nok\nloss=1\n')
        (self.run / 'config.json').write_text('{"loss": 0}\n')
        first = self.call('search', path='logs/worker.log', query='loss=', limit=1)
        self.assertEqual(first['items'][0]['line'], 1)
        self.assertEqual(first['items'][0]['byte_offset'], 0)
        second = self.call('search', path='logs/worker.log', query='loss=', limit=1,
                           offset=first['next_offset'])
        self.assertEqual(second['items'][0]['line'], 2)
        third = self.call('search', path='logs/worker.log', query='loss=', offset=second['next_offset'])
        self.assertEqual(third['items'][0]['line'], 4)
        self.assertFalse(third['truncated'])
        found = self.call('search', query='loss')
        self.assertEqual(len(found['items']), 4)
        self.assertEqual({item['path'] for item in found['items']}, {'logs/worker.log', 'config.json'})
        self.assertEqual(self.call('search', query='loss.*')['items'], [])

    def test_rejects_escape_absolute_unknown_root_and_malformed_arguments(self):
        outside = self.base / 'secret.txt'
        outside.write_text('PRIVATE')
        for kind in ('list', 'read', 'search'):
            extra = {'query': 'PRIVATE'} if kind == 'search' else {}
            for path in ('../secret.txt', 'child/../../secret.txt', str(outside), 'bad\x00path'):
                with self.subTest(kind=kind, path=path), self.assertRaises(ValueError):
                    self.call(kind, path=path, **extra)
        for args in ({'root': 'host'}, {'offset': -1}, {'offset': True}, {'limit': 0},
                     {'path': 1}, {'path': ''}, {'unknown': True}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.call('list', **args)
        with self.assertRaises(ValueError):
            self.call('read', path='anything', max_bytes=self.workspace.max_bytes + 1)
        for args in ({'query': ''}, {'query': 'a\nb'}, {'query': 'x', 'max_depth': -1},
                     {'query': 'x', 'scan_budget_bytes': True}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.call('search', **args)

    def test_symlinks_cannot_cross_even_between_two_configured_roots(self):
        (self.code / 'private').write_text('PRIVATE')
        (self.run / 'escape').symlink_to(self.code, target_is_directory=True)
        (self.run / 'file-link').symlink_to(self.code / 'private')
        (self.run / 'broken').symlink_to(self.run / 'missing')
        (self.run / 'cycle').symlink_to(self.run / 'cycle')
        for path in ('escape/private', 'file-link'):
            page = self.call('read', path=path)
            self.assertFalse(page['available'])
            self.assertEqual(page['unavailable_reason'], 'path_denied')
            self.assertNotIn(str(self.code), json.dumps(page))
        self.assertFalse(self.call('read', path='broken')['available'])
        self.assertFalse(self.call('read', path='cycle')['available'])
        found = self.call('search', query='PRIVATE')
        self.assertEqual(found['items'], [])
        self.assertTrue(found['truncated'])
        listing = self.call('list')
        self.assertTrue(all(item['kind'] == 'unavailable' for item in listing['items']))

    def test_in_root_links_work_and_directory_link_loops_are_not_walked(self):
        (self.run / 'text').write_text('hello')
        (self.run / 'alias').symlink_to('text')
        (self.run / 'back').symlink_to(self.run, target_is_directory=True)
        self.assertEqual(self.call('read', path='alias')['text'], 'hello')
        self.assertEqual(self.call('read', path='back/text')['text'], 'hello')
        result = self.call('search', query='hello')
        self.assertEqual(len(result['items']), 2)
        self.assertIn('directory_symlink', result['reasons'])

    def test_nofollow_rejects_link_swap_after_resolution(self):
        original = self.run / 'text'
        original.write_text('safe')
        private = self.base / 'private'
        private.write_text('PRIVATE')
        real_open = os.open

        def swapped(path, flags, *args, **kwargs):
            if path == 'text' and kwargs.get('dir_fd') is not None:
                original.unlink()
                original.symlink_to(private)
            return real_open(path, flags, *args, **kwargs)

        with patch('os.open', side_effect=swapped):
            result = self.call('read', path='text')
        self.assertFalse(result['available'])
        self.assertNotIn('PRIVATE', json.dumps(result))

    def test_missing_binary_empty_directory_and_special_file(self):
        self.assertEqual(self.call('read', path='missing')['unavailable_reason'], 'not_found')
        self.assertEqual(self.call('read')['unavailable_reason'], 'not_file')
        (self.run / 'empty').write_bytes(b'')
        self.assertEqual(self.call('read', path='empty')['text'], '')
        self.assertEqual(self.call('list', path='empty')['unavailable_reason'], 'not_directory')
        for name, raw in (('nul', b'a\x00b'), ('invalid-utf8', b'\xff\xfe')):
            (self.run / name).write_bytes(raw)
            result = self.call('read', path=name)
            self.assertTrue(result['binary'])
            self.assertFalse(result['available'])
            self.assertNotIn('text', result)
            self.assertEqual(self.call('search', path=name, query='a')['skipped_files'], 1)
        os.mkfifo(self.run / 'pipe')
        self.assertEqual(self.call('read', path='pipe')['unavailable_reason'], 'path_denied')

    def test_utf8_byte_boundaries_are_explicit_not_misclassified_as_binary(self):
        (self.run / 'utf8').write_text('中文日志')
        first = self.call('read', path='utf8', max_bytes=4)
        self.assertFalse(first['binary'])
        second = self.call('read', path='utf8', offset=4, max_bytes=4)
        self.assertFalse(second['binary'])
        self.assertEqual(second['decoding_errors'], 'replace')
        self.assertEqual(self.call('search', path='utf8', query='日志')['items'][0]['byte_offset'], 6)

    def test_giant_files_have_bounded_reads_and_search_budget(self):
        with (self.run / 'huge').open('wb') as file:
            file.write(b'needle' + b'x' * 100)
            file.truncate(1024 ** 3)
        workspace = WorkspaceTools({'run': self.run}, max_bytes=32, scan_budget_bytes=40)
        real_read = os.read
        counts = []

        def counted(fd, count):
            counts.append(count)
            return real_read(fd, count)

        with patch('os.read', side_effect=counted):
            read = self.call('read', workspace=workspace, path='huge')
        self.assertEqual(sum(counts), 32)
        self.assertEqual(read['size_bytes'], 1024 ** 3)
        self.assertTrue(read['truncated'])
        (self.run / 'other').write_bytes(b'needle' + b'x' * 100)
        counts.clear()
        with patch('os.read', side_effect=counted):
            found = self.call('search', workspace=workspace, query='needle')
        self.assertLessEqual(sum(counts), 40)
        self.assertLessEqual(max(counts), 32)
        self.assertEqual(found['scanned_bytes'], 40)
        self.assertTrue(found['truncated'])
        self.assertIn('scan_budget', found['reasons'])

    def test_depth_and_entry_budgets_report_incomplete_scans(self):
        (self.run / 'nested').mkdir()
        (self.run / 'nested' / 'hidden').write_text('needle')
        shallow = self.call('search', query='needle', max_depth=0)
        self.assertEqual(shallow['items'], [])
        self.assertIn('depth_budget', shallow['reasons'])
        for number in range(5):
            (self.run / str(number)).write_text('needle')
        limited = self.call('search', query='needle', max_entries=2)
        self.assertEqual(limited['scanned_entries'], 2)
        self.assertIn('entry_budget', limited['reasons'])
        workspace = WorkspaceTools({'run': self.run}, max_entries=2)
        listing = self.call('list', workspace=workspace)
        self.assertEqual(listing['scanned_entries'], 2)
        self.assertIn('entry_budget', listing['reasons'])

    def test_json_output_budget_including_escaped_text_and_long_lines(self):
        (self.run / 'text').write_text(('"\\\tneedle' + 'z' * 2000 + '\n') * 100)
        workspace = WorkspaceTools({'run': self.run}, max_bytes=100000, max_output_bytes=4096)
        for kind, extra in (('read', {}), ('search', {'query': 'needle'})):
            with self.subTest(kind=kind):
                result = self.call(kind, workspace=workspace, path='text', **extra)
                self.assertLessEqual(len(json.dumps(result).encode()), 4096)
                self.assertTrue(result['truncated'])
                if kind == 'search':
                    self.assertTrue(result['items'][0]['truncated'])
                    self.assertIn('needle', result['items'][0]['text'])

    def test_no_implicit_roots_or_mutable_root_mapping(self):
        with self.assertRaises(ValueError):
            WorkspaceTools({})
        with self.assertRaises(FileNotFoundError):
            WorkspaceTools({'missing': self.base / 'missing'})
        for kwargs in ({'max_bytes': True}, {'max_output_bytes': 10}, {'max_depth': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WorkspaceTools({'run': self.run}, **kwargs)
        roots = {'run': self.run}
        workspace = WorkspaceTools(roots)
        roots['host'] = self.base
        with self.assertRaises(ValueError):
            self.call('list', workspace=workspace, root='host')


if __name__ == '__main__':
    unittest.main()
