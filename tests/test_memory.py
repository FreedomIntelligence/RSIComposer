from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsisys.loops._memory import MemoryBuffer
from rsisys.loops.contracts.memory import HistoryEntry, LoopMemory, Memory
from rsisys.loops.decision_making.memory import DecisionMemory
from rsisys.loops.exploration.memory import ExplorationMemory


def entry(record_id='record', kind='decision', task_id='task', **kwargs):
    return HistoryEntry(record_id, kind, task_id, 'A local event', False, **kwargs)


def memory(memory_id='memory', **kwargs):
    return Memory(memory_id, 'procedural', 'Inspect outputs before accepting', **kwargs)


class MemoryContractTests(unittest.TestCase):
    def test_defaults_and_no_experience_scope(self):
        item = memory()
        self.assertEqual(item.source_refs, ())
        self.assertEqual(item.applicability, '')
        self.assertEqual(item.status, 'unverified')
        self.assertNotIn('scope', {field.name for field in fields(item)})
        first, second = entry(), entry('second')
        first.details['changed'] = True
        self.assertEqual(second.details, {})
        self.assertEqual(second.source_refs, ())
        self.assertIsNone(entry(task_id=None).task_id)

    def test_required_identifiers(self):
        for invalid in ('', '   ', None, 17):
            for factory in (
                lambda: memory(invalid),
                lambda: entry(invalid),
                lambda: entry(kind=invalid),
            ):
                with self.subTest(invalid=invalid, factory=factory):
                    with self.assertRaises(ValueError):
                        factory()
            if invalid is not None:
                with self.assertRaises(ValueError):
                    entry(task_id=invalid)

    def test_memory_kind_status_and_content_validation(self):
        for kind in ('episodic', 'semantic', 'procedural'):
            for status in ('unverified', 'supported', 'refuted'):
                self.assertEqual(Memory('id', kind, 'content', status=status).status, status)
        for changes in ({'kind': 'unknown'}, {'status': 'unknown'},
                        {'content': ''}, {'content': None}, {'applicability': None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(memory(), **changes)

    def test_invalid_history_fields_and_source_references(self):
        for changes in ({'summary': None}, {'simulated': 1}, {'details': []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(entry(), **changes)
        for refs in (['external'], ('',), (None,), 'external'):
            for factory in (memory, entry):
                with self.subTest(refs=refs, factory=factory), self.assertRaises(ValueError):
                    factory(source_refs=refs)


class MemoryBufferTests(unittest.TestCase):
    def test_loop_and_instance_isolation(self):
        decision: LoopMemory = DecisionMemory()
        other = DecisionMemory()
        actor = ExplorationMemory()
        decision.append(entry())
        decision.remember(memory())
        actor.append(entry(kind='trajectory'))
        actor.remember(replace(memory(), content='Independent actor memory'))
        self.assertEqual(other.history(), ())
        self.assertEqual(other.recall(), ())
        self.assertEqual(decision.history()[0].kind, 'decision')
        self.assertEqual(actor.history()[0].kind, 'trajectory')
        self.assertNotEqual(decision.recall(), actor.recall())

    def test_each_loop_accepts_only_its_history_kinds(self):
        for buffer, accepted in (
            (DecisionMemory(), ('decision', 'operation', 'feedback')),
            (ExplorationMemory(), ('trajectory',)),
        ):
            for kind in ('decision', 'operation', 'feedback', 'trajectory', 'unknown'):
                with self.subTest(owner=buffer.owner, kind=kind):
                    if kind in accepted:
                        buffer.append(entry(kind, kind))
                    else:
                        with self.assertRaises(ValueError):
                            buffer.append(entry(kind, kind))

    def test_default_buffer_is_pure_memory(self):
        with patch('rsisys.loops._memory.atomic_write') as write:
            buffer = MemoryBuffer()
            buffer.append(entry())
            buffer.remember(memory())
            write.assert_not_called()
        self.assertIsNone(buffer.path)
        self.assertEqual(buffer.history(), (entry(),))
        self.assertEqual(buffer.recall(), (memory(),))

    def test_history_idempotency_and_conflicting_ids(self):
        buffer = DecisionMemory()
        original = entry(details={'nested': [1]})
        buffer.append(original)
        buffer.append(entry('second'))
        buffer.append(deepcopy(original))
        self.assertEqual(buffer.history(), (original, entry('second')))
        before = buffer.export()
        with self.assertRaises(ValueError):
            buffer.append(replace(original, details={'nested': [2]}))
        self.assertEqual(buffer.export(), before)

    def test_history_filters_before_limit_and_preserves_chronology(self):
        buffer = DecisionMemory()
        entries = (
            entry('1'), entry('2', 'operation'), entry('3', task_id='other'),
            entry('4'), entry('5', 'feedback'), entry('6', task_id=None),
        )
        for item in entries:
            buffer.append(item)
        self.assertEqual(buffer.history(), entries)
        self.assertEqual(buffer.history(limit=2), entries[-2:])
        self.assertEqual(buffer.history(task_id='task', kind='decision', limit=1), (entries[3],))
        self.assertEqual(buffer.history(task_id='task', limit=2), (entries[3], entries[4]))
        self.assertEqual(buffer.history(kind='operation'), (entries[1],))
        self.assertEqual(buffer.history(task_id='missing'), ())
        self.assertEqual(buffer.history(kind='missing'), ())
        self.assertEqual(buffer.history(limit=100), entries)

    def test_memory_update_and_text_matching(self):
        buffer = DecisionMemory()
        first = memory('first', applicability='EXECUTION', source_refs=('artifact://output',))
        second = replace(memory('second'), content='Unrelated result', status='refuted')
        third = replace(memory('third'), content='Check outputs')
        for item in (first, second, third):
            buffer.remember(item)
        self.assertEqual(buffer.recall(), (third, second, first))
        self.assertEqual(buffer.recall(text='OUTPUTS', limit=1), (third,))
        self.assertEqual(buffer.recall(text='execution'), (first,))
        self.assertEqual(buffer.recall(text='absent'), ())
        updated = replace(first, status='supported')
        buffer.remember(updated)
        self.assertEqual(buffer.recall(), (updated, third, second))
        self.assertEqual(buffer.recall(text=''), buffer.recall())

    def test_recall_default_limit(self):
        buffer = DecisionMemory()
        for index in range(35):
            buffer.remember(memory(str(index)))
        self.assertEqual(len(buffer.recall()), 32)
        self.assertEqual(buffer.recall()[0].memory_id, '34')
        self.assertEqual(buffer.recall()[-1].memory_id, '3')

    def test_invalid_limits_even_on_empty_buffers(self):
        buffer = DecisionMemory()
        for invalid in (0, -1, True, False, 1.0, '2', [], float('inf')):
            for query in (buffer.history, buffer.recall):
                with self.subTest(limit=invalid, query=query), self.assertRaises(ValueError):
                    query(limit=invalid)
        with self.assertRaises(ValueError):
            buffer.recall(limit=None)
        self.assertEqual(buffer.history(limit=None), ())
        self.assertEqual(buffer.history(limit=1), ())
        self.assertEqual(buffer.recall(limit=1), ())

    def test_history_inputs_reads_and_exports_are_detached(self):
        buffer = DecisionMemory()
        details = {'nested': [{'values': [1]}]}
        original = entry(details=details)
        buffer.append(original)
        details['nested'][0]['values'].append(2)
        buffer.history()[0].details['nested'][0]['values'].append(3)
        exported = buffer.export()
        exported['history'][0]['details']['nested'][0]['values'].append(4)
        exported['history'].clear()
        self.assertEqual(buffer.history()[0].details, {'nested': [{'values': [1]}]})

    def test_memory_inputs_reads_and_exports_are_detached(self):
        buffer = DecisionMemory()
        original = memory(source_refs=('document://note',))
        buffer.remember(original)
        self.assertIsNot(buffer.recall()[0], original)
        self.assertIsNot(buffer.recall()[0], buffer.recall()[0])
        exported = buffer.export()
        exported['memories'][0]['content'] = 'Changed export'
        exported['memories'].clear()
        self.assertEqual(buffer.recall(), (original,))


class PersistentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_reopen_preserves_full_records_order_and_updates(self):
        for cls, kind in ((DecisionMemory, 'feedback'), (ExplorationMemory, 'trajectory')):
            with self.subTest(owner=cls.owner):
                path = self.root / cls.owner / 'memory.json'
                buffer = cls(path)
                self.assertFalse(path.exists())
                buffer.append(entry('1', kind, source_refs=('file://report',),
                                    details={'nested': [{'metric': 0.5}], 'unicode': '结果'}))
                buffer.append(replace(entry('2', kind, task_id=None), simulated=True))
                buffer.remember(memory('1', source_refs=('external:unregistered',)))
                buffer.remember(memory('2'))
                buffer.remember(replace(memory('1'), status='supported'))
                reopened = cls(path)
                self.assertEqual(reopened.export(), buffer.export())
                self.assertEqual(reopened.history(), buffer.history())
                self.assertEqual(reopened.recall(), buffer.recall())
                self.assertEqual(reopened.recall()[0].memory_id, '1')
                self.assertIsInstance(reopened.history()[0].source_refs, tuple)
                self.assertIsInstance(reopened.recall()[0].source_refs, tuple)
                with patch('rsisys.loops._memory.atomic_write') as write:
                    reopened.append(buffer.history()[0])
                    write.assert_not_called()

    def test_owner_mismatch_rejected_in_both_directions(self):
        for writer, reader, kind in (
            (DecisionMemory, ExplorationMemory, 'decision'),
            (ExplorationMemory, DecisionMemory, 'trajectory'),
        ):
            path = self.root / writer.owner
            writer(path).append(entry(kind=kind))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'owner'):
                reader(path)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_version_and_malformed_payload_rejected(self):
        path = self.root / 'invalid.json'
        valid = DecisionMemory().export()
        for version in (None, 0, 2, '1', True, 1.0):
            path.write_text(json.dumps(dict(valid, version=version)))
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'version'):
                DecisionMemory(path)
        for payload in ([], {}, dict(valid, history={}), dict(valid, memories=None),
                        dict(valid, history=[{}]), dict(valid, memories=[None])):
            path.write_text(json.dumps(payload))
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                DecisionMemory(path)
        path.write_text('{broken')
        with self.assertRaises(ValueError):
            DecisionMemory(path)

    def test_loaded_records_validate_kind_identity_and_duplicates(self):
        buffer = DecisionMemory()
        buffer.append(entry())
        buffer.remember(memory())
        valid = buffer.export()
        path = self.root / 'invalid-records.json'
        for collection, field, value in (
            ('history', 'kind', 'trajectory'), ('history', 'record_id', ''),
            ('history', 'task_id', ''), ('history', 'simulated', 'false'),
            ('memories', 'kind', 'unknown'), ('memories', 'status', 'unknown'),
            ('memories', 'memory_id', ''), ('memories', 'source_refs', 'not-a-list'),
        ):
            payload = deepcopy(valid)
            payload[collection][0][field] = value
            path.write_text(json.dumps(payload))
            with self.subTest(collection=collection, field=field), self.assertRaises(ValueError):
                DecisionMemory(path)
        for collection in ('history', 'memories'):
            payload = deepcopy(valid)
            payload[collection].append(deepcopy(payload[collection][0]))
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                DecisionMemory(path)

    def test_backup_restores_full_contents_without_rebinding(self):
        for persisted in (False, True):
            with self.subTest(persisted=persisted):
                path = self.root / 'source.json' if persisted else None
                buffer = DecisionMemory(path)
                buffer.append(entry(details={'nested': [1]}))
                for index in range(40):
                    buffer.remember(memory(str(index)))
                target = self.root / str(persisted) / 'snapshot.json'
                buffer.backup(target)
                restored = DecisionMemory(target)
                self.assertEqual(restored.export(), buffer.export())
                self.assertEqual(len(restored.recall(limit=100)), 40)
                buffer.append(entry('later'))
                self.assertEqual(len(DecisionMemory(target).history()), 1)
                self.assertEqual(buffer.path, path)
                if persisted:
                    self.assertEqual(DecisionMemory(path).export(), buffer.export())

    def test_failed_writes_leave_state_and_file_unchanged(self):
        path = self.root / 'memory.json'
        buffer = DecisionMemory(path)
        buffer.append(entry())
        buffer.remember(memory('old'))
        buffer.remember(memory('new'))
        before = buffer.export()
        disk = path.read_bytes()
        mutations = (
            lambda: buffer.append(entry('later')),
            lambda: buffer.remember(memory('later')),
            lambda: buffer.remember(replace(memory('old'), status='supported')),
            lambda: buffer.backup(self.root / 'failed-backup.json'),
        )
        for mutate in mutations:
            with patch('rsisys.loops._memory.atomic_write', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    mutate()
            self.assertEqual(buffer.export(), before)
            self.assertEqual(path.read_bytes(), disk)
            self.assertEqual(DecisionMemory(path).export(), before)
        buffer.append(entry('later'))
        buffer.remember(replace(memory('old'), status='supported'))
        self.assertEqual(DecisionMemory(path).export(), buffer.export())
        self.assertEqual(buffer.recall()[0].status, 'supported')

    def test_failed_initial_write_does_not_publish_candidate(self):
        path = self.root / 'memory.json'
        buffer = DecisionMemory(path)
        with patch('rsisys.loops._memory.atomic_write', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                buffer.append(entry())
            with self.assertRaises(OSError):
                buffer.remember(memory())
        self.assertEqual(buffer.history(), ())
        self.assertEqual(buffer.recall(), ())
        self.assertFalse(path.exists())

    def test_serialization_failure_leaves_state_and_file_unchanged(self):
        path = self.root / 'memory.json'
        buffer = DecisionMemory(path)
        buffer.append(entry())
        before = buffer.export()
        disk = path.read_bytes()
        for value, error in ((object(), TypeError), (float('nan'), ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                buffer.append(entry('invalid', details={'value': value}))
            self.assertEqual(buffer.export(), before)
            self.assertEqual(path.read_bytes(), disk)

    def test_atomic_replace_failure_leaves_state_and_file_unchanged(self):
        path = self.root / 'memory.json'
        buffer = DecisionMemory(path)
        buffer.append(entry())
        before = buffer.export()
        disk = path.read_bytes()
        with patch('rsisys.runtime.files.os.replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                buffer.append(entry('later'))
        self.assertEqual(buffer.export(), before)
        self.assertEqual(path.read_bytes(), disk)
        self.assertEqual(tuple(self.root.glob('.write-*')), ())


if __name__ == '__main__':
    unittest.main()
