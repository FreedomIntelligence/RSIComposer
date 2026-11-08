"""Run ancestry records identity; only snapshot restore copies state (CPU mocks)."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_toy.monitor import Monitor, enrich_completed_run
from rsisys.application import Application, ApplicationConfig
from rsisys.cli import main
from rsisys.experience.records import ExperienceRecord
from rsisys.loops._memory import Memory
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.runtime.provenance import RunSource
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot


class RunProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = ModelRef('mock', 'model', 'baseline')
        self.config = ApplicationConfig('Inspect evidence', self.root / 'run5',
            {'mock': BackendBinding('mock', 'mock')},
            {'decision_maker': self.model, 'actor': self.model},
            run_id='run5', parent_run_id='run4')

    def read(self, root, name='run.json'):
        return json.loads((root / name).read_text())

    def capture(self):
        with Application(self.config) as app:
            app.run()
            app.runner.decision_making.harness.memory.remember(Memory('note', 'semantic', 'Saved note'))
            manifest = LocalSnapshotCapture(app).capture('source')
            return app.config.data_dir / 'snapshots' / manifest.snapshot_id, app.store.records()

    def test_child_records_immediate_source_and_rejects_invalid_identity(self):
        self.assertEqual(RunSource('run5', 'run4').child('run6'), RunSource('run6', 'run5'))
        for run_id, parent in (('', None), (' run5', None), ('run6', ''), ('run6', 5), ('run6', 'run6')):
            with self.subTest(run_id=run_id, parent=parent), self.assertRaises(ValueError):
                RunSource(run_id, parent)

    def test_same_run_durable_resume_preserves_identity_and_parent(self):
        with Application(self.config, durable=True):
            pass
        before = self.read(self.config.data_dir)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.config.source, RunSource('run5', 'run4'))
            self.assertEqual(self.read(app.config.data_dir), before)
            self.assertEqual(self.read(app.config.data_dir, 'config.json')['parent_run_id'], 'run4')

    def test_snapshot_fork_restores_memory_preserves_old_records_and_isolates_new_writes(self):
        snapshot, old_records = self.capture()
        target = self.root / 'run6'
        with restore_snapshot(snapshot, target, run_id='run6') as app:
            self.assertEqual(app.config.source, RunSource('run6', 'run5'))
            self.assertEqual(app.store.records(), old_records)
            self.assertEqual(app.runner.decision_making.harness.memory.recall()[0].content, 'Saved note')
            app.experience.record(ExperienceRecord('child-only', 'insight', 'shared', None, 'New evidence', True))
            app.run()
            records = {record.record_id: record for record in app.store.records()}
            self.assertEqual(records['child-only'].run_id, 'run6')
            self.assertTrue(all(records[record.record_id] == record for record in old_records))
            new_records = [record for record in records.values()
                           if record.record_id not in {old.record_id for old in old_records}]
            self.assertGreater(len(new_records), 1)
            self.assertTrue(all(record.run_id == 'run6' for record in new_records))
            monitor = Monitor(target, self.model, app.config.objective)
            self.assertEqual(monitor.data['parent_run_id'], 'run5')
            next_manifest = LocalSnapshotCapture(app).capture('child')
            next_snapshot = target / 'snapshots' / next_manifest.snapshot_id
        with restore_snapshot(next_snapshot, self.root / 'run7', run_id='run7') as grandchild:
            self.assertEqual(grandchild.config.source, RunSource('run7', 'run6'))
        with Application(self.config, _restore=True) as original:
            self.assertEqual(original.store.records(), old_records)
        self.assertEqual(self.read(target)['parent_run_id'], 'run5')

    def test_snapshot_same_run_restore_keeps_original_parent_and_rejects_self_fork(self):
        snapshot, _ = self.capture()
        with restore_snapshot(snapshot, self.root / 'relocated') as app:
            self.assertEqual(app.config.source, self.config.source)
            with self.assertRaisesRegex(ValueError, 'directory name'):
                Monitor(app.config.data_dir, self.model, app.config.objective)
        with self.assertRaisesRegex(ValueError, 'own parent'):
            restore_snapshot(snapshot, self.root / 'invalid', run_id='run5')
        self.assertFalse((self.root / 'invalid').exists())

    def test_monitor_reads_saved_source_and_rejects_conflicts_before_writing(self):
        with Application(self.config):
            pass
        monitor = Monitor(self.config.data_dir, self.model, 'Public goal')
        self.assertEqual(monitor.data['parent_run_id'], 'run4')
        before = (self.config.data_dir / 'monitor.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            Monitor(self.config.data_dir, self.model, 'Goal', source=RunSource('run5', 'other'))
        self.assertEqual((self.config.data_dir / 'monitor.json').read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'directory name'):
            Monitor(self.root, self.model, 'Goal', source=RunSource('different'))
        self.assertFalse((self.root / 'monitor.json').exists())

    def test_offline_export_preserves_source_and_does_not_infer_history(self):
        root = self.config.data_dir
        root.mkdir()
        monitor = Monitor(root, self.model, 'Goal', source=self.config.source)
        before = monitor.data['updated_at']
        (root / 'result.json').write_text(json.dumps({'status': 'completed', 'phase': 'finished'}))
        (root / 'decision_memory.json').write_text('{"history": []}')
        exported = enrich_completed_run(root)
        self.assertEqual(RunSource.from_dict(exported), self.config.source)
        self.assertEqual(exported['updated_at'], before)
        # Explicit run metadata can supply a field missing in an old monitor.
        exported.pop('parent_run_id')
        (root / 'monitor.json').write_text(json.dumps(exported))
        (root / 'run.json').write_text(json.dumps(self.config.source.to_dict()))
        self.assertEqual(enrich_completed_run(root)['parent_run_id'], 'run4')
        (root / 'run.json').unlink()
        (root / 'monitor.json').write_text(json.dumps(exported))
        self.assertIsNone(enrich_completed_run(root)['parent_run_id'])

    def test_offline_export_rejects_conflicting_source_without_changing_snapshot(self):
        root = self.config.data_dir
        root.mkdir()
        Monitor(root, self.model, 'Goal', source=self.config.source)
        (root / 'run.json').write_text(json.dumps(RunSource('run5', 'different').to_dict()))
        (root / 'result.json').write_text('{"status": "completed", "phase": "finished"}')
        before = (root / 'monitor.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            enrich_completed_run(root)
        self.assertEqual((root / 'monitor.json').read_bytes(), before)

    def test_old_config_and_run_metadata_leave_unknown_parent_empty(self):
        raw = self.config.to_dict()
        raw.pop('parent_run_id')
        config = ApplicationConfig.from_dict(raw, base_dir=self.root)
        self.assertIsNone(config.parent_run_id)
        with Application(config, durable=True):
            pass
        metadata = self.read(config.data_dir)
        metadata.pop('parent_run_id')
        (config.data_dir / 'run.json').write_text(json.dumps(metadata))
        with Application.resume(config.data_dir) as app:
            self.assertIsNone(app.config.parent_run_id)
        self.assertIsNone(Monitor(config.data_dir, self.model, 'Goal').data['parent_run_id'])

    def test_cli_declares_source_and_snapshot_fork_automatically_uses_immediate_parent(self):
        config_path = self.root / 'input.json'
        config_path.write_text(json.dumps(replace(self.config, parent_run_id=None).to_dict()))

        def cli(*arguments):
            output = io.StringIO()
            with patch.object(sys, 'argv', ['rsisys', *arguments]), redirect_stdout(output):
                main()
            return json.loads(output.getvalue())

        first = cli('run', '--config', str(config_path), '--parent-run-id', 'run4')
        self.assertEqual(RunSource.from_dict(first), self.config.source)
        fork = cli('resume', '--snapshot', first['snapshot'], '--data-dir', str(self.root / 'run6'),
                   '--run-id', 'run6')
        self.assertEqual(RunSource.from_dict(fork), RunSource('run6', 'run5'))
