"""M02/M13 audit: local failure boundaries, with no model or training services."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import fcntl
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.application import Application, ApplicationConfig
from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import BasicExperienceSystem, RecentFeedbackSelector
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.coordination import ExplorationProgress, LoopStepResult
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.runtime.artifacts import FileArtifactStore, InMemoryArtifactStore
from rsisys.runtime.files import atomic_write, json_bytes, sync_tree
from rsisys.runtime.recovery import BuiltinRecovery, RecoveryRequired
from rsisys.runtime.runner import Runner
from rsisys.runtime.run_policy import TimeRunPolicy
from rsisys.snapshot.capture import UnimplementedCapture
from rsisys.snapshot.codec import decode, encode
from rsisys.snapshot.local import LocalSnapshotCapture, FileSnapshotStore, restore_snapshot
from rsisys.snapshot.restore import UnimplementedRestore


ROOT = Path(__file__).resolve().parents[1]


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class DecisionPort:
    """Minimal public endpoint: committed receives may lose their response."""
    def __init__(self, directive):
        self.directive = directive
        self.received = []
        self.calls = 0

    def step(self, objective):
        self.calls += 1
        return LoopStepResult('dispatch', self.directive)

    def pending(self):
        return self.directive

    def receive_feedback(self, directive, feedback):
        self.received.append((deepcopy(directive), deepcopy(feedback)))
        self.directive = None


class ExplorationPort:
    """UNKNOWN remains pending; a cancel acknowledgement is not a terminal."""
    def __init__(self, directive):
        self.directive = directive
        self.feedback = None
        self.accepts = self.advances = self.cancels = self.acks = 0
        self.active = False
        self.phase = 'unknown'

    def inspect(self, directive_id):
        return ExplorationProgress(directive_id, self.phase, self.directive, feedback=self.feedback)

    def pending(self):
        return self.inspect(self.directive.directive_id) if self.active else None

    def accept(self, directive):
        self.accepts += 1
        self.active = True
        return self.inspect(directive.directive_id)

    def advance(self, directive_id):
        self.advances += 1
        return self.inspect(directive_id)

    def cancel(self, directive_id):
        self.cancels += 1
        return self.inspect(directive_id)

    def acknowledge(self, directive_id):
        if self.feedback is None:
            raise AssertionError('Acknowledged nonterminal execution')
        self.acks += 1
        self.active = False


class RuntimeBoundaryAuditTests(unittest.TestCase):
    def ports(self, **kwargs):
        directive = Directive('audit-command', 'task', 'inspect')
        decision, exploration = DecisionPort(directive), ExplorationPort(directive)
        runner = Runner(decision, exploration, **kwargs)
        return runner, decision, exploration

    def test_receive_failure_never_acknowledges_and_retry_keeps_exact_outbox(self):
        runner, decision, exploration = self.ports()
        feedback = Feedback('audit-command', ExecutionStatus.COMPLETED, 'observed', False,
                            execution_metrics={'steps': 7}, evidence_refs=('source-1',))
        exploration.feedback = feedback
        receive = decision.receive_feedback
        with patch.object(decision, 'receive_feedback', side_effect=OSError('receive unavailable')):
            with self.assertRaises(OSError):
                runner.step('task')
        self.assertEqual(exploration.acks, 0)
        self.assertEqual(runner._feedback, feedback)
        decision.receive_feedback = receive
        self.assertEqual(runner.step('task').kind, 'completed')
        self.assertEqual(decision.received, [(exploration.directive, feedback)])
        self.assertEqual(exploration.advances, 0)
        self.assertEqual(len(runner._outcomes), 1)

    def test_lost_ack_retries_delivery_before_any_new_decision_or_execution(self):
        runner, decision, exploration = self.ports()
        exploration.feedback = Feedback('audit-command', ExecutionStatus.COMPLETED, 'done', False)
        acknowledge = exploration.acknowledge
        def lost(ident):
            acknowledge(ident)
            raise OSError('ack response lost')
        with patch.object(exploration, 'acknowledge', side_effect=lost):
            with self.assertRaises(OSError):
                runner.step('task')
        self.assertIsNone(decision.pending())
        self.assertIsNotNone(runner._feedback)
        self.assertEqual(runner.step('task').kind, 'completed')
        self.assertEqual(decision.received[0], decision.received[1])
        self.assertEqual(decision.calls, 0)  # recovered the already-owned command
        self.assertEqual(exploration.accepts, 1)

    def test_unknown_cancel_does_not_ack_or_claim_shutdown_then_late_success_delivers(self):
        clock = Clock()
        runner, decision, exploration = self.ports(clock=clock, run_policy=TimeRunPolicy(1))
        self.assertEqual(runner.step('task').kind, 'waiting')
        clock.now = 1
        stopped = runner.step('task')
        self.assertEqual(stopped.kind, 'waiting')
        self.assertFalse(stopped.event['run_stop']['shutdown_complete'])
        self.assertEqual(exploration.acks, 0)
        self.assertTrue(runner.busy)
        exploration.feedback = Feedback('audit-command', ExecutionStatus.COMPLETED, 'late success', False)
        self.assertEqual(runner.step('task').kind, 'completed')
        self.assertEqual(decision.received[0][1].status, ExecutionStatus.COMPLETED)
        self.assertTrue(runner.step('task').event['run_stop']['shutdown_complete'])

    def test_unknown_at_cleanup_deadline_remains_explicitly_unresolved(self):
        clock = Clock()
        runner, _, exploration = self.ports(clock=clock, run_policy=TimeRunPolicy(1),
                                            cleanup_timeout_seconds=2)
        runner.step('task')
        clock.now = 1
        runner.step('task')
        clock.now = 3
        result = runner.step('task')
        self.assertEqual(result.kind, 'finished')
        self.assertTrue(result.event['run_stop']['cleanup_timed_out'])
        self.assertFalse(result.event['run_stop']['shutdown_complete'])
        self.assertEqual(result.event['run_stop']['unresolved_directives'], ['audit-command'])
        self.assertEqual(exploration.acks, 0)

    def test_keyboard_interrupt_preserves_owned_directive_and_original_error(self):
        runner, _, exploration = self.ports()
        with patch.object(exploration, 'advance', side_effect=KeyboardInterrupt('operator')):
            with self.assertRaisesRegex(KeyboardInterrupt, 'operator'):
                runner.step('task')
        self.assertTrue(runner.busy)
        self.assertEqual(exploration.acks, 0)
        exploration.feedback = Feedback('audit-command', ExecutionStatus.CANCELLED, 'confirmed', False)
        self.assertEqual(runner.step('task').kind, 'completed')

    def test_wrong_endpoint_receipt_cannot_be_delivered(self):
        runner, decision, exploration = self.ports()
        other = Directive('other', 'task', 'inspect')
        wrong = ExplorationProgress('other', 'unknown', other)
        with patch.object(exploration, 'accept', return_value=wrong):
            with self.assertRaisesRegex(ValueError, 'another directive'):
                runner.step('task')
        self.assertEqual(decision.received, [])
        self.assertEqual(exploration.acks, 0)


class PersistenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'),
                              data_dir=self.root / 'run', execution='mock', learning='none')

    def snapshot(self):
        with Application(self.config) as app:
            manifest = LocalSnapshotCapture(app).capture('audit')
        return self.config.data_dir / 'snapshots' / manifest.snapshot_id

    def rewrite_snapshot_file(self, directory, name, value):
        # A consistent digest is not schema validation. Exercise that second boundary.
        path = directory / name
        path.write_bytes(json_bytes(value))
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['file_digests'][name] = sha256(path.read_bytes()).hexdigest()
        (directory / 'manifest.json').write_bytes(json_bytes(manifest))

    def test_recovery_rejects_nonfinite_contract_before_effect_or_intent(self):
        class Contract:
            contract_id = 'audit-json/v1'
            value = 0.0
            def capture(self, app):
                return {'value': self.value, 'builtin': BuiltinRecovery().capture(app)}
            def restore(self, app, state):
                self.value = state['value']
                BuiltinRecovery().restore(app, state['builtin'])
        contract = Contract()
        with Application(self.config, durable=True, recovery_contract=contract) as app:
            journal = (app.recovery.root / 'journal.json').read_bytes()
            contract.value = float('nan')
            effects = []
            with self.assertRaises(ValueError):
                app.recovery.mutate('fixture', (), lambda: effects.append('executed'))
            self.assertEqual(effects, [])
            self.assertEqual((app.recovery.root / 'journal.json').read_bytes(), journal)
            self.assertIsNone(app.recovery.intervention)

    def test_snapshot_and_builtin_recovery_reject_uncontracted_decision_hooks(self):
        hooks = ('_context_processors', '_feedback_listeners', '_receipt_listeners',
                 '_cleanup_callbacks', 'feedback_proposal_resolver')
        for hook in hooks:
            with self.subTest(hook=hook):
                config = replace(self.config, data_dir=self.root / hook)
                with Application(config) as app:
                    dm = app.runner.decision_making.harness
                    callback = lambda *args: None
                    if hook == 'feedback_proposal_resolver':
                        setattr(dm, hook, callback)
                    else:
                        getattr(dm, hook).append(callback)
                    with self.assertRaisesRegex(TypeError, 'hooks'):
                        LocalSnapshotCapture(app).capture('cannot-drop-hooks')
                    with self.assertRaisesRegex(TypeError, 'hooks'):
                        BuiltinRecovery().capture(app)
                    self.assertFalse((config.data_dir / 'snapshots').exists())

    def test_invalid_restored_compactor_does_not_publish_target_or_leave_lock(self):
        snapshot = self.snapshot()
        runtime = json.loads((snapshot / 'runtime.json').read_text())
        runtime['context_compactor']['target_tokens'] = 100000
        self.rewrite_snapshot_file(snapshot, 'runtime.json', runtime)
        target = self.root / 'restored'
        with self.assertRaisesRegex(ValueError, 'target_tokens'):
            restore_snapshot(snapshot, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob('.restore-*')), [])

    def test_postconstruction_restore_failure_closes_new_application(self):
        snapshot = self.snapshot()
        target = self.root / 'restored'
        constructed = []
        real = Application
        def constructor(*args, **kwargs):
            app = real(*args, **kwargs)
            constructed.append(app)
            return app
        # The first restore_research call validates before publication. Fail only
        # when attaching it to the newly constructed live application.
        from rsisys.snapshot.local import restore_research
        def fail_live(dm, agenda, compactor):
            if constructed:
                raise RuntimeError('attachment failed')
            return restore_research(dm, agenda, compactor)
        with patch('rsisys.application.Application', side_effect=constructor), \
                patch('rsisys.snapshot.local.restore_research', side_effect=fail_live):
            with self.assertRaisesRegex(RuntimeError, 'attachment failed'):
                restore_snapshot(snapshot, target)
        self.assertTrue(constructed[0]._closed)
        self.assertFalse(target.exists())

    def test_invalid_database_after_publication_rolls_back_only_new_target(self):
        snapshot = self.snapshot()
        data = b'not a sqlite database'
        (snapshot / 'experience.sqlite3').write_bytes(data)
        manifest = json.loads((snapshot / 'manifest.json').read_text())
        manifest['file_digests']['experience.sqlite3'] = sha256(data).hexdigest()
        (snapshot / 'manifest.json').write_bytes(json_bytes(manifest))
        target = self.root / 'restored'
        with self.assertRaises(sqlite3.DatabaseError):
            restore_snapshot(snapshot, target)
        self.assertFalse(target.exists())
        self.assertEqual((snapshot / 'experience.sqlite3').read_bytes(), data)
        self.assertEqual(list(self.root.glob('.restore-*')), [])

    def test_restore_initialization_lock_race_retains_other_owners_workspace(self):
        snapshot = self.snapshot()
        target = self.root / 'contended'
        held = []
        real = Application
        def competing_owner(*args, **kwargs):
            lock = (target / '.lock').open('a+b')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.append(lock)
            (target / 'other-owner').write_text('must survive')
            return real(*args, **kwargs)  # actual nonblocking Application lock rejection
        try:
            with patch('rsisys.application.Application', side_effect=competing_owner):
                with self.assertRaisesRegex(RuntimeError, 'another process') as raised:
                    restore_snapshot(snapshot, target)
            self.assertEqual((target / 'other-owner').read_text(), 'must survive')
            self.assertTrue(any('target cleanup' in note for note in raised.exception.__notes__))
        finally:
            for lock in held:
                lock.close()

    def test_restore_failure_retains_replaced_directory_or_lock_inode(self):
        snapshot = self.snapshot()
        for replace_directory in (False, True):
            with self.subTest(directory=replace_directory):
                target = self.root / ('foreign-directory' if replace_directory else 'foreign-lock')
                def replace_owner(*args, **kwargs):
                    if replace_directory:
                        target.rename(self.root / 'displaced-staging')
                        target.mkdir()
                    else:
                        (target / '.lock').unlink()
                    (target / '.lock').write_text('new owner')
                    (target / 'other-owner').write_text('must survive')
                    raise RuntimeError('fixture initialization failed')
                with patch('rsisys.application.Application', side_effect=replace_owner):
                    with self.assertRaisesRegex(RuntimeError, 'fixture initialization') as raised:
                        restore_snapshot(snapshot, target)
                self.assertEqual((target / 'other-owner').read_text(), 'must survive')
                self.assertTrue(any('identity changed' in note for note in raised.exception.__notes__))

    def test_snapshot_preserves_full_adaptive_compactor_state_and_limits(self):
        with Application(self.config) as app:
            dm = app.runner.decision_making.harness
            dm.context_compactor = BasicContextCompactor(trigger_tokens=80000, target_tokens=24000,
                max_target_tokens=60000, target_increment_tokens=12000, summary_tokens=3072)
            rolling = {'folded': {name: ['a' * 64] for name in dm.context_compactor.streams},
                       'summary': {'history': ['source-qualified observation']},
                       'compactions': 3, 'project_evidence': True}
            dm.context_compactor.restore_state(rolling)
            dm.experience_limit, dm.memory_limit, dm.review_tool_limit = 7, 5, 2
            manifest = LocalSnapshotCapture(app).capture('adaptive')
        snapshot = self.config.data_dir / 'snapshots' / manifest.snapshot_id
        with restore_snapshot(snapshot, self.root / 'fork') as app:
            dm = app.runner.decision_making.harness
            with self.assertRaisesRegex(RuntimeError, 'another process'):
                Application(app.config, _restore=True)
            self.assertEqual(dm.context_compactor.export_state(), rolling)
            self.assertEqual(dm.context_compactor.max_target_tokens, 60000)
            self.assertEqual(dm.context_compactor.target_increment_tokens, 12000)
            self.assertEqual(dm.context_compactor.summary_tokens, 3072)
            self.assertEqual((dm.experience_limit, dm.memory_limit, dm.review_tool_limit), (7, 5, 2))

    def test_snapshot_corrupt_bytes_and_symlinks_never_create_target(self):
        snapshot = self.snapshot()
        target = self.root / 'fork'
        path = snapshot / 'state.json'
        saved = path.read_bytes()
        path.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'digest'):
            restore_snapshot(snapshot, target)
        path.write_bytes(saved)
        (snapshot / 'link').symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            restore_snapshot(snapshot, target)
        self.assertFalse(target.exists())

    def test_capture_failure_never_publishes_or_updates_snapshot_ancestry(self):
        with Application(self.config) as app:
            capture = LocalSnapshotCapture(app)
            with patch('rsisys.snapshot.local.sync_tree', side_effect=OSError('fsync failed')):
                with self.assertRaises(OSError):
                    capture.capture('interrupted')
            self.assertIsNone(getattr(app, 'parent_snapshot_id', None))
            self.assertEqual(list(capture.store.root.iterdir()), [])
            manifest = capture.capture('retry')
            self.assertEqual(capture.store.load(manifest.snapshot_id), manifest)

    def test_feedback_service_is_rejected_without_claiming_builtin_recovery(self):
        with Application(self.config) as app:
            # No worker or model call: this tests only admission of the boundary.
            app.runner.decision_making.feedback_service = object()
            with self.assertRaisesRegex(TypeError, 'Feedback service'):
                LocalSnapshotCapture(app).capture('not-supported')
            with self.assertRaisesRegex(TypeError, 'Feedback service'):
                BuiltinRecovery().capture(app)
            self.assertFalse((self.config.data_dir / 'snapshots').exists())

    def test_intent_write_failure_never_runs_callback_and_blocks_same_process(self):
        with Application(self.config, durable=True) as app:
            effects = []
            with patch('rsisys.runtime.recovery.atomic_write', side_effect=OSError('journal unavailable')):
                with self.assertRaises(OSError):
                    app.recovery.mutate('fixture', (), lambda: effects.append('executed'))
            self.assertEqual(effects, [])
            with self.assertRaises(RecoveryRequired):
                app.runner.step('task')

    def test_unknown_sqlite_schema_is_rejected_without_rewriting_its_version(self):
        path = self.root / 'old.sqlite3'
        store = SQLiteExperienceStore(path)
        with store.connection:
            store.connection.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
        store.close()
        with self.assertRaisesRegex(ValueError, 'schema'):
            SQLiteExperienceStore(path)
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0], '999')

    def test_snapshot_store_rejects_escape_and_placeholders_remain_explicit(self):
        with self.assertRaises(ValueError):
            FileSnapshotStore(self.root).load('../escape')
        with self.assertRaises(NotImplementedError):
            UnimplementedCapture().capture('not-supported')
        with self.assertRaises(NotImplementedError):
            UnimplementedRestore().restore(None)
        with self.assertRaises(ValueError):
            decode({'$type': 'untrusted.module', 'fields': {}})
        value = {'tuple': ('a',), 'set': {'b'}, 'list': [1], 'map': {2: 'key'}}
        self.assertEqual(decode(encode(value)), value)

    def test_journal_generation_corruption_remains_closed(self):
        with Application(self.config, durable=True):
            pass
        journal = self.config.data_dir / 'recovery' / 'journal.json'
        journal.write_bytes(json_bytes({'generation': 3, 'status': 'committed'}))
        with self.assertRaisesRegex(ValueError, 'generation'):
            Application.resume(self.config.data_dir)

    def test_reconciliation_failure_stays_blocked_until_evidenced_repair(self):
        with Application(self.config, durable=True) as app:
            with self.assertRaises(RuntimeError):
                app.recovery.mutate('fixture', (), lambda: (_ for _ in ()).throw(RuntimeError('interrupted')))
            with self.assertRaises(TypeError):
                app.recovery.reconcile(lambda app, incident: False, evidence='read-only check')
            with self.assertRaises(RecoveryRequired):
                app.runner.step('task')
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.recovery.intervention['operation'], 'reconcile')
            app.recovery.reconcile(lambda app, incident: None, evidence='operator verified no external effect')
            self.assertIsNone(app.recovery.intervention)

    def test_experience_unicode_casefold_matches_reopened_sqlite_and_recent_limit(self):
        memory = InMemoryExperienceStore()
        path = self.root / 'experience.sqlite3'
        sqlite = SQLiteExperienceStore(path)
        for store in (memory, sqlite):
            for ident, summary in (('old', 'Straße ΔΟΚΙΜΗ'), ('new', 'STRASSE δοκιμη')):
                store.append(ExperienceRecord(ident, 'insight', 'shared', None, summary, False))
        sqlite.close()
        sqlite = SQLiteExperienceStore(path)
        self.addCleanup(sqlite.close)
        for query in ('STRASSE', 'straße', 'δοκιμη', 'ΔΟΚΙΜΗ'):
            for limit in (None, 1):
                with self.subTest(query=query, limit=limit):
                    self.assertEqual(sqlite.query('actor', text=query, limit=limit),
                                     memory.query('actor', text=query, limit=limit))

    def test_experience_conflicts_detachment_source_scope_and_run_identity(self):
        sqlite = SQLiteExperienceStore(self.root / 'experience.sqlite3')
        self.addCleanup(sqlite.close)
        for store in (InMemoryExperienceStore(), sqlite):
            with self.subTest(store=type(store).__name__):
                system = BasicExperienceSystem(store, run_id='audit-run')
                original = ExperienceRecord('source', 'insight', 'actor', None, 'observation', False,
                    source_refs=('raw:1',), details={'metrics': {'steps': 3}})
                system.record(original)
                system.record(original)
                original.details['metrics']['steps'] = 99
                selected = RecentFeedbackSelector().select(system, 'task', limit=1)
                self.assertEqual(selected[0].details['metrics']['steps'], 3)
                selected[0].details.clear()
                self.assertEqual(system.retrieve()[0].source_refs, ('raw:1',))
                self.assertEqual(system.retrieve()[0].run_id, 'audit-run')
                self.assertEqual(system.retrieve('decision_maker'), ())
                with self.assertRaises(ValueError):
                    system.record(replace(original, summary='conflict'))
                with self.assertRaises(ValueError):
                    system.record(replace(original, record_id='foreign', run_id='different'))
                for bad in (True, 0, -1):
                    with self.assertRaises(ValueError):
                        system.retrieve(limit=bad)

    def test_atomic_write_replace_failure_retains_old_content_and_removes_staging(self):
        path = self.root / 'state.json'
        atomic_write(path, b'old')
        with patch('rsisys.runtime.files.os.replace', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError):
                atomic_write(path, b'new')
        self.assertEqual(path.read_bytes(), b'old')
        self.assertEqual(list(self.root.glob('.write-*')), [])
        with self.assertRaises(ValueError):
            json_bytes({'not-json': float('nan')})
        (self.root / 'link').symlink_to(path)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            sync_tree(self.root)

    def test_artifact_metadata_failure_does_not_publish_partial_content(self):
        store = FileArtifactStore(self.root / 'artifacts')
        with patch('rsisys.runtime.files.atomic_write', side_effect=OSError('metadata unavailable')):
            with self.assertRaises(OSError):
                store.put(b'large evidence', kind='report')
        self.assertEqual(store.refs(), ())
        self.assertEqual(list(store.root.iterdir()), [])

    def test_artifact_reference_and_digest_must_match_in_both_stores(self):
        for store in (InMemoryArtifactStore(), FileArtifactStore(self.root / 'artifacts')):
            ref = store.put(b'raw evidence', kind='report')
            self.assertEqual(store.get(ref), b'raw evidence')
            for forged in (replace(ref, kind='different'), replace(ref, digest='0' * 64),
                           replace(ref, uri='https://external.invalid/evidence')):
                with self.assertRaises(ValueError):
                    store.get(forged)


if __name__ == '__main__':
    unittest.main()
