"""Local Lab receipt identities survive storage and restart independently of filenames."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationCall, OperationSpec
from rsicomposer.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsicomposer.runtime.orchestration.lab_group import LabGroup
from rsicomposer.snapshot.codec import decode, encode


class _Lab:
    def __init__(self, job=None):
        self.calls = []
        self.job = job

    def describe(self):
        return (OperationSpec('run', 'Test operation', {'type': 'object'}),)

    def execute(self, call):
        self.calls.append(call.call_id)
        return self.job or OperationResult('Completed', (), False)

    def poll_job(self, job):
        return JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('Job completed', (), False))


class ReceiptStorageTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'lab_receipts'

    def group(self, *, job=None):
        lab = _Lab(job)
        group = LabGroup(receipts=self.root)
        group.add('test', lab)
        return group, lab

    def legacy_receipt(self, group, call, outcome):
        path = self.root / (call.call_id + '.json')
        path.write_text(json.dumps(encode({
            'schema': 24, 'name': call.name, 'arguments': call.arguments, 'outcome': outcome,
            'lab_state': None, 'profile_state': group.profile_state()['bindings']['test'],
            'sequence': 1, 'elapsed_seconds': 0.5,
        })), encoding='utf-8')
        return path

    def test_opaque_ids_have_stable_distinct_filenames_and_replay(self):
        ids = ('模型 @ v1!', 'Café', 'Cafe\u0301', 'Example-1', 'example-1', '界' * 200)
        calls = [OperationCall('test.run', {}, call_id=ident) for ident in ids]
        group, writer = self.group()
        reopened, reader = self.group()
        with patch.object(Path, 'iterdir', side_effect=AssertionError('Unexpected receipt scan')):
            outcomes = [group.execute(call) for call in calls]
            self.assertEqual([reopened.execute(call) for call in calls], outcomes)
        self.assertEqual(writer.calls, list(ids))
        self.assertEqual(reader.calls, [])
        self.assertEqual(set(self.root.iterdir()), {
            self.root / f'sha256-{sha256(ident.encode("utf-8")).hexdigest()}.receipt' for ident in ids
        })
        self.assertEqual([outcome.call_id for outcome in outcomes], list(ids))
        self.assertEqual([reopened.receipt_sequence(call) for call in calls], list(range(1, len(ids) + 1)))
        reused = OperationCall(calls[0].name, {'different': True}, call_id=ids[0])
        with self.assertRaisesRegex(ValueError, 'already used for a different operation'):
            reopened.execute(reused)
        self.assertEqual(reader.calls, [])

    def test_safe_legacy_terminal_replays_without_rewrite(self):
        group, lab = self.group()
        call = OperationCall('test.run', {}, call_id='模型 + v1 @ Lab')
        outcome = OperationResult('Legacy terminal', (), False, call_id=call.call_id)
        path = self.legacy_receipt(group, call, outcome)
        self.assertEqual(group.execute(call), outcome)
        self.assertEqual(group.receipt_elapsed(call), 0.5)
        self.assertEqual(lab.calls, [])
        self.assertEqual(set(self.root.iterdir()), {path})

    def test_legacy_job_terminal_update_prefers_canonical_receipt(self):
        job = JobRef('job-1', 'fixture')
        group, lab = self.group(job=job)
        call = OperationCall('test.run', {}, call_id='Legacy-Job.1')
        legacy_path = self.legacy_receipt(group, call, job)
        legacy = legacy_path.read_bytes()
        self.assertEqual(group.execute(call), job)
        result, _ = group.poll_job(call, job, decision_id='decision-1', after=0)
        reopened, reader = self.group()
        self.assertEqual(reopened.execute(call), result)
        self.assertEqual(reopened.receipt_sequence(call), 2)
        self.assertEqual(reopened.receipt_elapsed(call), 0.5)
        self.assertEqual(legacy_path.read_bytes(), legacy)
        self.assertEqual(lab.calls, [])
        self.assertEqual(reader.calls, [])

    def test_legacy_job_requires_exact_filename_spelling(self):
        job = JobRef('job-1', 'fixture')
        group, _ = self.group(job=job)
        call = OperationCall('test.run', {}, call_id='Legacy-Case')
        self.legacy_receipt(group, call, job)
        other = OperationCall(call.name, call.arguments, call_id='legacy-case')
        if (self.root / 'legacy-case.json').exists():  # Case-insensitive filesystem
            with self.assertRaisesRegex(ValueError, 'filename does not match'):
                group.stored_receipt(other)
        else:
            self.assertIsNone(group.stored_receipt(other))
        self.assertEqual(group.stored_receipt(call), job)

    def test_invalid_canonical_job_identity_never_falls_back(self):
        job = JobRef('job-1', 'fixture')
        group, writer = self.group(job=job)
        call = OperationCall('test.run', {}, call_id='Check Job!')
        self.assertEqual(group.execute(call), job)
        [path] = self.root.glob('*.receipt')
        row = decode(json.loads(path.read_text(encoding='utf-8')))
        self.assertEqual(row['call_id'], call.call_id)
        self.legacy_receipt(group, call, job)
        path.write_text(json.dumps(encode({**row, 'call_id': 'Other Job!'})), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'call_id does not match'):
            group.execute(call)
        self.assertEqual(group._recovery_receipt_evidence(call), {
            'receipt': 'unreadable_or_invalid', 'receipt_error_type': 'ValueError',
        })
        self.assertEqual(writer.calls, [call.call_id])
