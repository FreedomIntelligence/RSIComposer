"""Evaluation protocol/admission tests; no model, data service or GPU required."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsisys.loops.contracts.evaluation import EvaluationProtocol
from rsisys.runtime.evaluation import EvaluationLedger, EvaluationAdmissionError


class EvaluationProtocolTests(unittest.TestCase):
    def test_rolling_window_failure_replay_and_restart_share_campaign_quota(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'campaign.json'
            now = [1000]
            ledger = EvaluationLedger(path, 'campaign', clock=lambda: now[0])
            first = ledger.reserve('first', 'identity', 'closed')
            self.assertFalse(first['replayed'])
            # No job attachment: an executor crash after acceptance still consumes the slot.
            restarted = EvaluationLedger(path, 'campaign', clock=lambda: now[0])
            self.assertTrue(restarted.reserve('first', 'identity', 'closed')['replayed'])
            with self.assertRaises(EvaluationAdmissionError):
                restarted.reserve('first', 'changed', 'closed')
            for instant in (999, 1000, 2799.999):
                now[0] = instant
                with self.subTest(now=instant), self.assertRaises(EvaluationAdmissionError) as error:
                    restarted.reserve('second', 'new-candidate', 'closed')
                self.assertEqual(error.exception.data['next_allowed_at'], 2800)
            now[0] = 2800
            self.assertFalse(restarted.reserve('second', 'new-candidate', 'closed')['replayed'])
            restarted.attach('second', 'job-2')
            restarted.attach('second', 'job-2')
            with self.assertRaises(EvaluationAdmissionError):
                restarted.attach('second', 'other-job')
            self.assertEqual(len(json.loads(path.read_text())['submissions']), 2)

    def test_concurrent_candidates_cannot_both_take_the_same_slot(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'campaign.json'
            def submit(index):
                try:
                    EvaluationLedger(path, 'same-campaign', clock=lambda: 1000).reserve(str(index), str(index), 'closed')
                    return True
                except EvaluationAdmissionError:
                    return False
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(submit, range(16)))
            self.assertEqual(sum(results), 1)

    def test_public_evaluation_does_not_consume_or_reset_closed_allowance(self):
        with TemporaryDirectory() as directory:
            ledger = EvaluationLedger(Path(directory) / 'campaign.json', 'campaign', clock=lambda: 1000)
            for key in ('public-1', 'public-2'):
                ledger.reserve(key, key, 'public')
            ledger.reserve('closed-1', 'closed', 'closed')
            ledger.reserve('public-3', 'public', 'public')
            with self.assertRaises(EvaluationAdmissionError):
                ledger.reserve('closed-2', 'closed', 'closed')
            with self.assertRaises(ValueError):
                EvaluationLedger(ledger.path, 'changed-account').lookup('closed-1')
            with self.assertRaises(ValueError):
                EvaluationProtocol(closed_interval_seconds=1)
