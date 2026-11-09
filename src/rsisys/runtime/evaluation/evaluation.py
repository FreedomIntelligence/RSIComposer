"""Durable campaign admission for evaluation; never a data isolation boundary."""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import time

from rsisys.loops.contracts.evaluation import EvaluationProtocol
from rsisys.runtime.storage.files import atomic_write, json_bytes


class EvaluationAdmissionError(ValueError):
    def __init__(self, reason, **data):
        super().__init__(reason)
        self.data = {'reason': reason, **data}


class EvaluationLedger:
    """Shared by all candidates/runs of a host-assigned campaign.

    A reservation is durable before an executor is invoked. Failure consumes it;
    recovery reconciles that same key instead of creating another submission.
    This file belongs outside the research executor's writable filesystem.
    """
    def __init__(self, path, campaign_id, *, protocol=None, clock=time.time):
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise ValueError('An operator-assigned evaluation campaign is required')
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.protocol = protocol or EvaluationProtocol()
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(self.path.suffix + '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            identity = {'campaign_id': self.campaign_id, 'protocol': asdict(self.protocol)}
            if self.path.exists():
                state = json.loads(self.path.read_text())
                if state['identity'] != identity:
                    raise ValueError('Evaluation ledger identity changed')
            else:
                state = {'identity': identity, 'last_clock': 0, 'submissions': {}}
            yield state

    def lookup(self, key):
        with self._locked() as state:
            row = state['submissions'].get(key)
            return json.loads(json.dumps(row)) if row is not None else None

    def reserve(self, key, identity, split):
        if split not in ('public', 'closed') or not isinstance(key, str) or not key.strip():
            raise ValueError('Invalid evaluation key/split')
        with self._locked() as state:
            previous = state['submissions'].get(key)
            if previous is not None:
                if previous['identity'] != identity or previous['split'] != split:
                    raise EvaluationAdmissionError('evaluation_key_conflict')
                return {**previous, 'replayed': True}
            now = self.clock()
            if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
                raise ValueError('Invalid host clock')
            now = max(now, state['last_clock'])  # Clock rollback cannot grant an extra slot.
            closed = [r['accepted_at'] for r in state['submissions'].values() if r['split'] == 'closed']
            next_at = max(closed) + self.protocol.closed_interval_seconds if closed else now
            if split == 'closed' and now < next_at:
                raise EvaluationAdmissionError('closed_evaluation_rate_limited', next_allowed_at=next_at,
                                               retry_after_seconds=next_at - now)
            row = {'identity': identity, 'split': split, 'accepted_at': now, 'job_id': None}
            state['submissions'][key] = row
            state['last_clock'] = now
            atomic_write(self.path, json_bytes(state))
            return {**row, 'replayed': False}

    def attach(self, key, job_id):
        with self._locked() as state:
            row = state['submissions'][key]
            if row['job_id'] is not None and row['job_id'] != job_id:
                raise EvaluationAdmissionError('evaluation_job_identity_conflict')
            row['job_id'] = job_id
            atomic_write(self.path, json_bytes(state))
