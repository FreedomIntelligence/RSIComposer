from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeVar
from uuid import uuid4

from rsisys.loops.contracts.jobs import JobRef, JobStatus, JobOutcome






# These helpers manage job handles only. Loops own requests, results and learning policy.
T = TypeVar('T')


class JobHandle(Generic[T]):
    """An in-process handle; polling never resubmits work."""

    def __init__(self, job: JobRef, poll: Callable[[JobRef], JobOutcome[T]],
                 cancel: Callable[[JobRef], None]):
        self.job, self._poll, self._cancel = job, poll, cancel
        self.outcome: JobOutcome[T] | None = None
        self.cancel_requested = False

    def poll(self, *, normalize=None) -> JobOutcome[T]:
        if self.outcome is not None and self.outcome.terminal:
            return deepcopy(self.outcome)
        outcome = self._poll(self.job)
        if outcome.job != self.job:
            raise ValueError('Job outcome does not match the requested job')
        # A status-only poll (including cancellation/failure) does not erase the
        # last observation. Never synthesize steps or advance its sample time.
        if outcome.progress is None and self.outcome is not None:
            outcome = replace(outcome, progress=self.outcome.progress)
        if normalize is not None:
            outcome = normalize(outcome)
            if outcome.job != self.job:
                raise ValueError('Normalized outcome does not match the requested job')
        self.outcome = deepcopy(outcome)
        return deepcopy(outcome)

    def cancel(self, *, normalize=None) -> JobOutcome[T]:
        if self.outcome is not None and self.outcome.terminal:
            return deepcopy(self.outcome)
        if not self.cancel_requested:
            self._cancel(self.job)
            self.cancel_requested = True
        return self.poll(normalize=normalize)


@dataclass
class _MockJob(Generic[T]):
    request: object
    factory: Callable[[], T]
    remaining_polls: int
    outcome: JobOutcome[T]


class MockJobQueue(Generic[T]):
    """Deferred, deterministic fixtures. No threads, real training, or network calls."""

    def __init__(self, name: str, *, pending_polls: int = 1):
        if type(pending_polls) is not int or pending_polls < 0:
            raise ValueError('pending_polls must be a nonnegative integer')
        self.backend = f'{name}-{uuid4()}'
        self.pending_polls = pending_polls
        self._jobs: dict[str, _MockJob[T]] = {}
        self._keys: dict[str, JobRef] = {}

    def submit(self, request: object, factory: Callable[[], T], *, idempotency_key: str) -> JobRef:
        if not idempotency_key:
            raise ValueError('An idempotency key is required')
        if idempotency_key in self._keys:
            job = self._keys[idempotency_key]
            if self._jobs[job.job_id].request != request:
                raise ValueError('Idempotency key was reused with a different request')
            return job
        job = JobRef(str(uuid4()), self.backend)
        self._jobs[job.job_id] = _MockJob(
            deepcopy(request), factory, self.pending_polls,
            JobOutcome(job, JobStatus.QUEUED, simulated=True),
        )
        self._keys[idempotency_key] = job
        return job

    def _get(self, job: JobRef) -> _MockJob[T]:
        if job.backend != self.backend or job.job_id not in self._jobs:
            raise ValueError('Job does not belong to this backend')
        return self._jobs[job.job_id]

    def poll(self, job: JobRef) -> JobOutcome[T]:
        state = self._get(job)
        if state.outcome.terminal:
            return deepcopy(state.outcome)
        if state.remaining_polls:
            state.remaining_polls -= 1
            state.outcome = JobOutcome(job, JobStatus.RUNNING, simulated=True)
        else:
            try:
                state.outcome = JobOutcome(job, JobStatus.SUCCEEDED, state.factory(), simulated=True)
            except Exception as exc:
                state.outcome = JobOutcome(job, JobStatus.FAILED,
                                           error=f'{type(exc).__name__}: {exc}', simulated=True)
        return deepcopy(state.outcome)

    def cancel(self, job: JobRef) -> None:
        state = self._get(job)
        if not state.outcome.terminal:
            state.outcome = JobOutcome(job, JobStatus.CANCELLED, simulated=True)

    def result(self, job: JobRef) -> T:
        # Reading a result is deliberately not a poll or an execution trigger.
        outcome = self._get(job).outcome
        if outcome.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f'Job has no successful result: {outcome.status}')
        return deepcopy(outcome.result)


R = TypeVar('R')


class AsyncBackend(Protocol[R, T]):
    def submit(self, request: R, *, idempotency_key: str) -> JobRef: ...
    def poll(self, job: JobRef) -> JobOutcome[T]: ...
    def cancel(self, job: JobRef) -> None: ...


class JobBook(Generic[R, T]):
    """Per-owner job bookkeeping with explicit polling and request deduplication."""

    def __init__(self, backend: AsyncBackend[R, T] | None):
        self.backend = backend
        self._requests: dict[JobRef, R] = {}
        self._handles: dict[JobRef, JobHandle[T]] = {}
        self._keys: dict[str, JobRef] = {}

    @property
    def busy(self) -> bool:
        return any(h.outcome is None or not h.outcome.terminal for h in self._handles.values())

    def submit(self, request: R, *, idempotency_key: str) -> JobRef:
        if self.backend is None:
            raise NotImplementedError('No backend was configured for this loop component')
        if not idempotency_key:
            raise ValueError('An idempotency key is required')
        if idempotency_key in self._keys:
            job = self._keys[idempotency_key]
            if self._requests[job] != request:
                raise ValueError('Idempotency key was reused with a different request')
            return job
        if self.busy:
            raise RuntimeError('This component already has an unfinished job')
        job = self.backend.submit(deepcopy(request), idempotency_key=idempotency_key)
        if job in self._handles:
            raise ValueError('Backend reused a job reference for another request')
        self._requests[job] = deepcopy(request)
        self._handles[job] = JobHandle(job, self.backend.poll, self.backend.cancel)
        self._keys[idempotency_key] = job
        return job

    def request(self, job: JobRef) -> R:
        if job not in self._requests:
            raise ValueError('Job is not owned by this loop component')
        return deepcopy(self._requests[job])

    def poll(self, job: JobRef, *, normalize=None) -> JobOutcome[T]:
        """Normalize domain results before caching; persistence remains retryable."""
        self.request(job)
        return self._handles[job].poll(normalize=normalize)

    def cancel(self, job: JobRef, *, normalize=None) -> JobOutcome[T]:
        self.request(job)
        return self._handles[job].cancel(normalize=normalize)
