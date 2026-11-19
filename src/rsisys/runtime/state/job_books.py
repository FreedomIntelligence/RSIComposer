"""Explicit recovery of Controller mock training job books."""
def _capture_book(book):
    from rsisys.runtime.orchestration.jobs import JobBook, MockJobQueue
    if type(book) is not JobBook:
        raise TypeError('Custom job books need an explicit recovery contract')
    queue = None
    if book.backend is not None:
        q = book.backend.jobs
        if type(q) is not MockJobQueue:
            raise TypeError('Only built-in mock jobs can be reconstructed locally')
        queue = {'backend': q.backend, 'pending_polls': q.pending_polls, 'keys': q._keys,
                 'jobs': {key: (value.request, value.remaining_polls, value.outcome)
                          for key, value in q._jobs.items()}}
    return {'requests': book._requests, 'keys': book._keys,
            'handles': {job: (handle.outcome, handle.cancel_requested) for job, handle in book._handles.items()},
            'queue': queue}


def _restore_book(book, state):
    from rsisys.runtime.orchestration.jobs import JobHandle, _MockJob
    queue = state['queue']
    if queue is not None:
        q = book.backend.jobs
        # Recreate only the known mock's deferred closure in an isolated scratch queue.
        # submit here cannot execute a factory, launch a process, or contact a service.
        scratch = type(book.backend)(book.backend.artifacts)
        for key, (request, polls, outcome) in queue['jobs'].items():
            temporary = scratch.submit(request, idempotency_key=key)
            factory = scratch.jobs._jobs[temporary.job_id].factory
            q._jobs[key] = _MockJob(request, factory, polls, outcome)
        q.backend, q.pending_polls, q._keys = queue['backend'], queue['pending_polls'], queue['keys']
    book._requests, book._keys = state['requests'], state['keys']
    book._handles = {}
    for job, (outcome, cancelled) in state['handles'].items():
        handle = JobHandle(job, book.backend.poll, book.backend.cancel)
        handle.outcome, handle.cancel_requested = outcome, cancelled
        book._handles[job] = handle

