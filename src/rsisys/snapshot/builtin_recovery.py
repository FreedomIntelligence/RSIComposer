"""Recovery adapter for the bundled decision/Actor harness pair and mock jobs."""
from time import monotonic

from rsisys.runtime.state.recovery import RecoveryRequired


def decision_extension(harness):
    from rsisys.harnesses.decision.engine import DefaultDecisionHarness
    if type(harness) is DefaultDecisionHarness:
        return False
    return (isinstance(harness, DefaultDecisionHarness)
            and isinstance(getattr(harness, 'recovery_id', None), str) and bool(harness.recovery_id)
            and callable(getattr(harness, 'export_recovery_state', None))
            and callable(getattr(harness, 'restore_recovery_state', None)))


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


class BuiltinRecovery:
    contract_id = 'rsisys-local/v1'

    def capture(self, app):
        from rsisys.snapshot.local import (
            research_configuration,
            validate_local_application,
        )
        dm, actor, _, permissions = validate_local_application(app, ongoing=True)
        if app.sandbox.active_count or actor.state.active_directive_id is not None:
            raise RecoveryRequired('An active environment/process needs reconciliation; it cannot be restored')
        extension = ({'id': dm.recovery_id, 'state': dm.export_recovery_state()}
                     if decision_extension(dm) else None)
        return {'decision': dm.state, 'actor': actor.state,
                'decision_extension': extension,
                'research_agenda': dm.research_agenda.export(),
                'context_compactor': research_configuration(dm),
                'models': (dm.policy.model, actor.policy.model),
                'limits': (dm.experience_limit, dm.memory_limit, dm.review_tool_limit, actor.memory_limit),
                'permissions': permissions,
                'commands': {'active': actor.commands.active, 'receipts': actor.commands.receipts},
                'command_clock': monotonic(),
                'books': [_capture_book(book) for book in (dm.training, actor.training, actor.evaluation)]}

    def restore(self, app, state):
        from rsisys.loops.decision_making.action_space.access import ActionPermissions
        from rsisys.snapshot.local import restore_research, validate_local_application
        dm, actor, _, _ = validate_local_application(app, ongoing=True)
        extension = state['decision_extension']
        if (extension is None) != (not decision_extension(dm)) or (
                extension is not None and extension['id'] != dm.recovery_id):
            raise TypeError('Decision recovery_id differs; supply the original decision_factory')
        dm.state, actor.state = state['decision'], state['actor']
        restore_research(dm, state['research_agenda'], state['context_compactor'])
        dm.policy.model, actor.policy.model = state['models']
        dm.experience_limit, dm.memory_limit, dm.review_tool_limit, actor.memory_limit = state['limits']
        app.runner.decision_making.action_space.permissions = ActionPermissions.from_dict(state['permissions'])
        actor.commands.active = state['commands']['active']
        actor.commands.receipts = state['commands']['receipts']
        for receipt in actor.commands.receipts.values():
            if receipt.started_at is not None:
                receipt.started_at = monotonic() + receipt.started_at - state['command_clock'] - app.recovery.downtime
        for book, saved in zip((dm.training, actor.training, actor.evaluation), state['books']):
            _restore_book(book, saved)
        if extension is not None:
            dm.restore_recovery_state(extension['state'])
