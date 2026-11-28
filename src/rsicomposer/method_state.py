"""Recovery state for whole methods built from a policy factory and named Labs.

A method is durable when its policy and every Lab implement the StatefulPolicy /
StatefulLab ports: state() returns detached data supported by the state codec
(JSON values plus the framework's registered contract types) and restore(state)
returns None. The host never inspects other attributes of a method's objects.
Factories run again on resume; context.restoring tells them the workspace exists.
Concurrent Labs must explicitly opt in to capture during execution, and every
capturable state must be valid even if their overlapping calls are interrupted.
"""
from copy import deepcopy
import json
from typing import Protocol

from rsicomposer.runtime.capabilities import optional_method


class StatefulPolicy(Protocol):
    def state(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class StatefulLab(Protocol):
    def state(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


def _ports(value, role):
    capture, restore = optional_method(value, 'state'), optional_method(value, 'restore')
    if capture is None or restore is None:
        raise TypeError(f'{role} {type(value).__qualname__} must implement state() and restore(state) '
                        'for durable runs and snapshots')
    return capture, restore


def _captured(value, role):
    state = _ports(value, role)[0]()
    if not isinstance(state, dict):
        raise TypeError(f'{role} state() must return a dict')
    return deepcopy(state)


def _restored(value, role, state):
    if _ports(value, role)[1](deepcopy(state)) is not None:
        raise TypeError(f'{role} restore(state) must return None')


class PolicyDecisionState:
    """One version's policy, Harness and evidence-backed Memory projection."""
    state_id = 'rsicomposer-method-decision/v9'

    def __init__(self, endpoint):
        self.endpoint, self.harness = endpoint, endpoint.harness

    @property
    def policy(self):
        from rsicomposer.harnesses.decision.public_policy import ResearchPolicyAdapter
        policy = self.endpoint.policy
        return policy.implementation if isinstance(policy, ResearchPolicyAdapter) else policy

    def capture(self):
        return {'harness': _captured(self.harness, 'Controller Harness'),
                'policy': _captured(self.policy, 'Controller policy'),
                'integration': (self.endpoint.integration.state()
                                if self.endpoint.integration is not None else None)}

    def restore(self, state):
        if not isinstance(state, dict) or set(state) != {'harness', 'policy', 'integration'}:
            raise ValueError('Invalid Controller version state')
        integration = self.endpoint.integration
        if (integration is None) != (state['integration'] is None):
            raise ValueError('Controller integration differs from its saved state')
        _restored(self.policy, 'Controller policy', state['policy'])
        _restored(self.harness, 'Controller Harness', state['harness'])
        validate_records = optional_method(self.harness, 'validate_execution_records')
        if validate_records is not None:
            validate_records(self.endpoint.host_execution.state.decision_records.values())
        if integration is not None:
            integration.restore(deepcopy(state['integration']))

    def validate_snapshot(self):
        _ports(self.harness, 'Controller Harness')
        if self.endpoint.integration is not None:
            self.endpoint.integration.validate_snapshot()
        validate_harness = optional_method(self.harness, 'validate_snapshot')
        if validate_harness is not None and validate_harness() is not None:
            raise TypeError('Controller Harness validate_snapshot must return None')
        _ports(self.policy, 'Controller policy')
        validate = optional_method(self.policy, 'validate_snapshot')
        if validate is not None and validate() is not None:
            raise TypeError('Controller validate_snapshot must return None')


class HostExecutionStateAdapter:
    """Capture the fixed run execution owner once, outside Controller versions."""
    state_id = 'rsicomposer-host-execution/v3'

    def __init__(self, execution):
        self.execution = execution

    def capture(self):
        return deepcopy(self.execution.state)

    def restore(self, state):
        from rsicomposer.host.execution import HostExecutionState
        if not isinstance(state, HostExecutionState):
            raise TypeError('Invalid host execution state')
        saved = deepcopy(state)
        saved.__post_init__()
        self.execution.state = saved

    def validate_snapshot(self):
        state = self.execution.state
        if state.operation_tasks or state.pending_operation_record is not None or state.pending_job is not None:
            raise RuntimeError('Portable snapshots require drained host execution')


class LabGroupState:
    """Every Lab of a method, keyed by its alias."""
    state_id = 'rsicomposer-method-labs/v26'

    def __init__(self, group):
        self.group = group

    def capture(self):
        with self.group._lock:
            self._validate_state_ports()
            return {'labs': self.group.capture_lab_states(),
                    'receipts': self.group.receipt_state(), 'profiles': self.group.profile_state()}

    def export_snapshot_files(self, directory, state):
        self.group.export_receipt_files(directory, state['receipts'])

    def import_snapshot_files(self, directory, state):
        self.group.restore_receipt_files(directory, state['receipts'])

    def restore(self, state):
        labs = state['labs']
        if set(labs) != set(self.group.labs):
            raise ValueError('Saved Lab aliases differ from the assembled method')
        for alias, lab in self.group.labs.items():
            _restored(lab, f'Lab {alias}', labs[alias])
        self.group.restore_receipt_state(state['receipts'])
        self.group.restore_profile_state(state['profiles'])

    def _validate_state_ports(self):
        for alias, lab in self.group.labs.items():
            _ports(lab, f'Lab {alias}')

    def validate_snapshot(self):
        self._validate_state_ports()
        for alias, lab in self.group.labs.items():
            validate = optional_method(lab, 'validate_snapshot')
            if validate is not None and validate() is not None:
                raise TypeError(f'Lab {alias} validate_snapshot must return None')


def validate_feedback_journal(learning, journal):
    """Reject changes to a persisted processor's protocol or configuration."""
    identity = ('version', 'limits', 'runtime', 'assembly', 'processor')
    if (not isinstance(journal, dict)
            or json.dumps({key: journal.get(key) for key in identity}, sort_keys=True, allow_nan=False)
            != json.dumps({key: learning.state.get(key) for key in identity}, sort_keys=True, allow_nan=False)
            or not isinstance(journal.get('jobs'), dict)
            or type(journal.get('attempted')) is not int or journal['attempted'] < 0):
        raise ValueError('Invalid feedback journal or changed configuration')
    from rsicomposer.harnesses.decision.feedback.learning import validate_journal_captures
    validate_journal_captures(journal)


class FeedbackJournalState:
    """Full feedback journal for portable state and same-workspace recovery.

    A fresh store imports the captured journal. An existing workspace may be
    ahead because feedback is derived asynchronously; it must retain every
    captured job and at least the recorded attempted count.
    """
    state_id = 'rsicomposer-feedback-journal/v5'

    def __init__(self, service):
        self.service, self.learning = service, service.learning

    def capture(self):
        return {'journal': deepcopy(self.learning.state)}

    def restore(self, state):
        if not isinstance(state, dict) or set(state) != {'journal'}:
            raise ValueError('Invalid feedback journal checkpoint')
        saved = deepcopy(state['journal'])
        validate_feedback_journal(self.learning, saved)
        current = self.learning.state
        if self.learning.path is None:
            self.learning.state = saved
        elif not current['jobs'] and current['attempted'] == 0:
            self.learning._commit(saved)
        elif (not set(saved['jobs']) <= set(current['jobs'])
                or current['attempted'] < saved['attempted']):
            from rsicomposer.runtime.state.recovery import RecoveryRequired
            raise RecoveryRequired('The feedback journal is older than the checkpoint')

    def validate_snapshot(self):
        self.service.validate_snapshot()


class ResearchMemoryState:
    """Export memory state, or rebuild it from the host's ordered observation log."""
    state_id = 'rsicomposer-research-memory/v4'

    def __init__(self, memory, *, experience=None, controller_version=None, resuming=False):
        self.memory, self.experience = memory, experience
        self.controller_version = controller_version
        self.rebuild = optional_method(memory, 'rebuild')
        self.restore_checkpoint = optional_method(memory, 'restore_checkpoint') if resuming else None

    def can_archive(self):
        if self.rebuild is not None:
            return self.experience is not None
        return all(optional_method(self.memory, name) is not None for name in ('export', 'restore'))

    def _events(self):
        from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
        from rsicomposer.runtime.storage.codec import decode
        return [(row.record_id, EvidenceEvent(**decode(deepcopy(row.details['event']))))
                for row in self.experience.retrieve(kinds=('memory_observation',))
                if self.controller_version is None
                or row.details.get('controller_version') == self.controller_version]

    def observe(self, event):
        # Write ahead of delivery: an observer can fail or the process can die
        # during delivery. rebuild() receives the full ordered input stream.
        from dataclasses import fields
        from uuid import uuid4
        from rsicomposer.runtime.storage.codec import encode
        from rsicomposer.runtime.evidence.records import ExperienceRecord
        self.experience.record(ExperienceRecord(str(uuid4()), 'memory_observation', 'decision_maker',
            None, 'Controller memory observation', False, details={
                'controller_version': self.controller_version,
                'event': encode({item.name: deepcopy(getattr(event, item.name)) for item in fields(event)})}))
        observer = optional_method(self.memory, 'observe')
        if observer is not None and observer(deepcopy(event)) is not None:
            raise TypeError('Controller memory observe(event) must return None')

    def capture(self):
        if self.rebuild is not None:
            if self.experience is None:
                raise TypeError('Controller memory rebuild(events) requires the host evidence store')
            return {'events': [key for key, _ in self._events()]}
        export, restore = optional_method(self.memory, 'export'), optional_method(self.memory, 'restore')
        if export is None or restore is None:
            raise TypeError(f'Controller memory {type(self.memory).__qualname__} must implement export() '
                            'and restore(state), or rebuild(events), for durable runs and snapshots')
        return {'memory': deepcopy(export())}

    def restore(self, state):
        self.capture()  # validate the chosen capability before mutating memory
        if self.rebuild is not None:
            events = self._events()
            ids = [key for key, _ in events]
            if ids[:len(state['events'])] != state['events']:
                from rsicomposer.runtime.state.recovery import RecoveryRequired
                raise RecoveryRequired('Controller memory evidence is older than or differs from the checkpoint')
            # The evidence store may be ahead of the checkpoint. Include those
            # deliveries because the feedback journal will not deliver them again.
            result = self.rebuild(tuple(event for _, event in events))
        else:
            restore = self.restore_checkpoint or self.memory.restore
            result = restore(deepcopy(state['memory']))
        if result is not None:
            raise TypeError('Controller memory restore/rebuild must return None')

    def validate_snapshot(self):
        self.capture()


def method_state_factory(context, decision):
    if getattr(context, 'controller_host', None) is not None:
        from rsicomposer.host.lifecycle import ControllerHostState
        return {'lab_execution': LabGroupState(context.lab_execution),
                'execution': HostExecutionStateAdapter(context.host_execution),
                'decision': ControllerHostState(context.controller_host)}
    return {'lab_execution': LabGroupState(context.lab_execution),
            'execution': HostExecutionStateAdapter(decision.host_execution),
            'decision': PolicyDecisionState(decision)}
