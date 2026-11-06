"""State ports for this explicit Open Lab assembly, not implicit plugin introspection.

Same-directory restart only. The host keeps checkpoints, source, data, memory DB
and accelerator/service bindings together. Unknown external effects stay blocked.
Training/optimizer checkpoints remain the responsibility of generated Lab code.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from rsisys.adapters.local_lab_state import LocalLabState
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.recovery import DurableRecovery, RecoveryRequired


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def example_identity():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('open_lab_experiment.py', 'open_lab_recovery.py', 'fixed_evaluation.py',
                     'candidate_adapter.py', 'checkpoint_evaluation.py', 'protocol_evaluation.py')}


@contextmanager
def controller_lock(root, *, resume):
    """The OS lock is authoritative; a historical PID claim is never a lease."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This Open Lab controller is already running') from None
        if not resume and (root / 'controller.claim').exists():
            raise ValueError('Run already exists; use --resume with its saved launch configuration')
        yield


def inspect_recovery(root):
    """Read metadata without constructing modules, contacting models or mutating state."""
    root = Path(root)
    state = DurableRecovery.read(root / 'application')
    journal = json.loads((root / 'application/recovery/journal.json').read_text())
    executor = state['components']['lab_execution']['state']['backend']
    from rsisys.adapters.local_lab_state import _read
    live = _read(root / 'application/allocated-workspaces/process-recovery.json')
    receipts = root / 'application/allocated-workspaces/process-receipts'
    return {'run_id': state['run_id'], 'generation': state['generation'],
        'saved_at': state['saved_at'], 'intervention': state['intervention'],
        'uncommitted_mutation': journal['generation'] > state['generation'],
        'journal': journal, 'owned_processes': [{
            **{k: v for k, v in row.items() if k != 'request'}, 'job_id': ident,
            'in_checkpoint': ident in executor['processes'],
            'terminal_receipt_present': (receipts / (ident + '.json')).is_file()}
            for ident, row in live['processes'].items()],
        'deadline_epoch': json.loads((root / 'recovery-launch.json').read_text())['arguments']['deadline_epoch']}


class DecisionState:
    state_id = 'medical-open-lab-decision/v1'

    def __init__(self, loop):
        self.loop = loop
        dm = loop.harness
        if dm.learner is not None or dm.training._requests:
            raise TypeError('This integration uses Lab-owned training, not built-in decision training')
        # Initialize the durable files before the observational state port runs.
        if dm.memory.path is None:
            raise TypeError('Open Lab recovery needs persistent decision memory')
        if not dm.memory.path.exists():
            dm.memory.backup(dm.memory.path)

    def configuration(self):
        dm = self.loop.harness
        compactor = dm.context_compactor
        return {'model': asdict(dm.current_model()), 'policy': type(dm.policy).__qualname__,
            'permissions': self.loop.action_space.permissions.to_dict(),
            'limits': [dm.experience_limit, dm.memory_limit, dm.review_tool_limit, dm.max_outstanding],
            'compactor': {key: getattr(compactor, key) for key in ('trigger_tokens', 'target_tokens',
                'summary_tokens', 'max_target_tokens', 'target_increment_tokens')},
            'project': dm.policy.project_context}

    def capture(self):
        dm = self.loop.harness
        return {'state': deepcopy(dm.state), 'memory_digest': digest(dm.memory.export()),
            'agenda': dm.research_agenda.export(), 'compactor': dm.context_compactor.export_state()}

    def restore(self, state):
        dm = self.loop.harness
        if digest(dm.memory.export()) != state['memory_digest']:
            raise RecoveryRequired('Decision memory changed after checkpoint; reconcile the interrupted boundary')
        if dm.research_agenda.export() != state['agenda']:
            raise RecoveryRequired('Research agenda changed after checkpoint; reconcile before continuing')
        dm.state = deepcopy(state['state'])
        dm.research_agenda.restore(state['agenda'])
        dm.context_compactor.restore_state(state['compactor'])


class OperationOnlyState:
    state_id = 'medical-operation-only-exploration/v1'

    def capture(self):
        return None

    def restore(self, state):
        if state is not None:
            raise ValueError('Unexpected exploration state')


class FeedbackState:
    state_id = 'medical-feedback-journal/v1'

    def __init__(self, service):
        self.service = service
        learning = service.learning
        if not learning.path.exists():
            learning._commit(learning.state)

    def configuration(self):
        learning = self.service.learning
        return {'path': str(learning.path), 'limits': learning.limits,
                'assembly': learning.state.get('assembly'), 'processor': learning.state.get('processor')}

    def capture(self):
        state = self.service.learning.state
        # The journal and LangMem DB are authoritative for late worker commits.
        # Restore verifies provenance and reservations; never rolls them back.
        return {'attempted': state['attempted'],
            'receipts': {key: row['receipt_digest'] for key, row in state['jobs'].items()}}

    def restore(self, state):
        current = self.service.learning.state
        if (current['attempted'] < state['attempted'] or any(
                key not in current['jobs'] or current['jobs'][key]['receipt_digest'] != value
                for key, value in state['receipts'].items())):
            raise ValueError('Feedback journal lost reservations or changed receipt identity')


class BackendState:
    state_id = 'medical-recorded-backend/v1'

    def __init__(self, backend):
        self.backend = backend

    def configuration(self):
        return {'max_calls': self.backend.max_calls, 'max_input_tokens': self.backend.max_input_tokens}

    def capture(self):
        with self.backend.lock:
            return {'calls': self.backend.calls}

    def restore(self, state):
        if self.backend.calls < state['calls']:
            raise ValueError('Model call ledger is behind the recovery checkpoint')


def state_ports(context, decision, exploration, *, lab, backend):
    from rsisys.adapters.execution import OperationOnlyExploration
    if type(exploration) is not OperationOnlyExploration:
        raise TypeError('Supply a state adapter for the replacement exploration module')
    ports = {'decision': DecisionState(decision), 'exploration': OperationOnlyState(),
        'feedback': FeedbackState(decision.feedback_service), 'backend': BackendState(backend),
        'lab_execution': LocalLabState(lab, capture_extra=lambda: deepcopy(lab._evaluations),
            restore_extra=lambda state: setattr(lab, '_evaluations', deepcopy(state)),
            configuration={'evaluator_sha256': lab.evaluator_sha256,
                'adapter_sha256': lab.adapter_sha256, 'require_input_contract': lab.require_input_contract,
                'groups': lab.evaluation_groups, 'python': lab.evaluation_python,
                'formal_evaluation': getattr(lab, 'formal_configuration', None)})}
    memory = getattr(decision.harness, 'agent_memory', None)
    if memory is not None:
        from rsisys.loops.decision_making.harness.agent_memory import AgentMemoryState
        ports['agent_memory'] = AgentMemoryState(memory)
    return ports


def pause_at_boundary(app, *, timeout=40):
    """Cancel owned programs, settle feedback, then commit a resumable boundary.

    Keep Runner's research state and budgets; pause is not a terminal run stop.
    Never clear an uncertain journal entry to make a failed pause resumable.
    """
    def settle():
        lab = app.assembly.lab_execution
        until = time.monotonic() + timeout
        while not lab.shutdown('cancel'):
            if time.monotonic() >= until:
                raise TimeoutError('Cannot checkpoint pause before owned programs exit')
            time.sleep(.02)
        app.runner.decision_making.feedback_service.pause(timeout=max(0, until - time.monotonic()))
        # Same terminal job IDs remain collectable; later actions get new keys.
        native = getattr(lab, 'lab', lab)
        native._accepting = True
    app.recovery.mutate('pause', (), settle)
