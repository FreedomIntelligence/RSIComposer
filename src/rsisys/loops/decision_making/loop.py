"""Decision loop lifecycle, composed from an independent harness and action space."""
from rsisys.loops.contracts.coordination import LoopStepResult
from rsisys.loops.contracts.harness import ControllerHarness, DecisionRunResult, NullHarness
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.decision_making.action_space import DecisionActionSpace

from .action_space.access import authorized_space


class DecisionMakingLoop:
    def __init__(self, policy, harness: ControllerHarness, action_space: DecisionActionSpace,
                 *, host_execution, model=None, model_scope=None, lab_execution=None):
        harness = NullHarness() if harness is None else harness
        self.policy, self.harness, self.action_space = policy, harness, action_space
        self.integration = None
        self.host_execution = host_execution
        self.model_binding, self.model_scope = model, model_scope
        self.operation_router = _ControllerToolRouter(harness, lab_execution)

    def current_model(self):
        from rsisys.loops.contracts.model_service import ModelRef
        declared = getattr(self.policy, 'model', None)
        if declared is not None and not isinstance(declared, ModelRef):
            raise TypeError('Policy attribute `model` is reserved for the decision model identity and '
                            'must be a ModelRef; rename the attribute if it holds something else')
        if declared is not None and self.model_binding is not None and declared != self.model_binding:
            raise ValueError('Policy model identity differs from endpoint binding')
        return declared if declared is not None else self.model_binding

    def bind_model(self, model):
        update = getattr(self.policy, 'update_model', None)
        if callable(update):
            update(model)
        self.model_binding = model

    def observed_decisions(self):
        query = getattr(self.host_execution, "observed_decisions", None)
        return query() if callable(query) else ()

    def step(self, objective: str):
        space = authorized_space(self.action_space, 'decision')
        return self._step(objective, space)

    def _step(self, objective, space):
        result = self.host_execution.step(objective, space)
        if not isinstance(result, LoopStepResult):
            raise TypeError('Host execution step must return LoopStepResult')
        result.__post_init__()
        return result

    def step_guarded(self, objective, before_action):
        return self._step(objective, _GuardedActionSpace(
            authorized_space(self.action_space, 'decision'), before_action))


    def pending_records(self):
        pending = getattr(self.host_execution, 'pending_records', None)
        return pending() if callable(pending) else False

    def flush_pending_records(self):
        flush = getattr(self.host_execution, 'flush_pending_records', None)
        if callable(flush):
            flush()

    def restart_research(self, context: dict[str, object]) -> None:
        restart = getattr(self.host_execution, 'restart_research', None)
        if not callable(restart):
            raise NotImplementedError('This decision harness does not support restart_research')
        return restart(context)

    def run_control_capabilities(self):
        """Advertise underlying optional hooks, rather than wrapper methods."""
        return frozenset(name for name in ('restart_research', 'update_run_control')
                         if callable(getattr(self.host_execution, name, None)))

    def update_run_control(self, context: dict[str, object]) -> None:
        update = getattr(self.host_execution, 'update_run_control', None)
        if not callable(update):
            raise NotImplementedError('This decision harness does not support update_run_control')
        return update(context)

    def pending_research_job(self) -> JobRef | None:
        pending = getattr(self.host_execution, 'pending_research_job', None)
        return pending() if callable(pending) else None

    def poll_research(self):
        poll = getattr(self.host_execution, 'poll_research', None)
        return poll() if callable(poll) else None

    def drain_research(self):
        drain = getattr(self.host_execution, 'drain_research', None)
        return drain() if callable(drain) else None

    def cancel_research(self):
        """Optional idempotent inference cancellation; does not cancel arbitrary tools/jobs."""
        cancel = getattr(self.host_execution, 'cancel_research', None)
        return cancel() if callable(cancel) else False

    def drain_operations(self, mode):
        drain = getattr(self.host_execution, 'drain_operations', None)
        return drain(authorized_space(self.action_space, 'decision'), mode) if callable(drain) else None

    def in_flight_operations(self):
        pending = getattr(self.host_execution, 'in_flight_operations', None)
        return pending() if callable(pending) else ()

    def drain_operation_plan(self, plan):
        drain = getattr(self.host_execution, 'drain_operation_plan', None)
        if not callable(drain):
            raise TypeError('Controller endpoint does not support planned operation draining')
        return drain(authorized_space(self.action_space, 'decision'), plan)

    def can_poll_job(self, job):
        pollable = getattr(self.host_execution, 'can_poll_job', None)
        return pollable(job) if callable(pollable) else False





    def run(self, objective: str, *, max_steps: int = 16) -> DecisionRunResult:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        steps = []
        for _ in range(max_steps):
            result = self.step(objective)
            steps.append(result)
            if result.kind != "progressed":
                return DecisionRunResult(result.kind, tuple(steps))
        return DecisionRunResult("step_limit", tuple(steps))


class _GuardedActionSpace:
    """Preserve capability authorization while checking an external action boundary."""
    def __init__(self, source, before_action):
        self.source, self.before_action = source, before_action

    @property
    def permissions(self):
        return self.source.permissions

    def __getattr__(self, name):
        return getattr(self.source, name)

    def validate(self, call):
        self.before_action()
        return self.source.validate(call)


class _ControllerToolRouter:
    """Dispatch Lab operations and Controller local tools through explicit ports."""
    def __init__(self, harness, lab):
        self.harness, self.lab = harness, lab

    def execute(self, call):
        from rsisys.loops.contracts.execution import LabToolError
        from rsisys.loops.contracts.coordination import OperationResult
        from rsisys.loops.contracts.lifecycle import ExecutionStatus
        if self.lab is not None and call.name in {spec.name for spec in self.lab.describe()}:
            try:
                return self.lab.execute(call)
            except LabToolError as error:
                return OperationResult('Lab tool failed; use this feedback to choose the next action', (),
                    False, ExecutionStatus.FAILED, f'{type(error).__name__}: {error}', data=error.data)
        return self.harness.execute_tool(call)

    def retry(self, call, *, after_sequence=None):
        if self.lab is None or not callable(getattr(self.lab, 'retry', None)):
            raise ValueError('Lab does not provide explicit infrastructure retry')
        return self.lab.retry(call, after_sequence=after_sequence)
