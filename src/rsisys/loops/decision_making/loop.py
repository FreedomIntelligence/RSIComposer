"""Decision loop lifecycle, composed from an independent harness and action space."""
from .action_space import DecisionActionSpace
from .action_space.access import authorized_space
from .harness import DecisionHarness
from .harness.decision import DecisionRunResult
from rsisys.loops.contracts.coordination import ReviewDecision, LoopStepResult
from rsisys.loops.contracts.jobs import JobRef


class DecisionMakingLoop:
    def __init__(self, harness: DecisionHarness, action_space: DecisionActionSpace):
        self.harness, self.action_space = harness, action_space

    def step(self, objective: str):
        space = authorized_space(self.action_space, 'decision')
        return self._step(objective, space)

    def _step(self, objective, space):
        result = self.harness.step(objective, space)
        if not isinstance(result, LoopStepResult):
            raise TypeError('Decision harness step must return LoopStepResult')
        result.__post_init__()
        if result.kind == 'dispatch':
            # A plugin's internal checks never replace host admission. Recheck
            # the current permission even if it skipped validation itself.
            space.validate_directive(result.directive)
        return result

    def step_guarded(self, objective, before_action):
        return self._step(objective, _GuardedActionSpace(
            authorized_space(self.action_space, 'decision'), before_action))

    def review_guarded(self, objective, progress, before_action):
        with_tools = getattr(self.harness, 'review_with_tools', None)
        if callable(with_tools):
            result = with_tools(objective, progress, _GuardedActionSpace(
                authorized_space(self.action_space, 'review'), before_action))
        else:
            result = self.review(objective, progress)
        before_action()
        return self._review_result(result)

    def pending_records(self):
        pending = getattr(self.harness, 'pending_records', None)
        return pending() if callable(pending) else False

    def flush_pending_records(self):
        flush = getattr(self.harness, 'flush_pending_records', None)
        if callable(flush):
            flush()

    def restart_research(self, context: dict[str, object]) -> None:
        restart = getattr(self.harness, 'restart_research', None)
        if not callable(restart):
            raise NotImplementedError('This decision harness does not support restart_research')
        return restart(context)

    def run_control_capabilities(self):
        """Advertise underlying optional hooks, rather than wrapper methods."""
        return frozenset(name for name in ('restart_research', 'update_run_control')
                         if callable(getattr(self.harness, name, None)))

    def update_run_control(self, context: dict[str, object]) -> None:
        update = getattr(self.harness, 'update_run_control', None)
        if not callable(update):
            raise NotImplementedError('This decision harness does not support update_run_control')
        return update(context)

    def pending_research_job(self) -> JobRef | None:
        pending = getattr(self.harness, 'pending_research_job', None)
        return pending() if callable(pending) else None

    def cancel_research(self):
        """Optional idempotent inference cancellation; does not cancel arbitrary tools/jobs."""
        cancel = getattr(self.harness, 'cancel_research', None)
        return cancel() if callable(cancel) else False

    def can_step(self):
        ready = getattr(self.harness, 'can_step', None)
        return ready() if ready is not None else True

    def pending(self):
        return self.harness.pending()

    def pending_all(self):
        discover = getattr(self.harness, 'pending_all', None)
        if discover is not None:
            return discover()
        pending = self.pending()
        return () if pending is None else (pending,)

    def update_executions(self, views):
        update = getattr(self.harness, 'update_executions', None)
        if update is not None:
            update(views)

    def compare_candidates(self, candidate_ids, *, metric, higher_is_better=True):
        return self.harness.compare_candidates(candidate_ids, metric=metric, higher_is_better=higher_is_better)

    def receive_feedback(self, directive, feedback):
        self.harness.receive_feedback(directive, feedback)

    def update_execution(self, view):
        update = getattr(self.harness, 'update_execution', None)
        if update is not None:
            update(view)

    def review(self, objective, progress):
        with_tools = getattr(self.harness, 'review_with_tools', None)
        if callable(with_tools):
            result = with_tools(objective, progress, authorized_space(self.action_space, 'review'))
            return self._review_result(result)
        review = getattr(self.harness, 'review', None)
        result = review(objective, progress) if review is not None else ReviewDecision()
        return self._review_result(result)

    def _review_result(self, result):
        if not isinstance(result, ReviewDecision):
            raise TypeError('Decision harness review must return ReviewDecision')
        result.__post_init__()
        self.authorize_review(result)
        return result

    def authorize_review(self, result):
        """Recheck current access immediately before Runner applies a model control."""
        authorized_space(self.action_space, 'review').validate_review(result)

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

    def validate_directive(self, directive):
        self.before_action()
        return self.source.validate_directive(directive)

    def validate_review(self, result):
        self.before_action()
        return self.source.validate_review(result)
