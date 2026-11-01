"""Decision loop lifecycle, composed from an independent harness and action space."""
from .action_space import DecisionActionSpace
from .action_space.access import authorized_space
from .harness import DecisionHarness
from .harness.decision import DecisionRunResult
from rsisys.loops.contracts.coordination import ReviewDecision


class DecisionMakingLoop:
    def __init__(self, harness: DecisionHarness, action_space: DecisionActionSpace):
        self.harness, self.action_space = harness, action_space

    def step(self, objective: str):
        return self.harness.step(objective, self.action_space)

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
            return with_tools(objective, progress, self.action_space)
        review = getattr(self.harness, 'review', None)
        return review(objective, progress) if review is not None else ReviewDecision()

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
