"""Decision loop lifecycle, composed from an independent harness and action space."""
from .action_space import DecisionActionSpace
from .harness import DecisionHarness
from .harness.decision import DecisionRunResult
from rsisys.loops.contracts.coordination import ReviewDecision


class DecisionMakingLoop:
    def __init__(self, harness: DecisionHarness, action_space: DecisionActionSpace):
        self.harness, self.action_space = harness, action_space

    def step(self, objective: str):
        return self.harness.step(objective, self.action_space)

    def pending(self):
        return self.harness.pending()

    def receive_feedback(self, directive, feedback):
        self.harness.receive_feedback(directive, feedback)

    def update_execution(self, view):
        update = getattr(self.harness, 'update_execution', None)
        if update is not None:
            update(view)

    def review(self, objective, progress):
        review = getattr(self.harness, 'review', None)
        return review(objective, progress) if review is not None else ReviewDecision()

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
