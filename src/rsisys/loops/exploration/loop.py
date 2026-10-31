"""Receives system directives; Actor actions are a separate environment vocabulary."""
from .action_space import ActionSpace
from .harness import ExplorationHarness


class ExplorationLoop:
    def __init__(self, harness: ExplorationHarness, action_space: ActionSpace):
        self.harness, self.action_space = harness, action_space

    def accept(self, directive):
        return self.harness.accept(directive)

    def advance(self, directive_id):
        return self.harness.advance(directive_id, self.action_space)

    def inspect(self, directive_id):
        return self.harness.inspect(directive_id)

    def cancel(self, directive_id):
        return self.harness.cancel(directive_id)

    def acknowledge(self, directive_id):
        self.harness.acknowledge(directive_id)
