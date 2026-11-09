"""Receives system directives; Actor actions are a separate environment vocabulary."""
from rsisys.loops.contracts.coordination import ExecutionView
from rsisys.loops.contracts.harness import ExplorationHarness
from rsisys.loops.contracts.queries import QueryProvider
from rsisys.loops.exploration.action_space import ActionSpace


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

    def pending(self):
        discover = getattr(self.harness, 'pending', None)
        return discover() if discover is not None else None

    def pending_all(self):
        discover = getattr(self.harness, 'pending_all', None)
        if discover is not None:
            return discover()
        pending = self.pending()
        return () if pending is None else (pending,)

    def execution_views(self):
        describe = getattr(self.harness, 'execution_views', None)
        if describe is not None:
            return describe()
        return {p.directive_id: self.execution_view() for p in self.pending_all()}

    def update(self, request):
        update = getattr(self.harness, 'update', None)
        if update is None:
            raise NotImplementedError('This exploration harness does not support updates')
        return update(request)

    def execution_view(self):
        view = getattr(self.harness, 'execution_view', None)
        return view() if view is not None else ExecutionView(progress=self.pending())

    def describe_queries(self):
        return self.harness.describe_queries() if isinstance(self.harness, QueryProvider) else ()

    def query(self, request):
        if not isinstance(self.harness, QueryProvider):
            raise NotImplementedError('This exploration harness does not support queries')
        return self.harness.query(request)
