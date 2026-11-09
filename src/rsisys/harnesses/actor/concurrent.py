"""Cooperative multiplexing of isolated experiment harnesses, never GPU threads."""
from copy import deepcopy
from typing import Callable

from rsisys.loops.contracts.coordination import ExecutionView
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.harness import ExplorationHarness
from rsisys.loops.contracts.requests import ApplyCandidateSpec


class ConcurrentExplorationHarness:
    """The factory owns resource placement and returns a fresh harness per ID.

    Mutable local components must not be shared. Stateless services and async,
    job-keyed backend clients may be shared; their resource quotas remain backend
    responsibilities. Acknowledged children are retained for receipt retries and
    candidate retrieval, not automatically rebound into another experiment.
    """

    def __init__(self, factory: Callable[[Directive], ExplorationHarness], *, max_in_flight: int = 2):
        if not callable(factory):
            raise TypeError('factory must be callable')
        if type(max_in_flight) is not int or max_in_flight < 1:
            raise ValueError('max_in_flight must be a positive integer')
        self.factory, self.max_in_flight = factory, max_in_flight
        self._children: dict[str, ExplorationHarness] = {}
        self._directives: dict[str, Directive] = {}
        self._acknowledged: set[str] = set()

    def _validate_isolation(self, child):
        for other in self._children.values():
            if child is other:
                raise ValueError('factory reused a harness; each experiment requires isolation')
            for name in ('state', 'policy', 'memory', 'environment', 'training', 'evaluation',
                         'commands', 'episodes', 'learning', 'evaluations', 'local_rl'):
                value = getattr(child, name, None)
                if value is not None and value is getattr(other, name, None):
                    raise ValueError(f'factory shared mutable {name} between experiments')

    def accept(self, directive):
        ident = directive.directive_id
        if ident in self._children:
            if directive != self._directives[ident]:
                raise ValueError('Directive ID reused with different content')
            return self._children[ident].accept(deepcopy(directive))
        if len(self._children) - len(self._acknowledged) >= self.max_in_flight:
            raise RuntimeError('Concurrent exploration capacity exhausted')
        child = self.factory(deepcopy(directive))
        self._validate_isolation(child)
        # A selected result is explicitly copied into the new application target.
        # Its learning executor still enforces full ModelRef base equality.
        if isinstance(directive.payload, ApplyCandidateSpec):
            result = self.candidate_result(directive.payload.candidate_id)
            state = getattr(child, 'state', None)
            if state is None or not hasattr(state, 'candidate_results'):
                raise NotImplementedError('Application target must support candidate import')
            if getattr(getattr(child, 'policy', None), 'model', None) != result.base:
                raise ValueError('Candidate base is stale relative to application target')
            state.candidate_results[result.candidate.candidate_id] = deepcopy(result)
        self._children[ident] = child
        self._directives[ident] = deepcopy(directive)
        # Retain the child even if prepare fails: it may own terminal evidence.
        return child.accept(deepcopy(directive))

    def advance(self, directive_id, action_space):
        return self._children[directive_id].advance(directive_id, action_space)

    def inspect(self, directive_id):
        return self._children[directive_id].inspect(directive_id)

    def cancel(self, directive_id):
        return self._children[directive_id].cancel(directive_id)

    def update(self, request):
        child = self._children[request.directive_id]
        update = getattr(child, 'update', None)
        if update is None:
            raise NotImplementedError('This experiment does not support live updates')
        return update(deepcopy(request))

    def acknowledge(self, directive_id):
        self._children[directive_id].acknowledge(directive_id)
        self._acknowledged.add(directive_id)

    def pending_all(self):
        return tuple(self.inspect(ident) for ident in self._children if ident not in self._acknowledged)

    def pending(self):
        pending = self.pending_all()
        if len(pending) > 1:
            raise RuntimeError('Multiple experiments: use pending_all() and a parallel runner')
        return pending[0] if pending else None

    def execution_views(self):
        return {ident: self.execution_view_for(ident) for ident in self._children
                if ident not in self._acknowledged}

    def execution_view_for(self, directive_id):
        child = self._children[directive_id]
        describe = getattr(child, 'execution_view', None)
        return deepcopy(describe() if describe else ExecutionView(progress=self.inspect(directive_id)))

    def execution_view(self):
        pending = self.pending()
        return self.execution_view_for(pending.directive_id) if pending else ExecutionView()

    def candidate_result(self, candidate_id):
        results = [child.candidate_result(candidate_id) for child in self._children.values()
                   if candidate_id in getattr(getattr(child, 'state', None), 'candidate_results', {})]
        if not results:
            raise KeyError(candidate_id)
        if any(result != results[0] for result in results[1:]):
            raise ValueError('Candidate ID has conflicting experiment results')
        return deepcopy(results[0])

    def apply_candidate(self, candidate_id, *, target_directive_id, expected_model):
        """Explicit local binding after ack, with a compare-and-swap model guard.

        This changes only the named experiment. There is no implicit global model
        promotion. An active target must finish/cancel and acknowledge first.
        """
        child = self._children[target_directive_id]
        if target_directive_id not in self._acknowledged:
            raise RuntimeError('Application target still owns an unacknowledged directive')
        if getattr(child.policy, 'model', None) != expected_model:
            raise ValueError('Application target model changed')
        result = self.candidate_result(candidate_id)
        if result.base != expected_model:
            raise ValueError('Candidate base is stale relative to application target')
        child.state.candidate_results[candidate_id] = deepcopy(result)
        return child.apply_candidate(candidate_id)
