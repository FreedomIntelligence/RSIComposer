"""Adapt the default harness view to the public decision input window."""
from copy import deepcopy
from dataclasses import asdict
import json

from rsisys.loops.contracts.coordination import ReviewDecision
from rsisys.loops.contracts.decision import DecisionInput
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.runtime.capabilities import optional_method


class DecisionInputAdapter:
    """The rich DecisionContext stays here, outside third-party policy contracts.

    context and references are optional rendering callbacks over the default
    harness view. The default text retains its observed facts without requiring
    consuming policies to understand Memory, HistoryEntry or research_state.
    """
    def __init__(self, *, context=None, references=None, directives=None):
        if ((context is not None and not callable(context))
                or (references is not None and not callable(references))):
            raise TypeError('Decision input renderers must be callable')
        self.context, self.references = context, references
        self.directives = directives if directives is not None else DirectiveDefinitions()

    def __call__(self, view):
        return self._build(view, review=False)

    def review_input(self, view):
        return self._build(view, review=True)

    def _build(self, view, *, review):
        capabilities = tuple({'kind': 'invoke', **asdict(spec)} for spec in view.operations)
        if review:
            capabilities += tuple({'kind': 'review', 'name': action} for action in view.review_actions)
        else:
            capabilities += tuple({'kind': 'explore', **asdict(spec)}
                                  for spec in self.directives.describe(view.directive_modes))
        if self.context is None:
            data = asdict(view)
            for field in ('objective', 'operations', 'directive_modes', 'review_actions'):
                data.pop(field, None)
            text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        else:
            text = self.context(view)
        return DecisionInput(view.objective, text, capabilities,
                             self.references(view) if self.references is not None else ())


class ContextDecisionPolicy:
    """Use a public decide(DecisionInput) policy inside the default harness.

    The policy may also be a plain callable. Native tool calling and SDK parsing
    belong to it; only the returned action crosses the host execution boundary.
    Optional capabilities are forwarded rather than requiring a base class.
    """
    def __init__(self, policy, *, input_adapter=None):
        if not callable(policy) and not callable(getattr(policy, 'decide', None)):
            raise TypeError('Decision policy requires decide(input) or a callable')
        self.policy = policy
        self.input_adapter = DecisionInputAdapter() if input_adapter is None else input_adapter
        if not callable(self.input_adapter):
            raise TypeError('Decision input adapter must be callable')

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _input(self, context, *, review=False):
        adapter = optional_method(self.input_adapter, 'review_input') if review else None
        value = (adapter or self.input_adapter)(deepcopy(context))
        if not isinstance(value, DecisionInput):
            raise TypeError('Decision input adapter must return DecisionInput')
        return value

    def decide(self, context):
        operation = getattr(self.policy, 'decide', None)
        return (operation if callable(operation) else self.policy)(self._input(context))

    def review(self, context):
        review = optional_method(self.policy, 'review')
        return review(self._input(context, review=True)) if review is not None else ReviewDecision()
