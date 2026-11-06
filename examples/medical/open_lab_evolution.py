"""Default medical RSI optimizer's optional reflect/coevolution bindings."""
from copy import deepcopy
from dataclasses import asdict

from rsisys.loops.contracts.candidate import SystemVersion
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.harness.policy.prompts import DecisionPrompts
from rsisys.runtime.evolution import ActivationPort, BindingSystemActivator, EvolutionLab, EvolutionState
from rsisys.runtime.systems import SystemRegistry


def bind_evolution(context, loop, lab, backend, model, *, config):
    """Create ports; restoration applies saved values only after state validation.

    Arbitrary code/weights are built by the meta-agent using the Lab. The
    built-in host ports cover prompt and harness configuration; a trusted method
    can install different/whole implementations and explicit state adapters.
    """
    dm = loop.harness
    policy = dm.policy
    def prepare_prompts(value):
        return dict(DecisionPrompts(value)._templates)
    def apply_prompts(value):
        policy.prompts = DecisionPrompts(value)
    fields = ('experience_limit', 'memory_limit', 'review_tool_limit', 'max_outstanding')
    def prepare_harness(value):
        if not isinstance(value, dict) or set(value) != set(fields) or any(
                type(v) is not int or not 1 <= v <= 1024 for v in value.values()):
            raise ValueError('Harness config requires positive bounded experience/memory/review/outstanding limits')
        return deepcopy(value)
    def apply_harness(value):
        for key, item in value.items():
            setattr(dm, key, item)
    ports = {
        'meta.prompts': ActivationPort('decision-prompts/v1', prepare_prompts,
            lambda: dict(policy.prompts._templates), apply_prompts, role='decision_maker'),
        'meta.harness': ActivationPort('decision-harness-limits/v1', prepare_harness,
            lambda: {key: getattr(dm, key) for key in fields}, apply_harness, role='decision_maker'),
    }
    if config.get('reflect', True) is False:
        ports = {}
    def prepare_model(raw):
        ref = ModelRef(**raw)
        # Existing host service binding only: a new checkpoint is not a ready
        # meta service. Deployment must be provided by a real activation port.
        if ref != model:
            raise ValueError('No ready accounted meta service for this model binding')
        return asdict(ref)
    def apply_model(raw):
        policy.backend = backend
        policy.update_model(ModelRef(**raw))
    model_ports = ({'decision_maker': ActivationPort('accounted-ready-meta/v1', prepare_model,
        lambda: asdict(dm.current_model()), apply_model)} if config.get('meta_model_updates', True) else {})
    activator = BindingSystemActivator(ports, context.artifacts,
        models=model_ports,
        path=context.data_dir / 'system-activation.json')
    initial = (SystemVersion('initial', {}, models={'decision_maker': model}) if model_ports else
               SystemVersion('initial', {}, model=model))
    registry = SystemRegistry(initial, activator,
                              path=context.data_dir / 'system-registry.json')
    wrapper = EvolutionLab(lab.lab, registry, context.artifacts)
    # The action space/tools already point at the Lab facade made by lab_factory.
    lab.attach(wrapper)
    if lab.reflection is not None:
        lab.reflection.bind(dm, registry)
        loop.reflection_state = lab.reflection
    loop.evolution_state = EvolutionState(registry)
    return registry


class EvolutionLabSlot:
    """Stable assembly binding; attach host ports before admission, never later."""
    def __init__(self, lab, reflection=None):
        self.lab, self._evolution = lab, None
        self.reflection = reflection

    def attach(self, evolution):
        if self._evolution is not None:
            raise RuntimeError('Evolution ports already installed')
        self._evolution = evolution

    def __getattr__(self, name):
        return getattr(self.lab, name)

    def describe(self):
        return (*self.lab.describe(), *EvolutionLab.specifications(),
                *(self.reflection.specifications() if self.reflection is not None else ()))

    def execute(self, call):
        if call.name == 'reflect_report' and self.reflection is not None:
            self.lab.require_admission()
            return self.reflection.report(call.arguments)
        return self._evolution.execute(call) if self._evolution is not None else self.lab.execute(call)

    def runtime_context(self):
        import json
        native = json.loads(self.lab.runtime_context())
        if self._evolution is not None:
            native['system'] = asdict(self._evolution.registry.current)
        if self.reflection is not None:
            native['reflection'] = self.reflection.snapshot()
        return json.dumps(native, ensure_ascii=False)
