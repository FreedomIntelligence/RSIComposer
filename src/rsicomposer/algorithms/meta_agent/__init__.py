"""Complete meta-agent harness-research method registration."""
from rsicomposer.algorithms.meta_agent.controller import (
    build_harness_research, validate_harness_research,
)


def _memory(config):
    if config.get('memory', True):
        from rsicomposer.api import MemoryAssembly, ResearchAssembly
        return MemoryAssembly(ResearchAssembly.single_meta())
    return None


def register(registry):
    from rsicomposer.api import ControllerDefinition, MethodDefinition, ObservationProbe
    from .lab import HarnessResearchLab, validate_config as validate_lab
    registry.register_method('meta_agent', MethodDefinition.from_factories(
        controller=ControllerDefinition(build_harness_research, memory=_memory, endpoint=True), labs={'math': HarnessResearchLab}, version='2',
        observation_probes=(ObservationProbe('meta_agent.candidate', 'decision', (), 'lineage', content=True),),
        validate_controller=validate_harness_research, validate_labs={'math': validate_lab}))
