"""Complete meta-agent harness-research method registration."""
from rsisys.algorithms.meta_agent.controller import (
    build_harness_research, validate_harness_research,
)


def _harness(config):
    if config.get('memory', True):
        from rsisys.api import HarnessAssembly, ResearchAssembly
        return HarnessAssembly(ResearchAssembly.single_meta())
    return None


def register(registry):
    from rsisys.api import ControllerDefinition, MethodDefinition
    from .lab import HarnessResearchLab, validate_config as validate_lab
    registry.register_method('meta_agent', MethodDefinition.from_factories(
        controller=ControllerDefinition(build_harness_research, harness=_harness, endpoint=True), labs={'math': HarnessResearchLab}, version='2',
        validate_controller=validate_harness_research, validate_labs={'math': validate_lab}))
