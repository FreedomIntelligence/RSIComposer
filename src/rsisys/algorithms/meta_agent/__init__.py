"""Complete meta-agent harness-research method and optional controller reuse."""
from rsisys.algorithms.meta_agent.controller import (
    build_harness_research, build_meta_agent, validate_config, validate_harness_research,
)


def _research(config):
    if config.get('memory', True):
        from rsisys.research import ResearchAssembly
        return ResearchAssembly.single_meta()
    return None


def register(registry):
    from rsisys.methods import MethodDefinition
    from .lab import HarnessResearchLab, validate_config as validate_lab
    registry.register_method('meta_agent', MethodDefinition.from_factories(
        controller=build_harness_research, labs={'math': HarnessResearchLab}, version='1',
        validate_controller=validate_harness_research, validate_labs={'math': validate_lab},
        endpoint=True, research_factory=_research))
    registry.register_controller('meta_agent', build_meta_agent, version='3',
                                 validate_config=validate_config, endpoint=True,
                                 research_factory=_research)
