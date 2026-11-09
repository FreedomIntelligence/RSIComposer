"""Model-driven controller template; harness and memory are method choices."""
from .controller import build_meta_agent, validate_config


def _research(config):
    if config.get('memory', True):
        from rsisys.research import ResearchAssembly
        return ResearchAssembly.single_meta()
    return None


def register(registry):
    registry.register_controller('meta_agent', build_meta_agent, version='2',
                                 validate_config=validate_config, endpoint=True,
                                 research_factory=_research)
