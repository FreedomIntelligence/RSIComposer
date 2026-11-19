"""Register the template method and its Controller.

The Controller is a Policy (controller.py), an optional Harness (harness.py),
and optional Memory (memory.py). Each is a separate factory, so each can be
replaced on its own. Set "harness": "scoreboard" or "memory": "best_score" in
the Controller config to try the prototypes; omit them for the defaults.
"""
from rsisys.api import ControllerDefinition, MemoryAssembly, MethodDefinition, ResearchAssembly

from .controller import PromptSearchController, validate_config
from .harness import ScoreboardHarness
from .memory import BestScoreMemory


def harness_factory(context, config):
    return ScoreboardHarness(context, config) if config.get('harness') == 'scoreboard' else None


def memory_factory(config):
    if config.get('memory') != 'best_score':
        return None
    return MemoryAssembly(ResearchAssembly(context_memory=BestScoreMemory(), trajectory_archive=None))


CONTROLLER = ControllerDefinition(PromptSearchController, harness=harness_factory, memory=memory_factory)


def register(registry):
    def lab_factory(context, config):
        return registry.resolve_component('scene', 'template_math', context, config)

    registry.register_controller('template_prompt_search', CONTROLLER, version='1',
                                 validate_config=validate_config)
    registry.register_method('template_prompt_search', MethodDefinition.from_factories(
        controller=CONTROLLER,
        labs={'math': lab_factory},
        version='1',
        validate_controller=validate_config,
    ))
