"""Register the template method and its Controller.

The Controller is a Policy (controller.py), an optional Harness (harness.py),
and optional Memory (memory.py). Each is a separate factory, so each can be
replaced on its own. Set "harness": "scoreboard" or "memory": "best_score" in
the Controller config to try the prototypes; omit them for the defaults.
"""
from rsicomposer.api import ControllerDefinition, MemoryAssembly, MethodDefinition, ResearchAssembly

from .controller import PromptSearchController, validate_config as validate_policy
from .harness import ScoreboardHarness, validate_rows
from .memory import BestScoreMemory


def policy_options(config):
    return {'max_evaluations': config['max_evaluations']} if 'max_evaluations' in config else {}


def validate_config(config):
    if set(config) - {'max_evaluations', 'harness', 'memory', 'scoreboard_rows'}:
        raise ValueError('Unknown template Controller option')
    validate_policy(policy_options(config))
    if config.get('harness', 'default') not in ('default', 'scoreboard'):
        raise ValueError('harness must be default or scoreboard')
    if config.get('memory', 'none') not in ('none', 'best_score'):
        raise ValueError('memory must be none or best_score')
    validate_rows(config.get('scoreboard_rows', 20))


def policy_factory(context, config):
    return PromptSearchController(context, policy_options(config))


def harness_factory(context, config):
    return ScoreboardHarness(config.get('scoreboard_rows', 20)) if config.get('harness') == 'scoreboard' else None


def memory_factory(config):
    if config.get('memory') != 'best_score':
        return None
    return MemoryAssembly(ResearchAssembly(context_memory=BestScoreMemory(), trajectory_archive=None))


CONTROLLER = ControllerDefinition(policy_factory, harness=harness_factory, memory=memory_factory)


def register(registry):
    def build(config):
        # Only the whole method needs this Lab; its standalone Controller can pair with another.
        return MethodDefinition.from_factories(controller=CONTROLLER, labs={'math': registry.lab('template_math')},
            version='1', validate_controller=validate_config).factory(config)

    registry.register_controller('template_prompt_search', CONTROLLER, version='1',
                                 validate_config=validate_config)
    registry.register_method('template_prompt_search', MethodDefinition(build, '1'))
