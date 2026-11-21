"""FunSearch method registration for the online bin-packing experiment."""

from rsicomposer.api import MethodDefinition


def register(registry):
    def controller(context, config):
        from .controller import FunSearchController
        return FunSearchController(context, config)

    def lab(context, config):
        from .lab import FunSearchLab
        return FunSearchLab(context, config)

    registry.register_method('funsearch', MethodDefinition.from_factories(
        controller=controller, labs={'funsearch': lab}, version='3',
    ))
