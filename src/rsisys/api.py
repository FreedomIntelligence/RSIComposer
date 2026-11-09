"""Supported entry point for external RSI controllers, Labs and registration."""
from typing import TYPE_CHECKING
from .assembly import AssemblyContext, ModuleAssembly
from .application import Application, ApplicationConfig
from .loops.contracts.coordination import OperationResult
from .loops.contracts.decision import Decision, DecisionInput, DecisionPolicy, OperationCall, OperationSpec
from .loops.contracts.execution import LabExecution, LabToolError
from .loops.contracts.lifecycle import ExecutionStatus
from .loops.contracts.rsi import (RSIController, Lab, LabRequest, LabFeedback,
                                ControllerOperationRequirements, ControllerReportProvider)
from .methods import MethodDefinition, RSIMethod
from .registry import Registry
from .loops.composition.single_loop import SingleLoopController


if TYPE_CHECKING:
    from rsisys.algorithms.rest_em.controller import RestEMController
    from rsisys.algorithms.rrsi.controller import RRSIController
    from rsisys.algorithms.seal.controller import SEALController
    from rsisys.algorithms.ttrl.controller import TTRLController, TTRLVerlController
    from rsisys.algorithms.ft_agent.controller import FTAgentController
    from rsisys.algorithms.aide.controller import AIDEController

_LAZY_METHOD_EXPORTS = {
    'RestEMController': 'rsisys.algorithms.rest_em.controller',
    'RRSIController': 'rsisys.algorithms.rrsi.controller',
    'SEALController': 'rsisys.algorithms.seal.controller',
    'TTRLController': 'rsisys.algorithms.ttrl.controller',
    'TTRLVerlController': 'rsisys.algorithms.ttrl.controller',
    'FTAgentController': 'rsisys.algorithms.ft_agent.controller',
    'AIDEController': 'rsisys.algorithms.aide.controller',
}


def __getattr__(name):
    module_name = _LAZY_METHOD_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = ['Registry', 'SingleLoopController', 'RestEMController', 'RRSIController', 'SEALController', 'TTRLController', 'TTRLVerlController', 'FTAgentController', 'AIDEController', 'DecisionPolicy', 'Decision', 'DecisionInput',
           'RSIController', 'Lab', 'LabRequest', 'LabFeedback',
           'ControllerOperationRequirements', 'ControllerReportProvider',
           'OperationCall', 'OperationSpec', 'OperationResult', 'ExecutionStatus',
           'LabExecution', 'LabToolError', 'AssemblyContext', 'ModuleAssembly',
           'MethodDefinition', 'RSIMethod', 'Application', 'ApplicationConfig']
