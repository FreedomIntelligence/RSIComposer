"""Public imports for RSI methods, Labs and trusted host integrations.

Import framework types and helpers here; keep method-private code in your own
package. Exports are resolved on first access, preserving the implementation's
identity (including dataclasses used in checkpoints). Importing this module or
the core contracts does not load optional serving/training/harness providers.
Using a provider still requires its documented environment and resources.
"""
from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING


_EXPORTS = {
    'SwapPlanner': ('rsicomposer.loops.contracts.controller_swap', 'SwapPlanner'),
    'NestedRun': ('rsicomposer.runtime.execution.nested', 'NestedRun'),
    'ReadOnlyEvidenceView': ('rsicomposer.runtime.evidence.view', 'ReadOnlyEvidenceView'),
    'load_controller_version': ('rsicomposer.runtime.state.controller_versions', 'load_controller_version'),
    'save_controller_version': ('rsicomposer.runtime.state.controller_versions', 'save_controller_version'),
    'EvidenceScope': ('rsicomposer.loops.contracts.controller_version', 'EvidenceScope'),
    'ControllerVersion': ('rsicomposer.loops.contracts.controller_version', 'ControllerVersion'),
    'HostedLoRAEngine': ('rsicomposer.backends.training.lora.hosted', 'HostedLoRAEngine'),
    'process_identity': ('rsicomposer.runtime.execution.process_identity', 'process_identity'),
    'WeightSync': ('rsicomposer.runtime.execution.weight_sync', 'WeightSync'),
    'FilesystemWeightSync': ('rsicomposer.runtime.execution.weight_sync', 'FilesystemWeightSync'),
    'ResourceRequest': ('rsicomposer.runtime.execution.resources', 'ResourceRequest'),
    'WorkerLaunchUncertainError': ('rsicomposer.runtime.execution.errors', 'WorkerLaunchUncertainError'),
    'WorkerLostError': ('rsicomposer.runtime.execution.errors', 'WorkerLostError'),
    'socket_relay_command': ('rsicomposer.runtime.execution.socket_relay', 'socket_relay_command'),
    'HostWorkerServices': ('rsicomposer.runtime.execution.remote_services', 'HostWorkerServices'),
    'RemoteArtifacts': ('rsicomposer.runtime.execution.remote_services', 'RemoteArtifacts'),
    'ResourceManager': ('rsicomposer.runtime.execution.resources', 'ResourceManager'),
    'Lease': ('rsicomposer.runtime.execution.resources', 'Lease'),
    'Executor': ('rsicomposer.runtime.execution.executor', 'Executor'),
    'WorkerSpec': ('rsicomposer.runtime.execution.executor', 'WorkerSpec'),
    'WorkerHandle': ('rsicomposer.runtime.execution.executor', 'WorkerHandle'),
    'execution_capabilities': ('rsicomposer.runtime.execution.executor', 'execution_capabilities'),
    'run_cpu': ('rsicomposer.runtime.execution.executor', 'run_cpu'),
    'HostBudget': ('rsicomposer.runtime.execution.budget', 'HostBudget'),
    'BudgetExhausted': ('rsicomposer.runtime.execution.budget', 'BudgetExhausted'),

    'DatasetBinding': ('rsicomposer.loops.contracts.data', 'DatasetBinding'),
    'MetricSpec': ('rsicomposer.loops.contracts.profiles', 'MetricSpec'),
    'SplitRole': ('rsicomposer.loops.contracts.profiles', 'SplitRole'),
    'Invariant': ('rsicomposer.loops.contracts.profiles', 'Invariant'),
    'LabProfile': ('rsicomposer.loops.contracts.profiles', 'LabProfile'),
    'get_profile': ('rsicomposer.loops.contracts.profiles', 'get_profile'),
    'TRAIN': ('rsicomposer.loops.contracts.profiles', 'TRAIN'),
    'TRAIN_UNLABELED': ('rsicomposer.loops.contracts.profiles', 'TRAIN_UNLABELED'),
    'DEV': ('rsicomposer.loops.contracts.profiles', 'DEV'),
    'SEALED': ('rsicomposer.loops.contracts.profiles', 'SEALED'),
    'WEIGHT_TRAINING': ('rsicomposer.loops.contracts.profiles', 'WEIGHT_TRAINING'),
    'ROUND_WEIGHT_TRAINING': ('rsicomposer.loops.contracts.profiles', 'ROUND_WEIGHT_TRAINING'),
    'REFERENCE_WEIGHT_TRAINING': ('rsicomposer.loops.contracts.profiles', 'REFERENCE_WEIGHT_TRAINING'),
    'ARTIFACT_SEARCH': ('rsicomposer.loops.contracts.profiles', 'ARTIFACT_SEARCH'),
    'PROGRAM_EXECUTION': ('rsicomposer.loops.contracts.profiles', 'PROGRAM_EXECUTION'),
    'CODING_TASK': ('rsicomposer.loops.contracts.profiles', 'CODING_TASK'),

    # Explicit composition, memory and state integration ports.
    'NullHarness': ('rsicomposer.loops.contracts.harness', 'NullHarness'),
    'ControllerHarness': ('rsicomposer.loops.contracts.harness', 'ControllerHarness'),
    'ResearchPolicyAdapter': ('rsicomposer.harnesses.decision.public_policy', 'ResearchPolicyAdapter'),
    'DecisionResearchMetadata': ('rsicomposer.host.ledger', 'DecisionResearchMetadata'),
    'ResearchControllerHarness': ('rsicomposer.harnesses.decision.engine', 'ResearchControllerHarness'),
    'ComponentDefinition': ('rsicomposer.components', 'ComponentDefinition'),
    'ComponentGraph': ('rsicomposer.components', 'ComponentGraph'),
    'LangMemMemory': ('rsicomposer.harnesses.decision.langmem', 'LangMemMemory'),
    'ControllerMemory': ('rsicomposer.loops.contracts.feedback_context', 'ControllerMemory'),
    'ContextQuery': ('rsicomposer.loops.contracts.feedback_context', 'ContextQuery'),
    'EvidenceEvent': ('rsicomposer.loops.contracts.feedback_context', 'EvidenceEvent'),
    'RecoveryContract': ('rsicomposer.runtime.state.contracts', 'RecoveryContract'),
    'SnapshotContract': ('rsicomposer.runtime.state.contracts', 'SnapshotContract'),
    'AssemblyRecovery': ('rsicomposer.runtime.state.component_state', 'AssemblyRecovery'),
    'LocalSnapshotCapture': ('rsicomposer.snapshot.local', 'LocalSnapshotCapture'),
    'restore_snapshot': ('rsicomposer.snapshot.local', 'restore_snapshot'),

    # Application and method assembly.
    'Application': ('rsicomposer.application', 'Application'),
    'ApplicationConfig': ('rsicomposer.application', 'ApplicationConfig'),
    'AssemblyContext': ('rsicomposer.assembly', 'AssemblyContext'),
    'MethodDefinition': ('rsicomposer.methods', 'MethodDefinition'),
    'ControllerDefinition': ('rsicomposer.methods', 'ControllerDefinition'),
    'StatefulPolicy': ('rsicomposer.method_state', 'StatefulPolicy'),
    'StatefulLab': ('rsicomposer.method_state', 'StatefulLab'),
    'MemoryAssembly': ('rsicomposer.research', 'MemoryAssembly'),
    'ModuleAssembly': ('rsicomposer.assembly', 'ModuleAssembly'),
    'RSIMethod': ('rsicomposer.methods', 'RSIMethod'),
    'Registry': ('rsicomposer.registry', 'Registry'),
    'ResearchAssembly': ('rsicomposer.research', 'ResearchAssembly'),
    'build_default_decision': ('rsicomposer.assembly', 'build_default_decision'),
    # Controller, Lab, artifact and job contracts.
    'ArtifactRef': ('rsicomposer.loops.contracts.artifacts', 'ArtifactRef'),
    'BackendBinding': ('rsicomposer.loops.contracts.model_service', 'BackendBinding'),
    'ControllerOperationRequirements': ('rsicomposer.loops.contracts.rsi', 'ControllerOperationRequirements'),
    'ControllerReportProvider': ('rsicomposer.loops.contracts.rsi', 'ControllerReportProvider'),
    'Decision': ('rsicomposer.loops.contracts.decision', 'Decision'),
    'DecisionInput': ('rsicomposer.loops.contracts.decision', 'DecisionInput'),
    'DecisionPolicy': ('rsicomposer.loops.contracts.decision', 'DecisionPolicy'),
    'ExecutionStatus': ('rsicomposer.loops.contracts.lifecycle', 'ExecutionStatus'),
    'GenerationRequest': ('rsicomposer.loops.contracts.model_service', 'GenerationRequest'),
    'GenerationResponse': ('rsicomposer.loops.contracts.model_service', 'GenerationResponse'),
    'JobOutcome': ('rsicomposer.loops.contracts.jobs', 'JobOutcome'),
    'OperationProgress': ('rsicomposer.loops.contracts.jobs', 'OperationProgress'),
    'JobControl': ('rsicomposer.loops.contracts.jobs', 'JobControl'),
    'JobProgress': ('rsicomposer.loops.contracts.jobs', 'JobProgress'),
    'JobRef': ('rsicomposer.loops.contracts.jobs', 'JobRef'),
    'JobStatus': ('rsicomposer.loops.contracts.jobs', 'JobStatus'),
    'LabExecution': ('rsicomposer.loops.contracts.execution', 'LabExecution'),
    'LabToolError': ('rsicomposer.loops.contracts.execution', 'LabToolError'),
    'ModelBackend': ('rsicomposer.loops.contracts.model_service', 'ModelBackend'),
    'ModelRef': ('rsicomposer.loops.contracts.model_service', 'ModelRef'),
    'ModelResponseError': ('rsicomposer.loops.contracts.model_service', 'ModelResponseError'),
    'ModelTransportError': ('rsicomposer.loops.contracts.model_service', 'ModelTransportError'),
    'OperationCall': ('rsicomposer.loops.contracts.decision', 'OperationCall'),
    'OperationResult': ('rsicomposer.loops.contracts.coordination', 'OperationResult'),
    'OperationSpec': ('rsicomposer.loops.contracts.decision', 'OperationSpec'),
    'TokenUsage': ('rsicomposer.loops.contracts.model_service', 'TokenUsage'),
    # Observation from the host and worker processes.
    'current_observations': ('rsicomposer.runtime.observation', 'current_observations'),
    'ObservationProbe': ('rsicomposer.runtime.observation.facts', 'ObservationProbe'),
    'emit_observation': ('rsicomposer.runtime.observation.facts', 'emit_observation'),
    'observation_id': ('rsicomposer.runtime.observation.facts', 'observation_id'),
    'log': ('rsicomposer.runtime.observation.client', 'log'),
    'observe': ('rsicomposer.runtime.observation', 'observe'),
    'register_probe': ('rsicomposer.runtime.observation', 'register_probe'),
    'report_devices': ('rsicomposer.runtime.observation', 'report_devices'),
    'report_model_calls': ('rsicomposer.runtime.observation', 'report_model_calls'),
    'report_samples': ('rsicomposer.runtime.observation', 'report_samples'),
    'report_training_log': ('rsicomposer.runtime.observation', 'report_training_log'),
    'report_version': ('rsicomposer.runtime.observation', 'report_version'),
    'report_version_evaluation': ('rsicomposer.runtime.observation', 'report_version_evaluation'),
    'report_version_selection': ('rsicomposer.runtime.observation', 'report_version_selection'),
    'trigger_probe': ('rsicomposer.runtime.observation', 'trigger_probe'),
    # The math Lab's answer verifiers, shared by math methods (Math-Verify is optional).
    'AnswerCheck': ('rsicomposer.labs.math.verifiers', 'AnswerCheck'),
    'MATH_VERIFIERS': ('rsicomposer.labs.math.verifiers', 'VERIFIERS'),
    'AnswerVerifier': ('rsicomposer.labs.math.verifiers', 'AnswerVerifier'),
    'CompetitionIntegerVerifier': ('rsicomposer.labs.math.verifiers', 'CompetitionIntegerVerifier'),
    'MarkedNumericVerifier': ('rsicomposer.labs.math.verifiers', 'MarkedNumericVerifier'),
    'MathVerifyVerifier': ('rsicomposer.labs.math.verifiers', 'MathVerifyVerifier'),
    'StrictNumericVerifier': ('rsicomposer.labs.math.verifiers', 'StrictNumericVerifier'),
    'build_verifier': ('rsicomposer.labs.math.verifiers', 'build_verifier'),
    'final_boxed': ('rsicomposer.labs.math.verifiers', 'final_boxed'),
    'marked_number': ('rsicomposer.labs.math.verifiers', 'marked_number'),
    'numeric_answer': ('rsicomposer.labs.math.verifiers', 'numeric_answer'),
    'parse_reference': ('rsicomposer.labs.math.verifiers', 'parse_reference'),
    'score_record': ('rsicomposer.labs.math.verifiers', 'score_record'),
    'validate_verifier': ('rsicomposer.labs.math.verifiers', 'validate_verifier'),
    # Containerized coding benchmark provider (Harbor is optional at runtime).
    'MathLab': ('rsicomposer.labs.math.lab', 'MathLab'),
    'MathAttempt': ('rsicomposer.labs.math.attempt', 'Attempt'),
    'CheckpointRuntime': ('rsicomposer.labs.math.runtimes', 'CheckpointRuntime'),
    'InstructionRuntime': ('rsicomposer.labs.math.runtimes', 'InstructionRuntime'),
    'PromptTemplateRuntime': ('rsicomposer.labs.math.runtimes', 'PromptTemplateRuntime'),
    'StepProgramRuntime': ('rsicomposer.labs.math.runtimes', 'StepProgramRuntime'),
    'CodingLab': ('rsicomposer.labs.coding.lab', 'CodingLab'),
    'ComputerUseLab': ('rsicomposer.labs.computer_use.lab', 'ComputerUseLab'),
    'ProfessionalWorkLab': ('rsicomposer.labs.professional_work.lab', 'ProfessionalWorkLab'),
    'SearchLab': ('rsicomposer.labs.search.lab', 'SearchLab'),
    'CodingArtifact': ('rsicomposer.labs.coding.artifacts', 'CodingArtifact'),
    'HarborAgentRuntime': ('rsicomposer.labs.coding.agents', 'HarborAgentRuntime'),
    'Benchmark': ('rsicomposer.loops.contracts.benchmark', 'Benchmark'),
    'DomainAgentRuntime': ('rsicomposer.loops.contracts.domain_agent', 'DomainAgentRuntime'),
    'DomainAgent': ('rsicomposer.loops.contracts.domain_agent', 'DomainAgent'),
    'DomainAgentRejected': ('rsicomposer.loops.contracts.domain_agent', 'DomainAgentRejected'),
    'DomainAgentSpec': ('rsicomposer.loops.contracts.domain_agent', 'DomainAgentSpec'),
    # Optional serving and training implementations.
    'ChatPool': ('rsicomposer.backends.serving.chat', 'ChatPool'),
    'LoRAEngine': ('rsicomposer.backends.training.lora.engine', 'LoRAEngine'),
    'ParallelLoRAEngine': ('rsicomposer.backends.training.lora.parallel', 'ParallelLoRAEngine'),
    'VLLMBackend': ('rsicomposer.backends.serving.vllm', 'VLLMBackend'),
    'build_serving_backend': ('rsicomposer.backends.serving', 'build_serving_backend'),
    'SGLangBackend': ('rsicomposer.backends.serving.sglang', 'SGLangBackend'),
    'WeightEngine': ('rsicomposer.backends.training.engine', 'WeightEngine'),
    'load_model': ('rsicomposer.backends.training.worker', 'load_model'),
    'merged_change_count': ('rsicomposer.backends.training.worker', 'merged_change_count'),
    'tensor_batch': ('rsicomposer.backends.training.worker', 'tensor_batch'),
    'validate_engine': ('rsicomposer.backends.training.engine', 'validate_engine'),
    'weights_digest': ('rsicomposer.backends.training.worker', 'weights_digest'),
    # Optional harnesses.
    'BasicContextCompactor': ('rsicomposer.harnesses.decision.context', 'BasicContextCompactor'),
    'DSHConfig': ('rsicomposer.harnesses.decision.dsh', 'DSHConfig'),
    'DecisionPrompts': ('rsicomposer.harnesses.decision.policy.prompts', 'DecisionPrompts'),
    'HarnessExecutionError': ('rsicomposer.harnesses.python_step', 'HarnessExecutionError'),
    'ModelDecisionPolicy': ('rsicomposer.harnesses.decision.policy', 'ModelDecisionPolicy'),
    'dsh_decision_factory': ('rsicomposer.harnesses.decision.dsh', 'dsh_decision_factory'),
    'sandbox_step': ('rsicomposer.harnesses.python_step', 'sandbox_step'),
    'validate_source': ('rsicomposer.harnesses.python_step', 'validate_source'),
    # Execution and configuration helpers.
    'ComponentModels': ('rsicomposer.runtime.models.component_access', 'ComponentModels'),
    'ComponentModelServer': ('rsicomposer.runtime.models.component_endpoint', 'ComponentModelServer'),
    'AccountedModelBackend': ('rsicomposer.runtime.models.accounting', 'AccountedModelBackend'),
    'ExecutionPolicy': ('rsicomposer.runtime.execution.isolation', 'ExecutionPolicy'),
    'ProcessIsolation': ('rsicomposer.runtime.execution.isolation', 'ProcessIsolation'),
    'ResearchBridge': ('rsicomposer.runtime.execution.research_bridge', 'ResearchBridge'),
    'valid_replay_point': ('rsicomposer.runtime.execution.replay', 'valid_replay_point'),
    'request_digest': ('rsicomposer.runtime.execution.replay', 'request_digest'),
    'ReplayLog': ('rsicomposer.runtime.execution.replay', 'ReplayLog'),
    'ReplayMismatchError': ('rsicomposer.runtime.execution.replay', 'ReplayMismatchError'),
    'UpstreamWorker': ('rsicomposer.runtime.execution.upstream_worker', 'UpstreamWorker'),
    'UpstreamAgentPolicy': ('rsicomposer.runtime.execution.upstream_policy', 'UpstreamAgentPolicy'),
    'atomic_write': ('rsicomposer.runtime.storage.files', 'atomic_write'),
    'checkout': ('rsicomposer.runtime.execution.upstream', 'checkout'),
    'config_json': ('rsicomposer.components', 'config_json'),
    'emit_event': ('rsicomposer.runtime.execution.research_bridge', 'emit_event'),
    'file_hash': ('rsicomposer.runtime.storage.files', 'file_hash'),
    'json_bytes': ('rsicomposer.runtime.storage.files', 'json_bytes'),
    'read_workspace_file': ('rsicomposer.runtime.storage.workspace_files', 'read_workspace_file'),
    'run_workers': ('rsicomposer.runtime.execution.workers', 'run_workers'),
    'system_runtime_paths': ('rsicomposer.runtime.execution.isolation', 'system_runtime_paths'),
    'worker_environment': ('rsicomposer.runtime.execution.workers', 'worker_environment'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


if _TYPE_CHECKING:
    from rsicomposer.loops.contracts.controller_swap import SwapPlanner as SwapPlanner
    from rsicomposer.runtime.execution.nested import NestedRun as NestedRun
    from rsicomposer.runtime.evidence.view import ReadOnlyEvidenceView as ReadOnlyEvidenceView
    from rsicomposer.runtime.state.controller_versions import load_controller_version as load_controller_version
    from rsicomposer.runtime.state.controller_versions import save_controller_version as save_controller_version
    from rsicomposer.loops.contracts.controller_version import EvidenceScope as EvidenceScope
    from rsicomposer.loops.contracts.controller_version import ControllerVersion as ControllerVersion
    from rsicomposer.backends.training.lora.hosted import HostedLoRAEngine as HostedLoRAEngine
    from rsicomposer.runtime.execution.process_identity import process_identity as process_identity
    from rsicomposer.runtime.execution.weight_sync import (
        WeightSync as WeightSync,
        FilesystemWeightSync as FilesystemWeightSync,
    )
    from rsicomposer.runtime.execution.resources import ResourceRequest as ResourceRequest
    from rsicomposer.runtime.execution.errors import WorkerLostError as WorkerLostError
    from rsicomposer.runtime.execution.errors import WorkerLaunchUncertainError as WorkerLaunchUncertainError
    from rsicomposer.runtime.execution.socket_relay import socket_relay_command as socket_relay_command
    from rsicomposer.runtime.execution.remote_services import HostWorkerServices as HostWorkerServices
    from rsicomposer.runtime.execution.remote_services import RemoteArtifacts as RemoteArtifacts
    from rsicomposer.runtime.execution.resources import ResourceManager as ResourceManager
    from rsicomposer.runtime.execution.resources import Lease as Lease
    from rsicomposer.runtime.execution.executor import Executor as Executor
    from rsicomposer.runtime.execution.executor import WorkerSpec as WorkerSpec
    from rsicomposer.runtime.execution.executor import WorkerHandle as WorkerHandle
    from rsicomposer.runtime.execution.executor import execution_capabilities as execution_capabilities
    from rsicomposer.runtime.execution.executor import run_cpu as run_cpu
    from rsicomposer.runtime.execution.budget import HostBudget as HostBudget
    from rsicomposer.runtime.execution.budget import BudgetExhausted as BudgetExhausted

    from rsicomposer.labs.math.lab import MathLab as MathLab
    from rsicomposer.labs.math.attempt import Attempt as MathAttempt
    from rsicomposer.labs.math.runtimes import (
        CheckpointRuntime as CheckpointRuntime, InstructionRuntime as InstructionRuntime,
        PromptTemplateRuntime as PromptTemplateRuntime,
        StepProgramRuntime as StepProgramRuntime,
    )
    from rsicomposer.labs.coding.lab import CodingLab as CodingLab
    from rsicomposer.labs.computer_use.lab import ComputerUseLab as ComputerUseLab
    from rsicomposer.labs.professional_work.lab import ProfessionalWorkLab as ProfessionalWorkLab
    from rsicomposer.labs.search.lab import SearchLab as SearchLab
    from rsicomposer.labs.coding.artifacts import CodingArtifact as CodingArtifact
    from rsicomposer.labs.coding.agents import HarborAgentRuntime as HarborAgentRuntime
    from rsicomposer.loops.contracts.benchmark import Benchmark as Benchmark
    from rsicomposer.loops.contracts.domain_agent import (
        DomainAgentRuntime as DomainAgentRuntime, DomainAgent as DomainAgent, DomainAgentRejected as DomainAgentRejected,
        DomainAgentSpec as DomainAgentSpec,
    )
    from rsicomposer.application import Application as Application, ApplicationConfig as ApplicationConfig
    from rsicomposer.assembly import (
        AssemblyContext as AssemblyContext,
        ModuleAssembly as ModuleAssembly,
        build_default_decision as build_default_decision,
    )
    from rsicomposer.backends.serving.chat import ChatPool as ChatPool
    from rsicomposer.backends.serving.vllm import VLLMBackend as VLLMBackend
    from rsicomposer.backends.serving import build_serving_backend as build_serving_backend
    from rsicomposer.backends.serving.sglang import SGLangBackend as SGLangBackend
    from rsicomposer.backends.training.engine import (
        WeightEngine as WeightEngine,
        validate_engine as validate_engine,
    )
    from rsicomposer.backends.training.lora.engine import LoRAEngine as LoRAEngine
    from rsicomposer.backends.training.lora.parallel import ParallelLoRAEngine as ParallelLoRAEngine
    from rsicomposer.backends.training.worker import (
        load_model as load_model,
        merged_change_count as merged_change_count,
        tensor_batch as tensor_batch,
        weights_digest as weights_digest,
    )
    from rsicomposer.components import config_json as config_json
    from rsicomposer.harnesses.decision.context import BasicContextCompactor as BasicContextCompactor
    from rsicomposer.harnesses.decision.dsh import (
        DSHConfig as DSHConfig,
        dsh_decision_factory as dsh_decision_factory,
    )
    from rsicomposer.harnesses.decision.policy import ModelDecisionPolicy as ModelDecisionPolicy
    from rsicomposer.harnesses.decision.policy.prompts import DecisionPrompts as DecisionPrompts
    from rsicomposer.harnesses.python_step import (
        HarnessExecutionError as HarnessExecutionError,
        sandbox_step as sandbox_step,
        validate_source as validate_source,
    )
    from rsicomposer.labs.math.verifiers import (
        AnswerCheck as AnswerCheck,
        VERIFIERS as MATH_VERIFIERS,
        AnswerVerifier as AnswerVerifier,
        CompetitionIntegerVerifier as CompetitionIntegerVerifier,
        MarkedNumericVerifier as MarkedNumericVerifier,
        MathVerifyVerifier as MathVerifyVerifier,
        StrictNumericVerifier as StrictNumericVerifier,
        build_verifier as build_verifier,
        final_boxed as final_boxed,
        marked_number as marked_number,
        numeric_answer as numeric_answer,
        parse_reference as parse_reference,
        score_record as score_record,
        validate_verifier as validate_verifier,
    )
    from rsicomposer.loops.contracts.artifacts import ArtifactRef as ArtifactRef
    from rsicomposer.loops.contracts.coordination import OperationResult as OperationResult
    from rsicomposer.loops.contracts.decision import (
        Decision as Decision,
        DecisionInput as DecisionInput,
        DecisionPolicy as DecisionPolicy,
        OperationCall as OperationCall,
        OperationSpec as OperationSpec,
    )
    from rsicomposer.loops.contracts.execution import LabExecution as LabExecution, LabToolError as LabToolError
    from rsicomposer.loops.contracts.jobs import (
        JobOutcome as JobOutcome,
        JobProgress as JobProgress,
        OperationProgress as OperationProgress,
        JobControl as JobControl,
        JobRef as JobRef,
        JobStatus as JobStatus,
    )
    from rsicomposer.loops.contracts.lifecycle import ExecutionStatus as ExecutionStatus
    from rsicomposer.loops.contracts.model_service import (
        BackendBinding as BackendBinding,
        GenerationRequest as GenerationRequest,
        GenerationResponse as GenerationResponse,
        ModelBackend as ModelBackend,
        ModelRef as ModelRef,
        ModelResponseError as ModelResponseError,
        ModelTransportError as ModelTransportError,
        TokenUsage as TokenUsage,
    )
    from rsicomposer.loops.contracts.rsi import (
        ControllerOperationRequirements as ControllerOperationRequirements,
        ControllerReportProvider as ControllerReportProvider,
    )
    from rsicomposer.methods import (
        ControllerDefinition as ControllerDefinition,
        MethodDefinition as MethodDefinition,
        RSIMethod as RSIMethod,
    )
    from rsicomposer.method_state import StatefulLab as StatefulLab, StatefulPolicy as StatefulPolicy
    from rsicomposer.registry import Registry as Registry
    from rsicomposer.harnesses.decision.langmem import LangMemMemory as LangMemMemory
    from rsicomposer.research import MemoryAssembly as MemoryAssembly, ResearchAssembly as ResearchAssembly
    from rsicomposer.runtime.execution.isolation import (
        ExecutionPolicy as ExecutionPolicy,
        ProcessIsolation as ProcessIsolation,
        system_runtime_paths as system_runtime_paths,
    )
    from rsicomposer.runtime.execution.research_bridge import (
        ResearchBridge as ResearchBridge,
        emit_event as emit_event,
    )
    from rsicomposer.runtime.execution.upstream import checkout as checkout
    from rsicomposer.runtime.execution.replay import valid_replay_point as valid_replay_point
    from rsicomposer.runtime.execution.replay import request_digest as request_digest
    from rsicomposer.runtime.execution.replay import ReplayLog as ReplayLog
    from rsicomposer.runtime.execution.replay import ReplayMismatchError as ReplayMismatchError
    from rsicomposer.runtime.execution.upstream_worker import UpstreamWorker as UpstreamWorker
    from rsicomposer.runtime.execution.upstream_policy import UpstreamAgentPolicy as UpstreamAgentPolicy
    from rsicomposer.runtime.execution.workers import (
        run_workers as run_workers,
        worker_environment as worker_environment,
    )
    from rsicomposer.runtime.models.component_access import ComponentModels as ComponentModels
    from rsicomposer.runtime.models.component_endpoint import ComponentModelServer as ComponentModelServer
    from rsicomposer.runtime.models.accounting import AccountedModelBackend as AccountedModelBackend
    from rsicomposer.runtime.observation import (
        current_observations as current_observations,
        observe as observe,
        register_probe as register_probe,
        report_devices as report_devices,
        report_model_calls as report_model_calls,
        report_samples as report_samples,
        report_training_log as report_training_log,
        report_version as report_version,
        report_version_evaluation as report_version_evaluation,
        report_version_selection as report_version_selection,
        trigger_probe as trigger_probe,
    )
    from rsicomposer.runtime.observation.client import log as log
    from rsicomposer.runtime.observation.facts import (
        ObservationProbe as ObservationProbe,
        emit_observation as emit_observation,
        observation_id as observation_id,
    )
    from rsicomposer.runtime.storage.files import (
        atomic_write as atomic_write,
        file_hash as file_hash,
        json_bytes as json_bytes,
    )
    from rsicomposer.runtime.storage.workspace_files import read_workspace_file as read_workspace_file

    from rsicomposer.components import ComponentDefinition as ComponentDefinition
    from rsicomposer.components import ComponentGraph as ComponentGraph
    from rsicomposer.loops.contracts.feedback_context import ControllerMemory as ControllerMemory
    from rsicomposer.loops.contracts.feedback_context import ContextQuery as ContextQuery
    from rsicomposer.loops.contracts.feedback_context import EvidenceEvent as EvidenceEvent
    from rsicomposer.runtime.state.contracts import RecoveryContract as RecoveryContract
    from rsicomposer.runtime.state.contracts import SnapshotContract as SnapshotContract
    from rsicomposer.runtime.state.component_state import AssemblyRecovery as AssemblyRecovery
    from rsicomposer.snapshot.local import LocalSnapshotCapture as LocalSnapshotCapture
    from rsicomposer.snapshot.local import restore_snapshot as restore_snapshot

    from rsicomposer.loops.contracts.harness import ControllerHarness as ControllerHarness
    from rsicomposer.loops.contracts.harness import NullHarness as NullHarness
    from rsicomposer.harnesses.decision.engine import ResearchControllerHarness as ResearchControllerHarness
    from rsicomposer.harnesses.decision.public_policy import ResearchPolicyAdapter as ResearchPolicyAdapter
    from rsicomposer.host.ledger import DecisionResearchMetadata as DecisionResearchMetadata

if _TYPE_CHECKING:
    from rsicomposer.loops.contracts.data import DatasetBinding as DatasetBinding
    from rsicomposer.loops.contracts.profiles import (
        MetricSpec as MetricSpec,
        SplitRole as SplitRole,
        Invariant as Invariant,
        LabProfile as LabProfile,
        get_profile as get_profile,
        TRAIN as TRAIN,
        TRAIN_UNLABELED as TRAIN_UNLABELED,
        DEV as DEV,
        SEALED as SEALED,
        WEIGHT_TRAINING as WEIGHT_TRAINING,
        ROUND_WEIGHT_TRAINING as ROUND_WEIGHT_TRAINING,
        REFERENCE_WEIGHT_TRAINING as REFERENCE_WEIGHT_TRAINING,
        ARTIFACT_SEARCH as ARTIFACT_SEARCH,
        PROGRAM_EXECUTION as PROGRAM_EXECUTION,
        CODING_TASK as CODING_TASK,
    )
