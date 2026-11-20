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
    'SwapPlanner': ('rsisys.loops.contracts.controller_swap', 'SwapPlanner'),
    'NestedRun': ('rsisys.runtime.execution.nested', 'NestedRun'),
    'ReadOnlyEvidenceView': ('rsisys.runtime.evidence.view', 'ReadOnlyEvidenceView'),
    'load_controller_version': ('rsisys.runtime.state.controller_versions', 'load_controller_version'),
    'save_controller_version': ('rsisys.runtime.state.controller_versions', 'save_controller_version'),
    'EvidenceScope': ('rsisys.loops.contracts.controller_version', 'EvidenceScope'),
    'ControllerVersion': ('rsisys.loops.contracts.controller_version', 'ControllerVersion'),
    'HostedLoRAEngine': ('rsisys.backends.training.lora.hosted', 'HostedLoRAEngine'),
    'process_identity': ('rsisys.runtime.execution.process_identity', 'process_identity'),
    'WeightSync': ('rsisys.runtime.execution.weight_sync', 'WeightSync'),
    'FilesystemWeightSync': ('rsisys.runtime.execution.weight_sync', 'FilesystemWeightSync'),
    'ResourceRequest': ('rsisys.runtime.execution.resources', 'ResourceRequest'),
    'WorkerLaunchUncertainError': ('rsisys.runtime.execution.errors', 'WorkerLaunchUncertainError'),
    'WorkerLostError': ('rsisys.runtime.execution.errors', 'WorkerLostError'),
    'socket_relay_command': ('rsisys.runtime.execution.socket_relay', 'socket_relay_command'),
    'HostWorkerServices': ('rsisys.runtime.execution.remote_services', 'HostWorkerServices'),
    'RemoteArtifacts': ('rsisys.runtime.execution.remote_services', 'RemoteArtifacts'),
    'ResourceManager': ('rsisys.runtime.execution.resources', 'ResourceManager'),
    'Lease': ('rsisys.runtime.execution.resources', 'Lease'),
    'Executor': ('rsisys.runtime.execution.executor', 'Executor'),
    'WorkerSpec': ('rsisys.runtime.execution.executor', 'WorkerSpec'),
    'WorkerHandle': ('rsisys.runtime.execution.executor', 'WorkerHandle'),
    'execution_capabilities': ('rsisys.runtime.execution.executor', 'execution_capabilities'),
    'run_cpu': ('rsisys.runtime.execution.executor', 'run_cpu'),
    'HostBudget': ('rsisys.runtime.execution.budget', 'HostBudget'),
    'BudgetExhausted': ('rsisys.runtime.execution.budget', 'BudgetExhausted'),

    'DatasetBinding': ('rsisys.loops.contracts.data', 'DatasetBinding'),
    'MetricSpec': ('rsisys.loops.contracts.profiles', 'MetricSpec'),
    'SplitRole': ('rsisys.loops.contracts.profiles', 'SplitRole'),
    'Invariant': ('rsisys.loops.contracts.profiles', 'Invariant'),
    'LabProfile': ('rsisys.loops.contracts.profiles', 'LabProfile'),
    'get_profile': ('rsisys.loops.contracts.profiles', 'get_profile'),
    'TRAIN': ('rsisys.loops.contracts.profiles', 'TRAIN'),
    'TRAIN_UNLABELED': ('rsisys.loops.contracts.profiles', 'TRAIN_UNLABELED'),
    'DEV': ('rsisys.loops.contracts.profiles', 'DEV'),
    'SEALED': ('rsisys.loops.contracts.profiles', 'SEALED'),
    'WEIGHT_TRAINING': ('rsisys.loops.contracts.profiles', 'WEIGHT_TRAINING'),
    'ARTIFACT_SEARCH': ('rsisys.loops.contracts.profiles', 'ARTIFACT_SEARCH'),
    'PROGRAM_EXECUTION': ('rsisys.loops.contracts.profiles', 'PROGRAM_EXECUTION'),
    'CODING_TASK': ('rsisys.loops.contracts.profiles', 'CODING_TASK'),

    # Explicit composition, memory and state integration ports.
    'NullHarness': ('rsisys.loops.contracts.harness', 'NullHarness'),
    'ControllerHarness': ('rsisys.loops.contracts.harness', 'ControllerHarness'),
    'ResearchPolicyAdapter': ('rsisys.harnesses.decision.public_policy', 'ResearchPolicyAdapter'),
    'DecisionResearchMetadata': ('rsisys.host.ledger', 'DecisionResearchMetadata'),
    'ResearchControllerHarness': ('rsisys.harnesses.decision.engine', 'ResearchControllerHarness'),
    'ComponentDefinition': ('rsisys.components', 'ComponentDefinition'),
    'ComponentGraph': ('rsisys.components', 'ComponentGraph'),
    'LangMemMemory': ('rsisys.harnesses.decision.langmem', 'LangMemMemory'),
    'ControllerMemory': ('rsisys.loops.contracts.feedback_context', 'ControllerMemory'),
    'ContextQuery': ('rsisys.loops.contracts.feedback_context', 'ContextQuery'),
    'EvidenceEvent': ('rsisys.loops.contracts.feedback_context', 'EvidenceEvent'),
    'RecoveryContract': ('rsisys.runtime.state.contracts', 'RecoveryContract'),
    'SnapshotContract': ('rsisys.runtime.state.contracts', 'SnapshotContract'),
    'AssemblyRecovery': ('rsisys.runtime.state.component_state', 'AssemblyRecovery'),
    'LocalSnapshotCapture': ('rsisys.snapshot.local', 'LocalSnapshotCapture'),
    'restore_snapshot': ('rsisys.snapshot.local', 'restore_snapshot'),

    # Application and method assembly.
    'Application': ('rsisys.application', 'Application'),
    'ApplicationConfig': ('rsisys.application', 'ApplicationConfig'),
    'AssemblyContext': ('rsisys.assembly', 'AssemblyContext'),
    'MethodDefinition': ('rsisys.methods', 'MethodDefinition'),
    'ControllerDefinition': ('rsisys.methods', 'ControllerDefinition'),
    'StatefulPolicy': ('rsisys.method_state', 'StatefulPolicy'),
    'StatefulLab': ('rsisys.method_state', 'StatefulLab'),
    'MemoryAssembly': ('rsisys.research', 'MemoryAssembly'),
    'ModuleAssembly': ('rsisys.assembly', 'ModuleAssembly'),
    'RSIMethod': ('rsisys.methods', 'RSIMethod'),
    'Registry': ('rsisys.registry', 'Registry'),
    'ResearchAssembly': ('rsisys.research', 'ResearchAssembly'),
    'build_default_decision': ('rsisys.assembly', 'build_default_decision'),
    # Controller, Lab, artifact and job contracts.
    'ArtifactRef': ('rsisys.loops.contracts.artifacts', 'ArtifactRef'),
    'BackendBinding': ('rsisys.loops.contracts.model_service', 'BackendBinding'),
    'ControllerOperationRequirements': ('rsisys.loops.contracts.rsi', 'ControllerOperationRequirements'),
    'ControllerReportProvider': ('rsisys.loops.contracts.rsi', 'ControllerReportProvider'),
    'Decision': ('rsisys.loops.contracts.decision', 'Decision'),
    'DecisionInput': ('rsisys.loops.contracts.decision', 'DecisionInput'),
    'DecisionPolicy': ('rsisys.loops.contracts.decision', 'DecisionPolicy'),
    'ExecutionStatus': ('rsisys.loops.contracts.lifecycle', 'ExecutionStatus'),
    'GenerationRequest': ('rsisys.loops.contracts.model_service', 'GenerationRequest'),
    'GenerationResponse': ('rsisys.loops.contracts.model_service', 'GenerationResponse'),
    'JobOutcome': ('rsisys.loops.contracts.jobs', 'JobOutcome'),
    'OperationProgress': ('rsisys.loops.contracts.jobs', 'OperationProgress'),
    'JobControl': ('rsisys.loops.contracts.jobs', 'JobControl'),
    'JobProgress': ('rsisys.loops.contracts.jobs', 'JobProgress'),
    'JobRef': ('rsisys.loops.contracts.jobs', 'JobRef'),
    'JobStatus': ('rsisys.loops.contracts.jobs', 'JobStatus'),
    'LabExecution': ('rsisys.loops.contracts.execution', 'LabExecution'),
    'LabToolError': ('rsisys.loops.contracts.execution', 'LabToolError'),
    'ModelBackend': ('rsisys.loops.contracts.model_service', 'ModelBackend'),
    'ModelRef': ('rsisys.loops.contracts.model_service', 'ModelRef'),
    'ModelResponseError': ('rsisys.loops.contracts.model_service', 'ModelResponseError'),
    'ModelTransportError': ('rsisys.loops.contracts.model_service', 'ModelTransportError'),
    'OperationCall': ('rsisys.loops.contracts.decision', 'OperationCall'),
    'OperationResult': ('rsisys.loops.contracts.coordination', 'OperationResult'),
    'OperationSpec': ('rsisys.loops.contracts.decision', 'OperationSpec'),
    'TokenUsage': ('rsisys.loops.contracts.model_service', 'TokenUsage'),
    # Observation from the host and worker processes.
    'current_observations': ('rsisys.runtime.observation', 'current_observations'),
    'log': ('rsisys.runtime.observation.client', 'log'),
    'observe': ('rsisys.runtime.observation', 'observe'),
    'register_probe': ('rsisys.runtime.observation', 'register_probe'),
    'report_devices': ('rsisys.runtime.observation', 'report_devices'),
    'report_model_calls': ('rsisys.runtime.observation', 'report_model_calls'),
    'report_samples': ('rsisys.runtime.observation', 'report_samples'),
    'report_training_log': ('rsisys.runtime.observation', 'report_training_log'),
    'trigger_probe': ('rsisys.runtime.observation', 'trigger_probe'),
    # The math Lab's answer verifiers, shared by math methods (Math-Verify is optional).
    'AnswerCheck': ('rsisys.labs.math.verifiers', 'AnswerCheck'),
    'MATH_VERIFIERS': ('rsisys.labs.math.verifiers', 'VERIFIERS'),
    'AnswerVerifier': ('rsisys.labs.math.verifiers', 'AnswerVerifier'),
    'CompetitionIntegerVerifier': ('rsisys.labs.math.verifiers', 'CompetitionIntegerVerifier'),
    'MarkedNumericVerifier': ('rsisys.labs.math.verifiers', 'MarkedNumericVerifier'),
    'MathVerifyVerifier': ('rsisys.labs.math.verifiers', 'MathVerifyVerifier'),
    'StrictNumericVerifier': ('rsisys.labs.math.verifiers', 'StrictNumericVerifier'),
    'build_verifier': ('rsisys.labs.math.verifiers', 'build_verifier'),
    'final_boxed': ('rsisys.labs.math.verifiers', 'final_boxed'),
    'marked_number': ('rsisys.labs.math.verifiers', 'marked_number'),
    'numeric_answer': ('rsisys.labs.math.verifiers', 'numeric_answer'),
    'parse_reference': ('rsisys.labs.math.verifiers', 'parse_reference'),
    'score_record': ('rsisys.labs.math.verifiers', 'score_record'),
    'validate_verifier': ('rsisys.labs.math.verifiers', 'validate_verifier'),
    # Containerized coding benchmark provider (Harbor is optional at runtime).
    'MathLab': ('rsisys.labs.math.lab', 'MathLab'),
    'MathAttempt': ('rsisys.labs.math.lab', 'Attempt'),
    'CheckpointRuntime': ('rsisys.labs.math.runtimes', 'CheckpointRuntime'),
    'InstructionRuntime': ('rsisys.labs.math.runtimes', 'InstructionRuntime'),
    'StepProgramRuntime': ('rsisys.labs.math.runtimes', 'StepProgramRuntime'),
    'CodingLab': ('rsisys.labs.coding.lab', 'CodingLab'),
    'CodingArtifact': ('rsisys.labs.coding.artifacts', 'CodingArtifact'),
    'HarborAgentRuntime': ('rsisys.labs.coding.agents', 'HarborAgentRuntime'),
    'Benchmark': ('rsisys.loops.contracts.benchmark', 'Benchmark'),
    'AgentRuntime': ('rsisys.loops.contracts.agent_runtime', 'AgentRuntime'),
    'Subject': ('rsisys.loops.contracts.agent_runtime', 'Subject'),
    'CandidateRejected': ('rsisys.loops.contracts.agent_runtime', 'CandidateRejected'),
    'CandidateSpec': ('rsisys.loops.contracts.agent_runtime', 'CandidateSpec'),
    # Optional serving and training implementations.
    'ChatPool': ('rsisys.backends.serving.chat', 'ChatPool'),
    'LoRAEngine': ('rsisys.backends.training.lora.engine', 'LoRAEngine'),
    'ParallelLoRAEngine': ('rsisys.backends.training.lora.parallel', 'ParallelLoRAEngine'),
    'VLLMBackend': ('rsisys.backends.serving.vllm', 'VLLMBackend'),
    'WeightEngine': ('rsisys.backends.training.engine', 'WeightEngine'),
    'load_model': ('rsisys.backends.training.worker', 'load_model'),
    'merged_change_count': ('rsisys.backends.training.worker', 'merged_change_count'),
    'tensor_batch': ('rsisys.backends.training.worker', 'tensor_batch'),
    'validate_engine': ('rsisys.backends.training.engine', 'validate_engine'),
    'weights_digest': ('rsisys.backends.training.worker', 'weights_digest'),
    # Optional harnesses.
    'BasicContextCompactor': ('rsisys.harnesses.decision.context', 'BasicContextCompactor'),
    'DSHConfig': ('rsisys.harnesses.decision.dsh', 'DSHConfig'),
    'DecisionPrompts': ('rsisys.harnesses.decision.policy.prompts', 'DecisionPrompts'),
    'HarnessExecutionError': ('rsisys.harnesses.python_step', 'HarnessExecutionError'),
    'ModelDecisionPolicy': ('rsisys.harnesses.decision.policy', 'ModelDecisionPolicy'),
    'dsh_decision_factory': ('rsisys.harnesses.decision.dsh', 'dsh_decision_factory'),
    'sandbox_step': ('rsisys.harnesses.python_step', 'sandbox_step'),
    'validate_source': ('rsisys.harnesses.python_step', 'validate_source'),
    # Execution and configuration helpers.
    'ComponentModels': ('rsisys.runtime.models.component_access', 'ComponentModels'),
    'ComponentModelServer': ('rsisys.runtime.models.component_endpoint', 'ComponentModelServer'),
    'AccountedModelBackend': ('rsisys.runtime.models.accounting', 'AccountedModelBackend'),
    'ExecutionPolicy': ('rsisys.runtime.execution.isolation', 'ExecutionPolicy'),
    'ProcessIsolation': ('rsisys.runtime.execution.isolation', 'ProcessIsolation'),
    'ResearchBridge': ('rsisys.runtime.execution.research_bridge', 'ResearchBridge'),
    'valid_replay_point': ('rsisys.runtime.execution.replay', 'valid_replay_point'),
    'request_digest': ('rsisys.runtime.execution.replay', 'request_digest'),
    'ReplayLog': ('rsisys.runtime.execution.replay', 'ReplayLog'),
    'ReplayMismatchError': ('rsisys.runtime.execution.replay', 'ReplayMismatchError'),
    'UpstreamWorker': ('rsisys.runtime.execution.upstream_worker', 'UpstreamWorker'),
    'UpstreamAgentPolicy': ('rsisys.runtime.execution.upstream_policy', 'UpstreamAgentPolicy'),
    'atomic_write': ('rsisys.runtime.storage.files', 'atomic_write'),
    'checkout': ('rsisys.runtime.execution.upstream', 'checkout'),
    'config_json': ('rsisys.components', 'config_json'),
    'emit_event': ('rsisys.runtime.execution.research_bridge', 'emit_event'),
    'file_hash': ('rsisys.runtime.storage.files', 'file_hash'),
    'json_bytes': ('rsisys.runtime.storage.files', 'json_bytes'),
    'read_workspace_file': ('rsisys.runtime.storage.workspace_files', 'read_workspace_file'),
    'run_workers': ('rsisys.runtime.execution.workers', 'run_workers'),
    'system_runtime_paths': ('rsisys.runtime.execution.isolation', 'system_runtime_paths'),
    'worker_environment': ('rsisys.runtime.execution.workers', 'worker_environment'),
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
    from rsisys.loops.contracts.controller_swap import SwapPlanner as SwapPlanner
    from rsisys.runtime.execution.nested import NestedRun as NestedRun
    from rsisys.runtime.evidence.view import ReadOnlyEvidenceView as ReadOnlyEvidenceView
    from rsisys.runtime.state.controller_versions import load_controller_version as load_controller_version
    from rsisys.runtime.state.controller_versions import save_controller_version as save_controller_version
    from rsisys.loops.contracts.controller_version import EvidenceScope as EvidenceScope
    from rsisys.loops.contracts.controller_version import ControllerVersion as ControllerVersion
    from rsisys.backends.training.lora.hosted import HostedLoRAEngine as HostedLoRAEngine
    from rsisys.runtime.execution.process_identity import process_identity as process_identity
    from rsisys.runtime.execution.weight_sync import (
        WeightSync as WeightSync,
        FilesystemWeightSync as FilesystemWeightSync,
    )
    from rsisys.runtime.execution.resources import ResourceRequest as ResourceRequest
    from rsisys.runtime.execution.errors import WorkerLostError as WorkerLostError
    from rsisys.runtime.execution.errors import WorkerLaunchUncertainError as WorkerLaunchUncertainError
    from rsisys.runtime.execution.socket_relay import socket_relay_command as socket_relay_command
    from rsisys.runtime.execution.remote_services import HostWorkerServices as HostWorkerServices
    from rsisys.runtime.execution.remote_services import RemoteArtifacts as RemoteArtifacts
    from rsisys.runtime.execution.resources import ResourceManager as ResourceManager
    from rsisys.runtime.execution.resources import Lease as Lease
    from rsisys.runtime.execution.executor import Executor as Executor
    from rsisys.runtime.execution.executor import WorkerSpec as WorkerSpec
    from rsisys.runtime.execution.executor import WorkerHandle as WorkerHandle
    from rsisys.runtime.execution.executor import execution_capabilities as execution_capabilities
    from rsisys.runtime.execution.executor import run_cpu as run_cpu
    from rsisys.runtime.execution.budget import HostBudget as HostBudget
    from rsisys.runtime.execution.budget import BudgetExhausted as BudgetExhausted

    from rsisys.labs.math.lab import MathLab as MathLab
    from rsisys.labs.math.lab import Attempt as MathAttempt
    from rsisys.labs.math.runtimes import (
        CheckpointRuntime as CheckpointRuntime, InstructionRuntime as InstructionRuntime,
        StepProgramRuntime as StepProgramRuntime,
    )
    from rsisys.labs.coding.lab import CodingLab as CodingLab
    from rsisys.labs.coding.artifacts import CodingArtifact as CodingArtifact
    from rsisys.labs.coding.agents import HarborAgentRuntime as HarborAgentRuntime
    from rsisys.loops.contracts.benchmark import Benchmark as Benchmark
    from rsisys.loops.contracts.agent_runtime import (
        AgentRuntime as AgentRuntime, Subject as Subject, CandidateRejected as CandidateRejected,
        CandidateSpec as CandidateSpec,
    )
    from rsisys.application import Application as Application, ApplicationConfig as ApplicationConfig
    from rsisys.assembly import (
        AssemblyContext as AssemblyContext,
        ModuleAssembly as ModuleAssembly,
        build_default_decision as build_default_decision,
    )
    from rsisys.backends.serving.chat import ChatPool as ChatPool
    from rsisys.backends.serving.vllm import VLLMBackend as VLLMBackend
    from rsisys.backends.training.engine import (
        WeightEngine as WeightEngine,
        validate_engine as validate_engine,
    )
    from rsisys.backends.training.lora.engine import LoRAEngine as LoRAEngine
    from rsisys.backends.training.lora.parallel import ParallelLoRAEngine as ParallelLoRAEngine
    from rsisys.backends.training.worker import (
        load_model as load_model,
        merged_change_count as merged_change_count,
        tensor_batch as tensor_batch,
        weights_digest as weights_digest,
    )
    from rsisys.components import config_json as config_json
    from rsisys.harnesses.decision.context import BasicContextCompactor as BasicContextCompactor
    from rsisys.harnesses.decision.dsh import (
        DSHConfig as DSHConfig,
        dsh_decision_factory as dsh_decision_factory,
    )
    from rsisys.harnesses.decision.policy import ModelDecisionPolicy as ModelDecisionPolicy
    from rsisys.harnesses.decision.policy.prompts import DecisionPrompts as DecisionPrompts
    from rsisys.harnesses.python_step import (
        HarnessExecutionError as HarnessExecutionError,
        sandbox_step as sandbox_step,
        validate_source as validate_source,
    )
    from rsisys.labs.math.verifiers import (
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
    from rsisys.loops.contracts.artifacts import ArtifactRef as ArtifactRef
    from rsisys.loops.contracts.coordination import OperationResult as OperationResult
    from rsisys.loops.contracts.decision import (
        Decision as Decision,
        DecisionInput as DecisionInput,
        DecisionPolicy as DecisionPolicy,
        OperationCall as OperationCall,
        OperationSpec as OperationSpec,
    )
    from rsisys.loops.contracts.execution import LabExecution as LabExecution, LabToolError as LabToolError
    from rsisys.loops.contracts.jobs import (
        JobOutcome as JobOutcome,
        JobProgress as JobProgress,
        OperationProgress as OperationProgress,
        JobControl as JobControl,
        JobRef as JobRef,
        JobStatus as JobStatus,
    )
    from rsisys.loops.contracts.lifecycle import ExecutionStatus as ExecutionStatus
    from rsisys.loops.contracts.model_service import (
        BackendBinding as BackendBinding,
        GenerationRequest as GenerationRequest,
        GenerationResponse as GenerationResponse,
        ModelBackend as ModelBackend,
        ModelRef as ModelRef,
        ModelResponseError as ModelResponseError,
        ModelTransportError as ModelTransportError,
        TokenUsage as TokenUsage,
    )
    from rsisys.loops.contracts.rsi import (
        ControllerOperationRequirements as ControllerOperationRequirements,
        ControllerReportProvider as ControllerReportProvider,
    )
    from rsisys.methods import (
        ControllerDefinition as ControllerDefinition,
        MethodDefinition as MethodDefinition,
        RSIMethod as RSIMethod,
    )
    from rsisys.method_state import StatefulLab as StatefulLab, StatefulPolicy as StatefulPolicy
    from rsisys.registry import Registry as Registry
    from rsisys.harnesses.decision.langmem import LangMemMemory as LangMemMemory
    from rsisys.research import MemoryAssembly as MemoryAssembly, ResearchAssembly as ResearchAssembly
    from rsisys.runtime.execution.isolation import (
        ExecutionPolicy as ExecutionPolicy,
        ProcessIsolation as ProcessIsolation,
        system_runtime_paths as system_runtime_paths,
    )
    from rsisys.runtime.execution.research_bridge import (
        ResearchBridge as ResearchBridge,
        emit_event as emit_event,
    )
    from rsisys.runtime.execution.upstream import checkout as checkout
    from rsisys.runtime.execution.replay import valid_replay_point as valid_replay_point
    from rsisys.runtime.execution.replay import request_digest as request_digest
    from rsisys.runtime.execution.replay import ReplayLog as ReplayLog
    from rsisys.runtime.execution.replay import ReplayMismatchError as ReplayMismatchError
    from rsisys.runtime.execution.upstream_worker import UpstreamWorker as UpstreamWorker
    from rsisys.runtime.execution.upstream_policy import UpstreamAgentPolicy as UpstreamAgentPolicy
    from rsisys.runtime.execution.workers import (
        run_workers as run_workers,
        worker_environment as worker_environment,
    )
    from rsisys.runtime.models.component_access import ComponentModels as ComponentModels
    from rsisys.runtime.models.component_endpoint import ComponentModelServer as ComponentModelServer
    from rsisys.runtime.models.accounting import AccountedModelBackend as AccountedModelBackend
    from rsisys.runtime.observation import (
        current_observations as current_observations,
        observe as observe,
        register_probe as register_probe,
        report_devices as report_devices,
        report_model_calls as report_model_calls,
        report_samples as report_samples,
        report_training_log as report_training_log,
        trigger_probe as trigger_probe,
    )
    from rsisys.runtime.observation.client import log as log
    from rsisys.runtime.storage.files import (
        atomic_write as atomic_write,
        file_hash as file_hash,
        json_bytes as json_bytes,
    )
    from rsisys.runtime.storage.workspace_files import read_workspace_file as read_workspace_file

    from rsisys.components import ComponentDefinition as ComponentDefinition
    from rsisys.components import ComponentGraph as ComponentGraph
    from rsisys.loops.contracts.feedback_context import ControllerMemory as ControllerMemory
    from rsisys.loops.contracts.feedback_context import ContextQuery as ContextQuery
    from rsisys.loops.contracts.feedback_context import EvidenceEvent as EvidenceEvent
    from rsisys.runtime.state.contracts import RecoveryContract as RecoveryContract
    from rsisys.runtime.state.contracts import SnapshotContract as SnapshotContract
    from rsisys.runtime.state.component_state import AssemblyRecovery as AssemblyRecovery
    from rsisys.snapshot.local import LocalSnapshotCapture as LocalSnapshotCapture
    from rsisys.snapshot.local import restore_snapshot as restore_snapshot

    from rsisys.loops.contracts.harness import ControllerHarness as ControllerHarness
    from rsisys.loops.contracts.harness import NullHarness as NullHarness
    from rsisys.harnesses.decision.engine import ResearchControllerHarness as ResearchControllerHarness
    from rsisys.harnesses.decision.public_policy import ResearchPolicyAdapter as ResearchPolicyAdapter
    from rsisys.host.ledger import DecisionResearchMetadata as DecisionResearchMetadata

if _TYPE_CHECKING:
    from rsisys.loops.contracts.data import DatasetBinding as DatasetBinding
    from rsisys.loops.contracts.profiles import (
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
        ARTIFACT_SEARCH as ARTIFACT_SEARCH,
        PROGRAM_EXECUTION as PROGRAM_EXECUTION,
        CODING_TASK as CODING_TASK,
    )
