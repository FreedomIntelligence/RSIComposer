"""Real medical meta-agent over the replaceable Open Lab assembly.

This launcher owns time, admission, evidence and device routing. Algorithms stay
in model-written programs; native Lab outputs are observations, not validated
candidate adoption. Use only in a trusted, scheduler-isolated allocation.
"""
import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback

from rsisys.adapters.local_lab_memory import LocalLabMemoryAdmission
from rsisys.adapters.local_lab_feedback import RoutedLab, ContextOrMemoryAdmission, ImmediateFeedbackWindow
from .fixed_evaluation import PROTOCOL
from .protocol_evaluation import ProtocolMedicalLab, PROTOCOL as EVALUATION_PROTOCOL, BUNDLE_SCHEMA
from rsisys.adapters.serving import VLLMBackend
from rsisys.application import Application, ApplicationConfig
from rsisys.assembly import ModuleAssembly, build_default_decision
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.policy.prompts import DecisionPrompts
from rsisys.research import ResearchAssembly
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.adapters.local_lab_state import RetainedLocalProcessBackend
from rsisys.runtime.model import ModelResponseError
from rsisys.runtime.run_policy import TimeRunPolicy, RunDecision
from rsisys.methods import MethodCatalog, MethodDefinition
from rsisys.runtime.isolation import ExecutionPolicy, IsolatedProcessBackend, system_runtime_paths
from rsisys.runtime.runtime_identity import inference_policy
from rsisys.runtime.model_session import HostModelSession, ModelSessionBroker
from rsisys.runtime.resource_account import ResourceAccount
from .open_lab_evolution import EvolutionLabSlot, bind_evolution


def write(path, value):
    atomic_write(Path(path), json_bytes(value))


def append(path, value):
    with Path(path).open('ab') as stream:
        stream.write(json_bytes(value) + b'\n')


class AbsoluteHostPolicy:
    def __init__(self, backend, seconds, deadline_epoch=None):
        self.backend, self.timed, self.deadline_epoch = backend, TimeRunPolicy(seconds), deadline_epoch

    def evaluate(self, context):
        if self.backend.stop_requested or (self.deadline_epoch is not None and time.time() >= self.deadline_epoch):
            return RunDecision('stop', 'absolute_host_deadline_or_signal', 'cancel')
        return self.timed.evaluate(context)


def sglang_template_tools(request):
    """Mirror the deployed SGLang 0.5.20 Tool.model_dump() before rendering.

    The server adds function.strict=False and tool.defer_loading=None; counting
    the raw wire declarations undercounts their rendered schema. This belongs
    to this deployment, rather than every interchangeable model backend.
    """
    if request.tool_choice == 'none':
        return []
    tools = request.chat_tools().get('tools', [])
    if isinstance(request.tool_choice, dict):
        name = request.tool_choice['function']['name']
        tools = [tool for tool in tools if tool['function']['name'] == name]
    normalized = []
    for tool in tools:
        raw = tool['function']
        function = {'description': raw.get('description'), 'name': raw['name'],
                    'parameters': raw.get('parameters'), 'strict': raw.get('strict', False)}
        deferred = raw.get('defer_loading')
        if deferred is None:
            deferred = tool.get('defer_loading')
        if deferred is not None:
            function['defer_loading'] = deferred
        normalized.append({'type': tool['type'], 'function': function,
                           'defer_loading': tool.get('defer_loading')})
    return normalized


# Compatibility import; every method uses the common host model session.
from rsisys.runtime.accounting import AccountedModelBackend, HostAdmissionStop


class RecordedBackend(AccountedModelBackend):
    """Medical integration's bounded JSON repair, sharing the host account."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, response_attempts=3, **kwargs)


class TimedOpenPolicy(ModelDecisionPolicy):
    """Host lifetime, open operations; no idle/finish decision envelopes."""
    def bind_reflection(self, reflection):
        self.scheduled_reflection = reflection

    def _context_inputs(self, context):
        return super()._context_inputs(replace(context, run_control={'stopping_owner': 'host'}))

    def _summary_current_state(self, context):
        return super()._summary_current_state(replace(context, run_control={'stopping_owner': 'host'}))

    def decision_schema(self, context):
        schema = super().decision_schema(context)
        schema['oneOf'] = [branch for branch in schema['oneOf']
            if branch['properties']['kind']['const'] not in ('wait', 'finish')]
        return schema

    def _decision_request(self, context):
        return replace(super()._decision_request(context), max_output_tokens=8192)

    def _parse_decision(self, response, context):
        decision = super()._parse_decision(response, context)
        if decision.kind not in ('invoke', 'explore'):
            raise ValueError('Select an available operation; the host owns stopping')
        if getattr(self, 'scheduled_reflection', None) is not None:
            self.scheduled_reflection.validate_decision(decision, context)
        return decision


def host_context_compactor(config):
    """Separate compression goals from capacity, reserving the full reply cap."""
    window = config.get('context_window_tokens', 65536)
    safety = config.get('context_safety_tokens', 1024)
    if type(window) is not int or window <= 8192 or type(safety) is not int or safety < 0:
        raise ValueError('Declare a valid service context window and nonnegative context safety reserve')
    budget = config.get('context_budget', {})
    allowed = {'trigger_tokens', 'target_tokens', 'summary_tokens',
               'max_target_tokens', 'target_increment_tokens'}
    if not isinstance(budget, dict) or set(budget) - allowed:
        raise ValueError('Unknown host context compression configuration')
    return BasicContextCompactor(**{
        'trigger_tokens': 48000, 'target_tokens': 16000, 'summary_tokens': 2048,
        'max_target_tokens': 40000, 'target_increment_tokens': 12000, **budget},
        input_budget_tokens=window - 8192 - safety)


class AllocatedProcessBackend(LocalProcessBackend):
    """Explicit host device/environment injection into scrubbed Lab children."""
    def __init__(self, root, environment, *, device_type='cuda'):
        super().__init__(root, max_output_bytes=128000)
        self.environment = dict(environment)
        device_key = {'cuda': 'CUDA_VISIBLE_DEVICES', 'npu': 'ASCEND_RT_VISIBLE_DEVICES'}.get(device_type)
        if device_key is None or not self.environment.get(device_key):
            raise ValueError('An explicit allocated device mapping is required')
        if any(not isinstance(k, str) or not isinstance(v, str) or '=' in k or '\x00' in k + v
               for k, v in self.environment.items()):
            raise ValueError('Invalid host environment')

    def exec(self, sandbox, request):
        argv = ('/usr/bin/env', *(f'{k}={v}' for k, v in sorted(self.environment.items())), *request.argv)
        return super().exec(sandbox, replace(request, argv=argv))


class RetainedAllocatedProcessBackend(AllocatedProcessBackend, RetainedLocalProcessBackend):
    """The same explicit device routing, with retained local executor state."""


class RetainedIsolatedAllocatedProcessBackend(IsolatedProcessBackend, RetainedLocalProcessBackend):
    """Real execution protection plus the same retained process ownership."""


def allocated_resources(environment, device_type, declaration=None):
    """Describe the host assignment, checking it against actual child routing.

    Visibility is supplied by the operator, never discovered as an authorization.
    Memory and other host devices are observations, not additional allocation.
    """
    from .experiment import copy_resource_allocation
    key = {'cuda': 'CUDA_VISIBLE_DEVICES', 'npu': 'ASCEND_RT_VISIBLE_DEVICES'}[device_type]
    physical = environment.get(key, '').split(',')
    if any(not item or item != item.strip() for item in physical) or len(set(physical)) != len(physical):
        raise ValueError('An explicit distinct allocated device mapping is required')
    declaration = copy_resource_allocation(declaration) or {}
    if ('physical_devices' in declaration
            and [str(item) for item in declaration['physical_devices']] != physical):
        raise ValueError('Declared allocation differs from the Lab process device mapping')
    logical = list(range(len(physical)))
    if 'logical_devices' in declaration and declaration['logical_devices'] != logical:
        raise ValueError('Declared logical devices differ from the Lab process device mapping')
    return {**declaration, 'device_type': device_type, 'device_count': len(physical),
        'physical_devices': physical, 'logical_devices': logical,
        'logical_to_physical': dict(zip(map(str, logical), physical)),
        'visibility_environment': {key: environment[key]},
        'allocation_status': 'host_assigned',
        'memory_availability': declaration.get('memory_availability', 'unknown'),
        'external_read_only_services': declaration.get('external_read_only_services', []),
        'scope': 'Use only these assigned devices. Logical device IDs apply inside generated programs. '
            'One program may use multiple assigned devices with the installed accelerator framework. '
            'Memory figures are timestamped observations, not guarantees of current free memory. '
            'Visibility is routing, not OS isolation. External services grant inference access only; '
            'do not stop, reload, reconfigure or train on their reserved devices.'}


CANDIDATE_ADAPTER = r'''import importlib.util, json
from pathlib import Path
cfg = json.loads(Path('lab_config.json').read_text())
adapter_path = cfg['candidate_input_adapter']['path']
spec = importlib.util.spec_from_file_location('medical_candidate_adapter', adapter_path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
'''

CANDIDATE = r'''"""Exploratory inference example, not a formal evaluation implementation.

Read lab_config.json for model paths, installed Python and allocated devices.
Choose direct inference or deploy your own service using Lab-owned programs.
For a persistent service, submit it with lab_start and collect its logs; run
your own exploratory measurement, then stop the service with lab_cancel.
Use request.render(tokenizer, enable_thinking=False) for local inference; tokenize
that returned text. For a service use request.chat_messages() as its messages.
Both preserve the user question/options. Never extract messages[0]: it is system.
Optional infer_batch(requests) may prepare requests then generate them together.
Formal medical_evaluate accepts a frozen submission bundle containing inference code and weights.
There is no predeployed candidate endpoint and no answer/scoring implementation.
"""
''' + CANDIDATE_ADAPTER + r'''
# Load/cache the model and tokenizer once, or deploy your own Lab-owned service.
def infer(request):
    # prompt = request.render(tokenizer, enable_thinking=False)
    # Tokenize prompt, generate, and return only the newly generated answer text.
    raise NotImplementedError('Implement model inference or deploy a Lab-owned inference service first')

if __name__ == '__main__':
    adapter.run(infer)  # For batching: adapter.run(infer_batch=infer_batch)
'''


EXTERNAL_CANDIDATE = r'''"""Exploratory external-service example; not used by formal evaluation."""
''' + CANDIDATE_ADAPTER + r'''
from urllib.request import Request, urlopen
def infer(request):
    body={'model':cfg['served_model'], 'messages':request.chat_messages(), 'max_tokens':32, 'temperature':0,
        'chat_template_kwargs':{'enable_thinking':False}}
    with urlopen(Request(cfg['endpoint']+'/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'}),timeout=120) as response:
        result=json.load(response)
    if result['model'] != cfg['served_model']: raise ValueError('Provider identity mismatch')
    return result['choices'][0]['message']['content']
if __name__ == '__main__':
    adapter.run(infer)
'''



RESOURCE_POOL_README = '''# Allocated research workspace

Read lab_config.json for the assigned devices, installed Python, available model
inventories and fixed data paths. You choose the research programs and device
layout. Nothing is running initially and no candidate implementation is selected.

Use lab_start with your own argv for environment queries, training, checkpoint
loading, inference, evaluation experiments or other research programs. One program
may use multiple assigned devices. You may query status/logs immediately, or use
the optional bounded lab_poll wait. Collect terminal results with lab_result.

Training code, optimizer choices, checkpoints and any service deployment belong
to you. A service is optional and occupies a program slot. Use only assigned
devices and the installed environment; preserve external services.

For formal medical scores, medical_evaluate accepts a frozen submission directory
with submission.json, your inference Python, and checkpoint/adapter files or a
registered base reference. You own loading, prompting and decoding. The host owns
data, exact-letter scoring, resources and submission admission. Public evaluation
supports debugging. Closed evaluation requires an isolated gateway, returns only
aggregate scores/status and accepts at most one submission per 1800 seconds
across the campaign. Read fixed_evaluation in lab_config.json. Release other
programs before evaluation. Compare matching protocol and dataset hashes.
Unverified stdout is exploratory. Retain real optimizer, loss, weight,
loading and evaluation evidence before claiming training or improvement.
'''


def main(argv=None):
    from .open_lab_recovery import controller_lock, example_identity, inspect_recovery
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--task')
    parser.add_argument('--model-path', help='Installed meta-agent/tokenizer weights; does not select a resource-pool candidate')
    parser.add_argument('--python', help='Installed accelerator/Transformers Python for generated programs')
    parser.add_argument('--endpoint')
    parser.add_argument('--model-id', default='rsisys-meta-qwen35-9b')
    parser.add_argument('--model-revision', default='lion-local-qwen35-9b')
    parser.add_argument('--lab-endpoint', help='Optional external read-only endpoint for exploratory inference only')
    parser.add_argument('--lab-model-id', help='Explicit served model alias required with --lab-endpoint')
    parser.add_argument('--lab-model-path', help='Installed research weights to advertise; selects a base only in candidate_scaffold mode')
    parser.add_argument('--available-model-path', action='append', default=[],
                        help='Additional installed weights available for the meta-agent to choose')
    parser.add_argument('--workspace-mode', choices=('resource_pool', 'candidate_scaffold'),
                        default='resource_pool',
                        help='Default: an empty programmable resource pool; candidate_scaffold explicitly seeds an inference example')
    parser.add_argument('--device-type', choices=('cuda', 'npu'), default='cuda')
    parser.add_argument('--resource-allocation', type=json.loads,
                        help='Host JSON declaration: physical_devices, logical_devices, memory_availability, external_read_only_services')
    parser.add_argument('--execution-policy', type=json.loads,
                        help='Host-only ExecutionPolicy JSON: runtime read_only mounts and exact device nodes; no direct network')
    parser.add_argument('--inference-runtime-path', action='append', default=[],
                        help='Host-only extra read-only installed runtime/driver path for formal candidate inference; repeat as needed')
    parser.add_argument('--rsi-method', type=json.loads, default={'method': 'open_lab'},
                        help='Host catalog selection {method,config}; default optimizer supports optional reflect')
    parser.add_argument('--action-permissions', type=json.loads, default={},
                        help='Host ActionPermissions JSON; explicit sensitive rules, ordinary installed research stays open')
    parser.add_argument('--template-family', choices=('sglang', 'vllm'), default='sglang')
    parser.add_argument('--context-window-tokens', type=int, default=65536,
                        help='Actual serving context capacity, including input and output')
    parser.add_argument('--context-safety-tokens', type=int, default=1024)
    parser.add_argument('--context-budget', type=json.loads, default={},
                        help='Host JSON compression goals; input capacity always reserves the full reply cap')
    parser.add_argument('--feedback-routing', choices=('agent_memory', 'legacy', 'context_or_memory'), default='agent_memory',
                        help='Default: one meta-agent with explicit memory tools; other modes opt into legacy extraction')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--max-calls', type=int, default=128)
    parser.add_argument('--deadline-epoch', type=float, help='Absolute host UTC deadline, never extended by initialization or restart')
    parser.add_argument('--max-jobs', type=int, default=1024)
    parser.add_argument('--max-parallel', type=int, default=2,
                        help='Host program slots, including any persistent Lab-owned inference services')
    parser.add_argument('--program-timeout', type=int, default=1800)
    parser.add_argument('--evaluation-campaign-id', help='Shared evaluation account across candidates and restarts')
    parser.add_argument('--evaluation-ledger', help='Host-owned persistent campaign ledger outside research workspaces')
    parser.add_argument('--evaluation-score-protocol', default='exact-option-letter-v1',
                        choices=('exact-option-letter-v1', 'medxpertqa-official-zero-shot-ao-v1'),
                        help='Host-fixed scoring/extraction rules; unchanged across candidate submissions')
    parser.add_argument('--closed-evaluator-config', help='Operator-only isolated gateway transport config; never a closed dataset path')
    parser.add_argument('--revision', default='unreported')
    parser.add_argument('--run-id')
    parser.add_argument('--resume', action='store_true', help='Resume the same durable run using its saved configuration')
    parser.add_argument('--inspect-recovery', action='store_true', help='Read recovery status without constructing or executing modules')
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    root = Path(args.root).resolve()
    if args.inspect_recovery:
        result = inspect_recovery(root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    with controller_lock(root, resume=args.resume):
        path = root / 'recovery-launch.json'
        if args.resume:
            for action in parser._actions:
                if action.dest in ('root', 'resume', 'help'):
                    continue
                if any(word == flag or word.startswith(flag + '=')
                       for word in arguments for flag in action.option_strings):
                    parser.error('--resume uses the saved configuration; supply only --root and --resume')
            if not path.is_file():
                raise ValueError('This run has no durable launch state; legacy runs cannot be resumed in place')
            saved = json.loads(path.read_text())
            if saved['schema'] != 1 or saved['source'] != example_identity():
                raise ValueError('Resume requires the same Open Lab launcher/evaluator source')
            for name in saved['required_files']:
                if not (root / name).is_file():
                    raise ValueError('Required recovery state is missing: ' + name)
            if (root / 'result.json').exists():
                previous = json.loads((root / 'result.json').read_text())
                if previous['status'] in ('completed', 'host_limit_reached', 'deadline_reached_before_start'):
                    raise ValueError('This run is terminal; resuming cannot extend its deadline or budget')
            args = argparse.Namespace(**saved['arguments'], resume=True, inspect_recovery=False)
        else:
            for name in ('task', 'model_path', 'python', 'endpoint', 'run_id'):
                if not getattr(args, name):
                    parser.error('--' + name.replace('_', '-') + ' is required for a new run')
            if path.exists() or (root / 'application').exists():
                raise ValueError('Run directory is not fresh; use its explicit recovery path')
            if min(args.seconds, args.max_calls, args.max_jobs, args.program_timeout, args.max_parallel) < 1:
                parser.error('Host limits must be positive')
            if args.lab_endpoint and not args.lab_model_id:
                parser.error('--lab-endpoint requires an explicit --lab-model-id')
            if args.closed_evaluator_config and (not args.evaluation_campaign_id or not args.evaluation_ledger):
                parser.error('Closed evaluation requires --evaluation-campaign-id and --evaluation-ledger')
            if (args.closed_evaluator_config and Path(args.evaluation_ledger).resolve().is_relative_to(root)):
                parser.error('Closed evaluation ledger must be shared outside the individual run directory')
            args.root = str(root)
            # Python locates a venv from the invoked executable path. Resolving
            # its symlink selects the system interpreter and drops its packages.
            args.python = str(Path(args.python).absolute())
            for name in ('task', 'model_path', 'lab_model_path', 'evaluation_ledger', 'closed_evaluator_config'):
                if getattr(args, name):
                    setattr(args, name, str(Path(getattr(args, name)).resolve()))
            args.available_model_path = [str(Path(path).resolve()) for path in args.available_model_path]
            # Fix the wall deadline once, including setup and future downtime.
            args.deadline_epoch = min(args.deadline_epoch or float('inf'), time.time() + args.seconds)
            saved_args = {k: v for k, v in vars(args).items() if k not in ('resume', 'inspect_recovery')}
            write(path, {'schema': 1, 'source': example_identity(), 'arguments': saved_args,
                'required_files': ['application/config.json', 'application/run.json',
                    'application/recovery/checkpoint.json', 'application/recovery/journal.json',
                    'application/decision_memory.json', 'application/experience.sqlite3',
                    'application/feedback_learning.json', 'application/module_assembly.json',
                    'application/research_assembly.json', 'application/lab_execution.json',
                    'application/allocated-workspaces/process-recovery.json',
                    *(['application/agent-memory.json'] if args.feedback_routing == 'agent_memory' else ['memory.sqlite3']),
                    'application/system-registry.json', 'application/system-activation.json',
                    'model_calls.json', 'resource-account.json', 'resource-allocation.json']})
        write(root / 'controller.claim', {'pid': os.getpid(), 'uid': os.getuid(),
            'resume': args.resume, 'at': time.time()})
        return run(args)


def legacy_memory_assembly(root, args):
    """Explicit compatibility workflow; new runs default to single-meta memory."""
    from importlib import resources
    memory_instructions = resources.files('rsisys.loops.decision_making.feedback').joinpath(
        'prompts/langmem_research.txt').read_text()
    return ResearchAssembly.langmem(memory_path=root / 'memory.sqlite3',
        namespace=('rsisys', args.run_id), admission=(ContextOrMemoryAdmission()
            if args.feedback_routing == 'context_or_memory' else LocalLabMemoryAdmission()),
        memory_options={'max_calls_per_job': 6, 'max_steps': 1,
            'parallel_tool_calls': False,
            'additional_instructions': memory_instructions + '\nCapture concise reusable medical research and debugging observations. '
                'Make one focused tool call per reply. Keep each memory to a few sentences; '
                'do not copy code, complete prompts or unchanged environment descriptions into memory. '
                'Distinguish program outputs from validated evaluations and adoption. '
                'Absence from the supplied snapshot does not prove absence from the full history.'})


def run(args, *, method_catalog=None):
    from .open_lab_recovery import state_ports, pause_at_boundary
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    binding = BackendBinding('meta', 'chat_completions', args.endpoint, timeout_seconds=120)
    model = ModelRef('meta', args.model_id, args.model_revision)
    transport = VLLMBackend(binding, generation_options={
        'temperature': 0.6, 'chat_template_kwargs': {'enable_thinking': False}}, native_schema_mode='portable')
    backend = RecordedBackend(transport, tokenizer, root, max_calls=args.max_calls,
        template_tool_normalizer=sglang_template_tools if args.template_family == 'sglang' else None,
        provider_token_counter=args.template_family == 'vllm', resume=args.resume)
    from .open_lab_monitor import OpenLabMonitor
    monitor = None
    previous_handlers = {}
    def request_stop(signum, frame):
        # Finish the current decision boundary, then pause safely. The absolute
        # deadline and call limits still independently stop inference admission.
        backend.pause_requested = True
    compactor = host_context_compactor(vars(args))
    prompts = DecisionPrompts({
        'control': 'The host owns stopping. Select and execute your next research action. Do not use wait or finish.',
        'scheduling': 'Use lab_poll to query running programs immediately, or set wait_seconds '
            'to wait within its advertised limit. You may also inspect logs or do other research work. '
            'Use lab_result to collect terminal evidence. ',
    })
    task_path = Path(args.task).resolve()
    task = json.loads(task_path.read_text())
    # task.json is a public data contract. Never copy operator-only fields or
    # closed dataset paths from a combined input manifest into the workspace.
    task = {'train': task['train'], 'development': task['development']}
    evaluation_protocol = replace(EVALUATION_PROTOCOL,
        score_protocol=getattr(args, 'evaluation_score_protocol', EVALUATION_PROTOCOL.score_protocol))
    closed_gateway = (json.loads(Path(args.closed_evaluator_config).read_text())
                      if args.closed_evaluator_config else None)
    # Do not discover or read any sibling heldout/private paths.
    relative_files = [task['train'], *(group['path'] for group in task['development'])]
    data_manifest = []
    for name in relative_files:
        path = (task_path.parent / name).resolve()
        if not path.is_relative_to(task_path.parent) or not path.is_file():
            raise ValueError('Task file escapes fixed input directory')
        content = path.read_bytes()
        data_manifest.append({'path': name, 'sha256': hashlib.sha256(content).hexdigest(),
            'rows': len([line for line in content.splitlines() if line.strip()])})
    def persist_identity(name, value):
        path = root / name
        if args.resume:
            if not path.is_file() or json.loads(path.read_text()) != value:
                raise ValueError('Recovered deployment identity changed: ' + name)
        else:
            write(path, value)
    persist_identity('data-manifest.json', data_manifest)
    def inventory(path):
        model_path = Path(path).resolve()
        return {'model_path': str(model_path), 'config_sha256': hashlib.sha256(
            (model_path / 'config.json').read_bytes()).hexdigest(),
            'weight_shards': [{'name': p.name, 'bytes': p.stat().st_size}
                              for p in sorted(model_path.glob('*.safetensors'))],
            'binding': 'local_config_and_weight_inventory; full_weight_digest_not_verified'}
    weight_identity = inventory(args.model_path)
    actor_weight_identity = inventory(args.lab_model_path or args.model_path)
    resource_pool = args.workspace_mode == 'resource_pool' and not args.lab_endpoint
    available_models = [inventory(path) for path in dict.fromkeys(
        [args.lab_model_path or args.model_path, *args.available_model_path])]
    persist_identity('model-inventory.json', {'decision': weight_identity,
        'actor': None if resource_pool else actor_weight_identity,
        'available_models': available_models})
    device_keys = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER') if args.device_type == 'cuda' else (
        'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_AICPU_PATH')
    environment = {k: os.environ[k] for k in (*device_keys, 'LD_LIBRARY_PATH', 'CUDA_HOME', 'LIBRARY_PATH',
                                            'TORCH_DEVICE_BACKEND_AUTOLOAD')
                   if k in os.environ}
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src') + ':' +
        str(Path(__file__).resolve().parents[2]))
    supplied_policy = getattr(args, 'execution_policy', None)
    if supplied_policy is None and sys.platform == 'linux':
        raise ValueError('Linux Open Lab requires an explicit --execution-policy with assigned device nodes and runtimes')
    execution_policy = (ExecutionPolicy.from_dict(supplied_policy) if supplied_policy is not None else
        ExecutionPolicy(read_only=system_runtime_paths(args.python)))
    environment.update(dict(execution_policy.environment))
    persist_identity('backend-environment.json', environment)
    resources = allocated_resources(environment, args.device_type, args.resource_allocation)
    resources['enforced_device_nodes'] = list(execution_policy.devices)
    persist_identity('resource-allocation.json', resources)
    if any(root.is_relative_to(Path(p)) for p in execution_policy.read_only):
        raise ValueError('Runtime mounts cannot expose host run/account/recovery directories')
    model_session = HostModelSession({'meta': backend}, (model,))
    project = {'scenario': 'medical', 'python': args.python,
        'model_path': args.lab_model_path or args.model_path, 'device_type': args.device_type,
        'resources': resources,
        'task': 'data/task.json',
        'rsi_method': getattr(args, 'rsi_method', {'method': 'open_lab'}),
        'coevolution': 'Use system_status/register/activate for optional joint changes. Meta-side changes '
            'are reflect within the RSI optimizer. You build candidates; the host validates installed target ports. '
            'No candidate, reflect, training or evaluation phase is required on every decision.',
        'execution_policy': execution_policy.configuration(),
        'inference': {'owner': 'external_read_only' if args.lab_endpoint else 'meta_agent',
            'initial_service': 'external' if args.lab_endpoint else 'none',
            'program_slots': args.max_parallel,
            'deployment': 'Choose local inference or deploy services with lab_start on assigned devices. '
                'A persistent service occupies a program slot. Release devices before formal evaluation. '
                'Readiness and configuration are your responsibility. Stop owned services with lab_cancel; '
                'pause, timeout and host shutdown clean up their process groups. '
                'A stopped service is not automatically relaunched on recovery.'},
        'path_base': 'lab_workspace', 'data_root': 'data',
        'available_data': [{**entry, 'path': str(Path('data') / entry['path'])}
                           for entry in data_manifest],
        'candidate_reference': 'candidate.py',
        'feedback_routing': args.feedback_routing,
        'candidate_input_adapter': {'path': 'candidate_adapter.py',
            'sha256': hashlib.sha256(Path(__file__).with_name('candidate_adapter.py').read_bytes()).hexdigest(),
            'contract': 'medical-input-adapter-v1', 'required': False,
            'interface': 'infer(request) or infer_batch(requests); use request.render(tokenizer,...) '
                         'for local inference or request.chat_messages() in the actual HTTP request',
            'scope': 'Optional adapter for candidate inference; candidate scripts may also implement the JSON protocol directly'},
        'fixed_evaluation': {'operation': 'medical_evaluate', 'protocol': evaluation_protocol.score_protocol,
            'owner': 'medical_lab', 'evaluation_protocol': asdict(evaluation_protocol), 'scoring_editable': False,
            'base_model_paths': [entry['model_path'] for entry in available_models],
            'submission': 'key, candidate_id, split (public|closed), submission_path (workspace directory). '
                'submission.json selects entrypoint, registered base_model_path, and optional checkpoint_path OR adapter_path.',
            'bundle_schema': BUNDLE_SCHEMA,
            'submission_manifest': {'schema': BUNDLE_SCHEMA, 'entrypoint': 'infer.py',
                'base_model_path': 'one of base_model_paths',
                'weights': 'optional checkpoint_path OR adapter_path, relative to the bundle'},
            'inference_interface': {
                'execution': 'Installed Python runs entrypoint with the frozen bundle as cwd. Read submission.json for paths.',
                'stdin': '{protocol, items:[{benchmark_id,id,messages:[{role,content},...]}]}; '
                    'complete question/options are in user messages, without gold',
                'stdout': '{predictions:[{benchmark_id,id,text}]} with one raw answer per item; '
                    'send public diagnostics to stderr. The supplied system prompt may be adapted by your inference code.'},
            'closed': {'enabled': closed_gateway is not None, 'interval_seconds': 1800,
                'feedback': 'aggregate_scores_and_status_only', 'isolation_owner': 'operator_gateway'},
            'comparison': 'Match protocol_sha256, dataset_sha256 and execution budgets. '
                'This measures the whole candidate including inference; isolate training effects with a fixed inference script.',
            'execution': 'Host executes a frozen candidate inference bundle. Closed data stays at an isolated gateway. '
                'Release other programs using the assigned cards before submitting.'},
        'model_loading': 'Qwen3.5 uses transformers.AutoModelForImageTextToText, dtype=bfloat16; '
                         'use local_files_only=True and inspect installed accelerator support. '
                         'For NPU import torch_npu and use npu:0 within the assigned visible device mapping.',
        'scientific_contract': 'Train only from task.train; development is for evaluation. '
            'Paths in task.json are relative to data_root; available_data paths are relative to the Lab workspace. '
            'Do not read or search for private/heldout data, install packages, or change external services. '
            'Lab code uses only the accelerators assigned to generated programs. '
            'The host supplies the meta-agent service; deployment of Lab inference is your choice. '
            'You may write, repair and execute arbitrary research implementations within this Lab. '
            'Formal evaluation uses medical_evaluate with a frozen inference script plus checkpoint/adapter or registered base. '
            'You own inference, tokenizer, prompting and decoding. The host owns data, the selected fixed scoring/extraction protocol, '
            'resource/time limits and submission quota. Closed evaluation returns only aggregates/status and accepts '
            'one submission per 1800 seconds across the campaign; accepted failures consume the slot. '
            'This contract does not constrain training or query programs. '
            'Self-written scoring produces exploratory observations; formal scores come only from medical_evaluate receipts. '
            'Preserve candidate identity, protocol, dataset identity and exact metrics. '
            'Pareto adoption requires primary improvement without guardrail regression; '
            'do not equate a program succeeding, a proposal, or unvalidated stdout with adoption. '
            'Development accuracy is not medical generalization.'}
    project['workspace_mode'] = args.workspace_mode
    if resource_pool:
        for name in ('model_path', 'model_loading', 'candidate_reference', 'inference', 'candidate_input_adapter'):
            project.pop(name)
        project.update(available_models=available_models, initial_programs=[],
            research_execution={'owner': 'meta_agent', 'operation': 'lab_start',
                'program_slots': args.max_parallel,
                'choice': 'Choose your own experiments, programs, models and allocation across the assigned devices. '
                    'Environment/status/log queries, training, checkpoint loading, inference and evaluation '
                    'are available. No candidate or deployment step is prescribed. '
                    'Use the installed Python and inspect its libraries as needed. '
                    'A program may use multiple assigned devices; coordinate concurrent programs yourself. '
                    'Collect real terminal receipts and retain optimizer/loss/checkpoint/loading evidence '
                    'when training. An available model inventory is not a selected or loaded candidate.'})
    if args.lab_endpoint:
        project.update(endpoint=args.lab_endpoint, served_model=args.lab_model_id)
    def make_lab(context):
        import tempfile
        broker_root = Path(tempfile.gettempdir()) / ('rsisys-brokers-' + str(os.getuid()))
        broker_root.mkdir(mode=0o700, exist_ok=True)
        broker_path = broker_root / (hashlib.sha256(str(root).encode()).hexdigest()[:24] + '.sock')
        broker = ModelSessionBroker(broker_path, model_session)
        context.add_owned_cleanup(broker.close)
        project['model_session'] = {'protocol': 'generation-request-jsonl/v1', 'socket': str(broker_path),
            'models': [asdict(model)], 'account': 'shared_host_model_calls',
            'request': 'One GenerationRequest JSON object per line; response is {response:GenerationResponse} or {error,message}. '
                'All attempts consume the shared host account. Offline local inference/training is recorded separately.'}
        source = Path(__file__).resolve().parents[2] / 'src'
        policy = replace(execution_policy,
            read_only=(*execution_policy.read_only, str(source), *[entry['model_path'] for entry in available_models]),
            sockets=(*execution_policy.sockets, str(broker_path)), environment=tuple(environment.items()))
        account = ResourceAccount(root / 'resource-account.json', allocation=policy.configuration())
        allocated = RetainedIsolatedAllocatedProcessBackend(context.data_dir / 'allocated-workspaces',
            policy=policy, max_output_bytes=128000, resource_account=account)
        context.add_owned_cleanup(allocated.close_all)
        manifest = {entry['path']: entry for entry in data_manifest}
        lab = ProtocolMedicalLab(allocated, context.artifacts, timeout_seconds=args.program_timeout,
            score_protocol=evaluation_protocol.score_protocol,
            max_jobs=args.max_jobs, max_parallel=args.max_parallel, python=args.python,
            evaluation_root=context.data_dir / 'fixed-evaluations',
            base_model_paths=[entry['model_path'] for entry in available_models],
            campaign_id=args.evaluation_campaign_id or args.run_id,
            ledger_path=args.evaluation_ledger or root / 'evaluation-ledger.json', closed_gateway=closed_gateway,
            resource_policy={'device_type': args.device_type, 'device_count': resources['device_count']},
            candidate_execution=inference_policy(args.python, devices=policy.devices,
                read_only=getattr(args, 'inference_runtime_path', ()),
                environment=tuple((k, v) for k, v in environment.items() if k != 'PYTHONPATH')),
            wait_interrupted=lambda: backend.stop_requested or backend.pause_requested or (
                backend.deadline is not None and time.monotonic() >= backend.deadline),
            admission_open=lambda: not backend.stop_requested and (
                args.deadline_epoch is None or time.time() < args.deadline_epoch) and (
                backend.deadline is None or time.monotonic() < backend.deadline),
            defer_session=args.resume,
            evaluation_groups=[{'benchmark_id': group['benchmark_id'],
                'path': str(task_path.parent / group['path']), 'sha256': manifest[group['path']]['sha256']}
                for group in task['development']])
        project['fixed_evaluation']['protocol_sha256'] = lab.protocol_sha256
        native_labs.append(lab)
        reflection = None
        if optimizer_config.get('reflect_interval_seconds') is not None:
            from .open_lab_reflection import PeriodicReflection
            reflection = PeriodicReflection(context.data_dir / 'reflection-state.json',
                optimizer_config['reflect_interval_seconds'], started_at=optimizer_config.get('reflect_started_at'))
        def slot():
            return EvolutionLabSlot(RoutedLab(lab) if args.feedback_routing == 'context_or_memory' else lab,
                                    reflection=reflection)
        if args.resume:
            return slot()
        # Seed only new workspaces. Restore never overwrites edited code/data.
        (lab.workspace / 'data').mkdir()
        for name in relative_files:
            target = lab.workspace / 'data' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(task_path.parent / name, target)
        write(lab.workspace / 'data/task.json', task)
        shutil.copyfile(Path(__file__).with_name('candidate_adapter.py'), lab.workspace / 'candidate_adapter.py')
        if resource_pool:
            (lab.workspace / 'README.md').write_text(RESOURCE_POOL_README)
            write(lab.workspace / 'lab_config.json', project)
        else:
            (lab.workspace / 'candidate.py').write_text(EXTERNAL_CANDIDATE if args.lab_endpoint else CANDIDATE)
            write(lab.workspace / 'lab_config.json', {**project, 'weight_identity': actor_weight_identity})
        return slot()
    def make_decision(context, exploration):
        from rsisys.loops.decision_making.action_space.access import ActionPermissions
        context.action_permissions = ActionPermissions.from_dict(getattr(args, 'action_permissions', {}))
        context.project_context = project
        context.backends = {'meta': backend}
        loop = build_default_decision(context, exploration)
        loop.harness.context_compactor = compactor
        loop.harness.policy = TimedOpenPolicy(backend, model, allowed_actions=(),
            directives=DirectiveDefinitions(()), max_validation_attempts=3, structured_output=True,
            context_compactor=compactor, project_context=project, prompts=prompts)
        if args.feedback_routing in ('agent_memory', 'context_or_memory'):
            loop.harness.add_context_processor(ImmediateFeedbackWindow())
        bind_evolution(context, loop, context.lab_execution, backend, model, config=optimizer_config)
        return loop
    if args.feedback_routing == 'agent_memory':
        assembly = ResearchAssembly.single_meta()
    else:
        assembly = legacy_memory_assembly(root, args)
    native_labs = []
    def build_method(config):
        nonlocal optimizer_config
        import math
        if (set(config) - {'reflect', 'reflect_interval_seconds', 'reflect_started_at', 'meta_model_updates'}
                or any(key in config and type(config[key]) is not bool for key in ('reflect', 'meta_model_updates'))
                or ('reflect_interval_seconds' in config and (
                    type(config['reflect_interval_seconds']) is not int or config['reflect_interval_seconds'] < 60
                    or config.get('reflect') is False))
                or ('reflect_started_at' in config and ('reflect_interval_seconds' not in config
                    or type(config['reflect_started_at']) not in (int, float)
                    or not math.isfinite(config['reflect_started_at']) or config['reflect_started_at'] < 0))):
            raise ValueError('open_lab optimizer config requires boolean reflect/meta_model_updates and '
                             'an optional reflect_interval_seconds >=60 with reflection enabled')
        optimizer_config = deepcopy(config)
        return ModuleAssembly(lab_factory=make_lab, decision_factory=make_decision, research=assembly,
        model_session=model_session,
        state_factory=lambda context, decision, exploration: state_ports(
            context, decision, exploration, lab=native_labs[0], backend=backend),
        research_backend=backend, feedback_options={'max_calls': args.max_calls, 'max_pending': 8,
            'max_input_tokens': 24000, 'max_output_tokens': 4096})
    optimizer_config = {}
    catalog = method_catalog or MethodCatalog({'open_lab': MethodDefinition(build_method, 'open-lab-coevolution/v1')})
    modules = catalog.select(getattr(args, 'rsi_method', {'method': 'open_lab'}))
    if modules.model_session is not None and modules.model_session is not model_session:
        raise ValueError('Selected method must use this run\'s host model session')
    modules = replace(modules, model_session=model_session)
    objective = ('Improve the medical Lab using real evidence. Choose your own hypotheses and implement, '
        'execute, debug and compare them using the available tools. Read the installed task and environment '
        'information as needed. A submission is not completion: collect real terminal results with lab_result '
        'and use that feedback to choose your next action. Continue exploring; the host handles lifetime.')
    config = ApplicationConfig(objective, root / 'application', {'meta': binding}, {'decision_maker': model},
        execution='local_process', experience_limit=12, memory_limit=8, run_id=args.run_id)
    result = {'run_id': args.run_id, 'status': 'initializing', 'simulated': False,
        'weight_identity': weight_identity, 'actor_weight_identity': None if resource_pool else actor_weight_identity,
        'available_model_inventories': available_models,
        'memory_provider': assembly.memory_provider, 'feedback_routing': args.feedback_routing, 'started_at': None,
        'note': 'Model decisions and native program results are evidence; experiment/training counts require review.'}
    started = time.monotonic()
    try:
        remaining = min(args.seconds, args.deadline_epoch - time.time()) if args.deadline_epoch else args.seconds
        if not args.resume and (remaining <= 0 or backend.stop_requested):
            result.update(status='deadline_reached_before_start')
            return result
        backend.deadline = time.monotonic() + remaining
        policy = AbsoluteHostPolicy(backend, args.seconds, args.deadline_epoch)
        application = (Application.resume(config.data_dir, modules=modules, run_policy=policy) if args.resume else
                       Application(config, modules=modules, run_policy=policy,
                                   cleanup_timeout_seconds=40, durable=True))
        with application as app:
            if app.recovery.intervention is not None:
                from rsisys.runtime.recovery import RecoveryRequired
                raise RecoveryRequired('Interrupted mutation requires evidence-backed reconciliation; '
                    'inspect with --root <run> --inspect-recovery. No action was replayed')
            monitor = OpenLabMonitor(root, args.run_id, model, args.revision)
            if args.resume:
                monitor.update(app.runner.decision_making.harness, backend, compactor, None)
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, request_stop)
            started = time.monotonic()
            result['started_at'] = time.time()
            remaining = min(args.seconds, args.deadline_epoch - time.time()) if args.deadline_epoch else args.seconds
            if not args.resume:
                write(root / 'research-started.json', {'started_at': result['started_at'], 'seconds': remaining,
                    'absolute_deadline_epoch': args.deadline_epoch})
            append(root / 'controller-lifecycle.jsonl', {'event': 'resumed' if args.resume else 'started',
                'at': time.time(), 'model_calls': backend.calls, 'deadline_epoch': args.deadline_epoch})
            backend.deadline = started + max(0, remaining)
            try:
                while True:
                    if backend.stop_requested or time.monotonic() >= backend.deadline:
                        result.update(status='completed', stop='host_deadline_or_signal')
                        break
                    if backend.pause_requested:
                        pause_at_boundary(app)
                        result.update(status='paused', stop='host_pause', resumable=True)
                        break
                    try:
                        step = app.runner.step(objective)
                    except HostAdmissionStop:
                        if backend.stop_requested:
                            result.update(status='completed', stop='host_signal')
                        elif ((backend.deadline is not None and time.monotonic() >= backend.deadline)
                              or (args.deadline_epoch is not None and time.time() >= args.deadline_epoch)):
                            result.update(status='completed', stop='host_deadline')
                        else:
                            result.update(status='host_limit_reached', stop='inference_admission_limit')
                        break
                    append(root / 'steps.jsonl', {'at': time.time(), 'step': asdict(step)})
                    monitor.update(app.runner.decision_making.harness, backend, compactor, step)
                    write(root / 'context-audit.json', compactor.last_measurement)
                    if step.kind == 'finished':
                        result.update(status='completed', stop=step.event.get('run_stop'))
                        break
                    time.sleep(0.2)
            finally:
                lab = app.assembly.lab_execution
                until = time.monotonic() + 20
                while not lab.shutdown('cancel'):
                    if time.monotonic() > until:
                        raise TimeoutError('Owned Lab work did not stop')
                    time.sleep(0.1)
                # Retain generated programs/results/weights remotely; bundles may omit large/private files.
                if result['status'] != 'paused':
                    app.runner.decision_making.feedback_service.close()
                    if app.recovery.intervention is None:
                        app.recovery.mutate('shutdown', (), lambda: None)
                write(root / 'retained-workspace.json', {'path': str(lab.workspace),
                    'mode': 'same_directory_recovery', 'resumable': result['status'] == 'paused'})
                write(root / 'controller-cleanup.json', {'owned_lab_closed': True, 'at': time.time()})
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_at=time.time(), elapsed_seconds=time.monotonic() - started,
            model_calls=backend.calls, compactions=compactor.compactions)
        write(root / 'context-audit.json', compactor.last_measurement)
        if monitor is not None:
            monitor.finish(result)
        if (root / 'result.json').exists():
            append(root / 'result-history.jsonl', json.loads((root / 'result.json').read_text()))
        write(root / 'result.json', result)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
