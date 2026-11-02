"""Model-directed medical experiment (no prescribed research steps).

Run as ``python -m examples.medical.experiment --task ... --run-dir ...
--model-path ...`` inside the operator's allocated GPU job. Never resumes a run.
Optional frameworks are imported only by the injected serving/tokenizer/worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from inspect import getattr_static
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import signal
from threading import get_ident
import time
import traceback
from types import MappingProxyType
from typing import Callable

from rsisys.adapters.training import ProcessTrainingBackend
from rsisys.loops.contracts.backend import BackendCapabilities
from rsisys.loops.contracts.coordination import OperationResult, ReviewDecision
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelBackend, ModelRef
from rsisys.loops.contracts.training import ResumeMode
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall, OperationSpec
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.dsh import DSHConfig, DeepSeekDecisionHarness
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learning_policy import MemoryPrefixDecisionPolicy
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.exploration.harness.commands import CommandHandler
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.files import atomic_write
from rsisys.runtime.provenance import RunSource
from rsisys.runtime.run_policy import RunContext, RunDecision, TimeRunPolicy
from examples.qwen_toy.monitor import ACTIVE, Monitor
from .evaluation import MedicalEvaluator
from .native import MedicalTrainingBackend
from .run import build
from .task import LABELS, DataRef, Sample, encoded, prepare, samples, validate_training


class BudgetExhausted(BaseException):
    """External stop; must bypass operation/evaluator Exception recovery."""


class ResearchInputError(ValueError):
    """Known domain rejection before any external side effect."""


def write_json(path: Path, value: object) -> None:
    atomic_write(path, encoded(value))


def row_json(row: Sample) -> dict:
    return {**asdict(row), 'options': dict(row.options)}


def jsonl(rows) -> bytes:
    return b''.join(encoded(row_json(row)) + b'\n' for row in rows)


def copy_resource_allocation(value: dict | None) -> dict | None:
    """Copy a small operator declaration; never discover allocation from the host."""
    if value is None:
        return None
    if type(value) is not dict:
        raise TypeError('resource_allocation must be a JSON object or None')
    remaining = 512
    def check(node, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 8:
            raise ValueError('resource_allocation exceeds JSON depth/node bounds')
        if type(node) is dict:
            if any(type(key) is not str for key in node):
                raise TypeError('resource_allocation requires string JSON keys')
            for child in node.values():
                check(child, depth + 1)
        elif type(node) is list:
            for child in node:
                check(child, depth + 1)
        elif node is not None and type(node) not in (str, bool, int, float):
            raise TypeError('resource_allocation requires JSON values')
    check(value)
    content = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
    if len(content.encode('utf-8')) > 8192:
        raise ValueError('resource_allocation exceeds 8192 bytes')
    copied = json.loads(content)
    allowed = {'physical_devices', 'logical_devices', 'memory_availability', 'external_read_only_services'}
    if set(copied) - allowed:
        raise ValueError('Unknown resource_allocation fields')
    for name in ('physical_devices', 'logical_devices'):
        if name not in copied:
            continue
        devices = copied[name]
        if (type(devices) is not list or len(devices) > 32 or any(
                not ((type(device) is int and device >= 0)
                     or (type(device) is str and device.strip() and len(device) <= 128))
                for device in devices) or len(set(devices)) != len(devices)):
            raise ValueError(f'{name} must list distinct explicit device IDs (at most 32)')
    if ('memory_availability' in copied
            and not isinstance(copied['memory_availability'], (str, dict))
            and copied['memory_availability'] is not None):
        raise ValueError('memory_availability must be an explicit description, JSON object, or null')
    services = copied.get('external_read_only_services', [])
    if type(services) is not list or any(type(service) is not dict for service in services):
        raise ValueError('external_read_only_services must be a list of JSON service declarations')
    return copied


@dataclass(frozen=True)
class ExperimentSettings:
    task: Path
    run_dir: Path
    model_path: Path
    seconds: float = 5400
    max_trainings: int = 3
    max_model_calls: int = 96
    max_seq_length: int = 2048
    max_train_steps: int = 64
    private_exclusions: tuple[Path, ...] = ()
    decision_thinking: bool = False
    parent_run_id: str | None = None
    training_processes: int = 1
    serving_backend: str = 'transformers'
    tensor_parallel_size: int = 1
    training_device_type: str = 'cuda'
    project_id: str = 'medical-rsi'
    project_name: str = '医疗自主研究'
    experiment_name: str | None = None
    decision_harness: str = 'default'
    dsh_config: Path | None = None
    run_mode: str = 'researcher'
    restart_delay_seconds: float = 10.0
    resource_allocation: dict | None = None
    feedback_learning: bool = False

    def __post_init__(self):
        for name in ('task', 'run_dir', 'model_path'):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f'{name} must be a Path')
        if type(self.seconds) not in (int, float) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('seconds must be finite and positive')
        if self.run_mode not in ('researcher', 'time'):
            raise ValueError('run_mode must be researcher or time')
        TimeRunPolicy(self.seconds, self.restart_delay_seconds)
        for name, maximum in (('max_trainings', 3), ('max_train_steps', 64),
                              ('max_model_calls', 10000), ('max_seq_length', 2048)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in 1..{maximum}')
        if not isinstance(self.private_exclusions, tuple) or any(
                not isinstance(path, Path) for path in self.private_exclusions):
            raise TypeError('private_exclusions must be a tuple of Paths')
        if type(self.decision_thinking) is not bool:
            raise TypeError('decision_thinking must be boolean')
        for name in ('training_processes', 'tensor_parallel_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.serving_backend not in ('transformers', 'sglang'):
            raise ValueError('serving_backend must be transformers or sglang')
        if self.serving_backend == 'transformers' and self.tensor_parallel_size != 1:
            raise ValueError('Tensor parallel inference requires the sglang serving backend')
        if self.training_device_type not in ('cuda', 'npu'):
            raise ValueError('training_device_type must be cuda or npu')
        RunSource(self.project_id)
        if not isinstance(self.project_name, str) or not self.project_name.strip():
            raise ValueError('project_name must be a nonempty string')
        if self.experiment_name is not None and (not isinstance(self.experiment_name, str) or not self.experiment_name.strip()):
            raise ValueError('experiment_name must be a nonempty string when supplied')
        RunSource(self.run_dir.resolve().name, self.parent_run_id)
        if self.decision_harness not in ('default', 'dsh'):
            raise ValueError('decision_harness must be default or dsh')
        if self.dsh_config is not None and not isinstance(self.dsh_config, Path):
            raise TypeError('dsh_config must be a Path')
        if self.decision_harness == 'dsh' and self.dsh_config is None:
            raise ValueError('dsh requires an explicit dsh_config JSON file')
        if self.decision_harness == 'default' and self.dsh_config is not None:
            raise ValueError('dsh_config requires decision_harness=dsh')
        object.__setattr__(self, 'resource_allocation', copy_resource_allocation(self.resource_allocation))
        if type(self.feedback_learning) is not bool:
            raise TypeError('feedback_learning must be boolean')
        if self.feedback_learning and self.decision_harness != 'default':
            raise ValueError('feedback_learning currently requires the default decision harness')


def build_project_context(settings: ExperimentSettings, task, *, target_model: ModelRef,
                          decision_model: ModelRef) -> dict:
    """Stable project facts in user context, separate from learned interpretations."""
    declaration = copy_resource_allocation(settings.resource_allocation)
    resources = {
        'device_type': settings.training_device_type,
        'process_count': settings.training_processes,
        'allocation_status': 'unknown' if declaration is None else 'operator_declared',
        'memory_availability': 'unknown',
        'external_read_only_services': [],
        'scope': ('Device type and process count are configured execution settings, not capacity evidence. '
                  'Only explicitly declared device IDs are allocated. Host visibility and environment variables '
                  'do not grant device access; allocation does not guarantee current free memory. '
                  'External read-only services grant no training, adapter loading, restart or reconfiguration rights.'),
    }
    resources.update(declaration or {})
    return {
        'schema_version': 1,
        'project': {'project_id': settings.project_id, 'name': settings.project_name,
                    'experiment_name': settings.experiment_name or settings.run_dir.name},
        'scope': {
            'target_exploration': {
                'model': asdict(target_model),
                'goal': 'Improve medical benchmark accuracy through evidence-driven data, SFT and inference hypotheses.',
            },
            'decision_improvement': {
                'model': asdict(decision_model),
                'goal': ('Improve hypotheses, experiment design, evidence use, memory and decision quality; '
                         'keep original receipts separate from interpretations and proposed training examples.'),
                'feedback_learning_enabled': settings.feedback_learning,
            },
        },
        'capabilities': {
            'authority': ('Live context.operations, context.directive_modes, their schemas and executor permissions '
                          'are authoritative. This project description installs no executable capability.'),
            'proposal_only_unless_installed': ['harness_changes', 'system_changes', 'decision_model_training'],
            'training': 'Only installed SFT with restart from the fixed initializer; no real RL is installed.',
            'feedback_learning': ('When enabled, feedback learning produces unverified proposals and a read-only '
                                  'operation; it does not install changes or train either model.'),
        },
        'resources': resources,
        'scientific_evaluation': {
            'protocol_ref': task.protocol_ref,
            'development': [{'benchmark_id': benchmark.benchmark_id, 'role': benchmark.role,
                             'legal_labels': list(LABELS[benchmark.benchmark_id])}
                            for benchmark in task.development],
            'selection_rule': task.selection_rule,
            'guardrails': task.guardrails,
            'max_guardrail_drop': task.max_guardrail_drop,
            'max_output_tokens': task.max_output_tokens,
            'requirements': ('Use the frozen development protocol and exact legal-label grading. '
                             'Pareto application requires at least one improvement and no decreases against the '
                             'current incumbent; legacy primary_guardrail uses the fixed initial anchor. '
                             'Separate raw measurements, interpretation and acceptance. Repeated development '
                             'selection is not a fresh holdout or clinical efficacy result. Cross-runtime '
                             'comparisons cannot isolate training quality; negative trials do not exhaust a search.'),
        },
        'data_restrictions': (
            'Train only on validated training_data artifacts. Never train on development, held-out or protected '
            'examples, including operator-private exclusions. Preserve source versions, licenses and provenance; '
            'do not seek private examples through tools or external services. Exact ID/question overlap checks '
            'do not establish semantic deduplication, contamination freedom or medical correctness.'),
        'budgets': {
            'trainings': settings.max_trainings, 'decision_calls': settings.max_model_calls,
            'research_seconds': settings.seconds, 'max_train_steps': settings.max_train_steps,
            'max_seq_length': settings.max_seq_length,
            'current_remaining': 'Use live run_control; the static limits do not reset between research cycles.',
        },
    }


class JournalMonitor(Monitor):
    """Existing public schema plus an unbounded append-only event archive."""
    def __init__(self, root, base, objective, *, decision_model: ModelRef, **kwargs):
        super().__init__(root, base, objective, **kwargs)
        self.data['models']['decision_maker'] = {
            'model_id': decision_model.model_id, 'revision': decision_model.revision}
        self.write()

    def advance(self, result):
        super().advance(result)
        # Terminal feedback ends exploration even when research continues.
        if self.data['current'].get('status') not in ACTIVE:
            self.stage('researching')

    def stage(self, phase, *, status='running'):
        super().stage(phase, status=status)
        # Keep the last directive as evidence, without presenting its mode as
        # ongoing work after feedback or controller shutdown.
        if status != 'running' or self.data['current'].get('status') not in ACTIVE:
            self.data['loops']['exploration']['summary'] = (
                '等待下一条指令' if status == 'running' else f'当前阶段：{phase}')
            self.write()

    def event(self, event, **details):
        super().event(event, **details)
        with (self.root / 'events.jsonl').open('ab') as stream:
            stream.write(encoded(self.data['events'][-1]) + b'\n')


class DecisionBackend:
    def __init__(self, serving, root: Path, monitor: Monitor, limit: int, *, decision_thinking=None):
        self.serving, self.root, self.monitor, self.limit = serving, root, monitor, limit
        self.decision_thinking = (getattr(serving, 'decision_thinking', False)
                                  if decision_thinking is None else decision_thinking)
        self.calls = 0
        self._monitor_owner = get_ident()
        self._monitor_events = SimpleQueue()
        self.check_deadline: Callable[[], None] = lambda: None
        self.training_active: Callable[[], bool] = lambda: False
        self.decision_pending: Callable[[], bool] = lambda: False

    def flush_monitor_events(self):
        """Only the controller may mutate the monitor or its atomic output file."""
        if get_ident() != self._monitor_owner:
            raise RuntimeError('Monitor events must be flushed by the controller thread')
        while True:
            try:
                event, details = self._monitor_events.get_nowait()
            except Empty:
                return
            self.monitor.event(event, **details)

    def _monitor_event(self, event, **details):
        if get_ident() == self._monitor_owner:
            self.flush_monitor_events()
            self.monitor.event(event, **details)
        else:
            self._monitor_events.put((event, details))

    def count_tokens(self, request):
        counter = (getattr(self.serving, 'count_tokens', None)
                   if getattr_static(self.serving, 'count_tokens', None) is not None else None)
        if callable(counter):
            return counter(request)
        return len(json.dumps({'messages': [{'role': 'system', 'content': request.system},
                                           {'role': 'user', 'content': request.prompt}]}).encode('utf-8'))

    @property
    def token_count_mode(self):
        if getattr_static(self.serving, 'count_tokens', None) is None:
            return 'utf8_conservative_estimate'
        return (getattr(self.serving, 'token_count_mode')
                if getattr_static(self.serving, 'token_count_mode', None) is not None else 'provider_tokenizer')

    def native_notification(self, notification):
        """Reserve auxiliary native compaction attempts, never scheduler ticks.

        The SDK does not expose a synchronous transport admission hook. This is
        a conservative reservation at the native compaction/start event, not an
        exact HTTP-request count or a guarantee to interrupt an in-flight call.
        """
        event = notification.payload.get('event', {})
        if notification.method == 'session.event' and event.get('type') == 'compaction/start':
            self.check_deadline()
            if self.calls >= self.limit:
                raise BudgetExhausted('model_call_budget_exhausted')
            self.calls += 1
            write_json(self.root / 'model_calls.json', {'attempted': self.calls, 'limit': self.limit,
                'count_mode': 'native_turns_and_compaction_reservations', 'token_usage': 'unknown'})
            self.monitor.event('native_compaction_reserved', call=self.calls, sdk_event=event)
        with (self.root / 'dsh_events.jsonl').open('ab') as stream:
            stream.write(encoded({'method': notification.method, 'payload': notification.payload}) + b'\n')

    @property
    def supports_response_schema(self):
        return getattr(self.serving, 'supports_response_schema', False) is True

    def generate(self, request):
        try:
            return self._generate(request)
        except BudgetExhausted as exc:
            if get_ident() != self._monitor_owner:
                # Learning records Exception failures; controller stops still
                # bypass ordinary operation/evaluator recovery as before.
                raise RuntimeError(str(exc)) from exc
            raise

    def _generate(self, request):
        self.check_deadline()
        if self.training_active():
            raise RuntimeError('Decision inference prohibited while a training process is active')
        if self.calls >= self.limit:
            raise BudgetExhausted('model_call_budget_exhausted')
        self.calls += 1
        call_id = self.calls
        model = asdict(request.model)
        write_json(self.root / 'model_calls.json', {'attempted': self.calls, 'limit': self.limit,
                                                  'requested_model': model})
        self._monitor_event('model_decision_requested', call=self.calls, requested_model=model)
        cap = getattr(self.serving, 'native_output_tokens', 8192 if self.decision_thinking else 1024)
        request = replace(request, max_output_tokens=min(cap, request.max_output_tokens))
        with (self.root / 'model_requests.jsonl').open('ab') as stream:
            # Retain schema property order so request logs can reproduce decoding.
            record = {'call': self.calls, 'request': asdict(request)}
            encoder = getattr(self.serving, 'native_response_schema', None)
            if request.response_schema is not None and callable(encoder):
                record.update(native_schema_mode=self.serving.native_schema_mode,
                              native_response_schema=encoder(request.response_schema))
            stream.write(json.dumps(record,
                ensure_ascii=False, allow_nan=False).encode() + b'\n')
        response = self.serving.generate(request)
        with (self.root / 'model_responses.jsonl').open('ab') as stream:
            stream.write(encoded({'call': call_id, 'requested_model': model,
                                  'response': asdict(response)}) + b'\n')
        self.check_deadline()
        return response


class ExperimentRunPolicy:
    """Deadline and researcher finish both stop; neither restarts research."""
    def __init__(self, seconds: float, backend: DecisionBackend):
        self.max_seconds, self.backend = seconds, backend

    def evaluate(self, context: RunContext) -> RunDecision:
        if context.elapsed_seconds >= self.max_seconds:
            return RunDecision('stop', 'deadline', 'cancel')
        if context.researcher_finished:
            return RunDecision('stop', 'researcher_finished', 'cancel')
        if self.backend.calls >= self.backend.limit and not self.backend.decision_pending():
            return RunDecision('stop', 'model_call_budget_exhausted', 'cancel')
        return RunDecision()


class RecipeLearner:
    """Small recipe router; shared process backend, recipe-scoped native keys.

    Reservations are persisted BEFORE submission. Unknown submissions consume
    budget and are never automatically replayed by the controller.
    """
    def __init__(self, root, artifacts, converter, base, model_path, monitor, release_memory,
                 *, max_trainings=3, max_steps=64, process_backend=None, training_processes=1,
                 native_model_kwargs=None, training_device_type='cuda'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts, self.converter, self.base = artifacts, converter, base
        self.model_path, self.monitor, self.release_memory = str(model_path), monitor, release_memory
        self.max_trainings, self.max_steps = max_trainings, max_steps
        self.training_processes = training_processes
        if training_device_type not in ('cuda', 'npu'):
            raise ValueError('training_device_type must be cuda or npu')
        self.training_device_type = training_device_type
        self.native_model_kwargs = deepcopy(native_model_kwargs) if native_model_kwargs is not None else {}
        self.process = process_backend or ProcessTrainingBackend(self.root / 'process')
        self.backends: dict[str, MedicalTrainingBackend] = {}
        self.recipes: dict[str, dict] = {}
        self.ledger_path = self.root / 'budget.json'
        self.ledger: dict[str, dict] = (json.loads(self.ledger_path.read_text())
                                      if self.ledger_path.exists() else {})
        self.default_recipe = self.define_recipe({'steps': min(32, max_steps), 'lr': 1e-4, 'seed': 0})

    @property
    def remaining(self) -> int:
        return self.max_trainings - len(self.ledger)

    def capabilities(self):
        return BackendCapabilities(methods=('sft',), resume_modes=(ResumeMode.RESTART,), supports_reattach=True)

    def recipe_properties(self):
        """One parameter contract for model proposals and native recipe creation."""
        return {name: {'type': kind, 'minimum': lo, 'maximum': hi, 'default': default}
            for name, kind, lo, hi, default in (
                ('steps', 'integer', 1, self.max_steps, min(32, self.max_steps)),
                ('lr', 'number', 1e-6, 1e-3, 1e-4),
                ('seed', 'integer', 0, 2**31 - 1, 0),
                ('weight_decay', 'number', 0, .1, 0.0),
                ('warmup_ratio', 'number', 0, .2, 0.0))}

    def validate_recipe(self, fields):
        properties = self.recipe_properties()
        if not isinstance(fields, dict) or set(fields) - properties.keys():
            raise ResearchInputError('define_recipe accepts steps, lr, seed, weight_decay, warmup_ratio only')
        values = {}
        for name, field in properties.items():
            value = fields.get(name, field['default'])
            types = (int,) if field['type'] == 'integer' else (int, float)
            if (type(value) not in types or not field['minimum'] <= value <= field['maximum']
                    or not math.isfinite(value)):
                raise ResearchInputError(f"{name} must be a finite {field['type']} in {field['minimum']}..{field['maximum']}")
            values[name] = value if field['type'] == 'integer' else float(value)
        return values

    def define_recipe(self, fields: dict) -> str:
        values = self.validate_recipe(fields)
        template = {'model_name_or_path': self.model_path, 'initial_model': asdict(self.base),
            'num_processes': self.training_processes, 'device_type': self.training_device_type,
            'model_auto_class': 'image_text_to_text', 'use_cache': False,
            'model_kwargs': {'local_files_only': True, 'dtype': 'bfloat16', 'attn_implementation': 'sdpa',
                             **deepcopy(self.native_model_kwargs)},
            'seed': values['seed'],
            'lora_config': {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'bias': 'none',
                            'task_type': 'CAUSAL_LM', 'target_modules': ['q_proj', 'v_proj']},
            'training_args': {'max_steps': values['steps'], 'learning_rate': values['lr'],
                'seed': values['seed'], 'data_seed': values['seed'], 'weight_decay': values['weight_decay'],
                'warmup_ratio': values['warmup_ratio'], 'per_device_train_batch_size': 1,
                'gradient_accumulation_steps': 2, 'bf16': True, 'gradient_checkpointing': True,
                'gradient_checkpointing_kwargs': {'use_reentrant': False}, 'logging_steps': 1,
                'save_strategy': 'no', 'report_to': 'none', 'dataloader_num_workers': 0}}
        digest = sha256(encoded(template)).hexdigest()
        backend = MedicalTrainingBackend(self.root / 'recipes' / digest, self.artifacts,
            tokenize=self.converter, conversion_id=self.converter.conversion_id,
            recipe_template=template, process_backend=self.process)
        ref = backend.recipe_ref
        self.backends[ref], self.recipes[ref] = backend, {'parameters': values, 'template': template}
        write_json(self.root / 'recipes.json', self.recipes)
        return ref

    def bind_inputs(self, request, artifacts):
        if request.recipe_ref not in self.backends:
            raise ResearchInputError('Unknown recipe_ref; call define_recipe first')
        return self.backends[request.recipe_ref].bind_inputs(request, artifacts)

    def signature(self, request) -> str:
        rows = [row_json(row) for ref in request.data_refs for row in samples(self.artifacts.get(ref))]
        rows.sort(key=lambda row: row['id'])
        if len({row['id'] for row in rows}) != len(rows):
            raise ResearchInputError('Duplicate sample IDs across training inputs')
        return sha256(encoded({'rows': rows, 'base': asdict(request.base), 'method': request.method,
                               'recipe': self.recipes[request.recipe_ref]})).hexdigest()

    def check_request(self, request, key: str) -> str:
        self.bind_inputs(request, self.artifacts)
        signature = self.signature(request)
        if key in self.ledger:
            if self.ledger[key]['request'] != json.loads(encoded(asdict(request))):
                raise ResearchInputError('Idempotency key reused with a different request')
            return signature
        if self.remaining <= 0:
            raise ResearchInputError('Training budget exhausted; prompt research and evaluation remain available')
        if any(row['signature'] == signature for row in self.ledger.values()):
            raise ResearchInputError('Exact duplicate data/initializer/seed/recipe training rejected; no process started')
        return signature

    def submit(self, request, *, idempotency_key):
        signature = self.check_request(request, idempotency_key)
        backend = self.backends[request.recipe_ref]
        native_key = request.recipe_ref + ':' + idempotency_key
        native_id = sha256(native_key.encode()).hexdigest()
        if idempotency_key not in self.ledger:
            self.ledger[idempotency_key] = {'signature': signature, 'request': asdict(request),
                'recipe_ref': request.recipe_ref, 'native_key': native_key,
                'process_job': asdict(JobRef(native_id, self.process.backend)),
                'status': 'reserved', 'submission_attempted': False}
            write_json(self.ledger_path, self.ledger)
        row = self.ledger[idempotency_key]
        self.release_memory()  # Only before training, never while polling.
        row['submission_attempted'] = True
        write_json(self.ledger_path, self.ledger)
        job = backend.submit(request, idempotency_key=native_key)
        row.update(job=asdict(job), status='submitted')
        write_json(self.ledger_path, self.ledger)
        self.monitor.event('job', kind='train', job_id=job.job_id, status='submitted',
                           total_steps=self.recipes[request.recipe_ref]['parameters']['steps'])
        return job

    def _owner(self, job):
        for row in self.ledger.values():
            expected = row.get('job') or {'job_id': row['process_job']['job_id'],
                                         'backend': self.backends[row['recipe_ref']].backend}
            if expected == asdict(job):
                return self.backends[row['recipe_ref']], row
        raise ValueError('Foreign recipe training job')

    def poll(self, job):
        backend, row = self._owner(job)
        outcome = backend.poll(job)
        row['status'] = outcome.status.value
        write_json(self.ledger_path, self.ledger)
        details = asdict(outcome.progress) if outcome.progress else {}
        self.monitor.event('job', kind='train', job_id=job.job_id, status=outcome.status.value, **details)
        if outcome.status == JobStatus.UNKNOWN:
            raise RuntimeError('Training process state unknown; no restart attempted')
        return outcome

    def cancel(self, job):
        self._owner(job)[0].cancel(job)

    def result(self, job):
        return self._owner(job)[0].result(job)

    reattach = poll

    def active(self) -> bool:
        return any(row['status'] not in ('succeeded', 'failed', 'cancelled') for row in self.ledger.values())

    def cleanup(self, timeout=30.0) -> dict:
        """Cancel only process IDs reserved in THIS controller's durable ledger."""
        pending, results = {}, {}
        for key, row in self.ledger.items():
            job = JobRef(**row['process_job'])
            directory = self.process.root / job.job_id
            if not (directory / 'receipt.json').exists():
                results[key] = {'status': 'not_launched', 'confirmed': True}
                continue
            outcome = self.process.poll(job)
            if outcome.terminal:
                results[key] = {'status': outcome.status.value, 'confirmed': True}
                continue
            self.process.cancel(job)
            pending[key] = job
            self.monitor.event('job', kind='train', job_id=job.job_id, status='cancel_requested')
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            for key, job in tuple(pending.items()):
                outcome = self.process.poll(job)
                if outcome.terminal:
                    results[key] = {'status': outcome.status.value, 'confirmed': True}
                    del pending[key]
            if pending:
                time.sleep(.2)
        for key in pending:
            results[key] = {'status': 'unresolved', 'confirmed': False}
        result = {'confirmed': not pending, 'jobs': results}
        write_json(self.root / 'cleanup.json', result)
        return result


class CheckedTraining(CommandHandler):
    """Pure business rejections become feedback; external exceptions propagate."""
    def __init__(self, delegate, learner):
        self.delegate, self.learner = delegate, learner

    def prepare(self, harness, state):
        try:
            self.delegate.prepare(harness, state)
            self.learner.check_request(state.request, state.idempotency_key)
        except (ValueError, KeyError, NotImplementedError) as exc:
            state.data['business_rejection'] = str(exc)

    def advance(self, harness, state, action_space):
        if 'business_rejection' in state.data:
            return Feedback(state.directive.directive_id, ExecutionStatus.FAILED,
                'Training request rejected before submission; no budget consumed', False,
                error=state.data['business_rejection'])
        return self.delegate.advance(harness, state, action_space)


class OperationDefinition:
    def __init__(self, name, description, properties, required, validate, *, read_only=False, alternatives=()):
        schema = {'type': 'object', 'properties': properties,
            'required': required, 'additionalProperties': False}
        self.alternatives = tuple(frozenset(fields) for fields in alternatives)
        if alternatives:
            schema['oneOf'] = [{'type': 'object', 'properties': {name: properties[name] for name in fields},
                'required': list(fields), 'additionalProperties': False} for fields in alternatives]
        self.spec = OperationSpec(name, description, schema, read_only=read_only)
        self.check = validate

    def validate(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) - set(self.spec.input_schema['properties']) or any(
                name not in arguments for name in self.spec.input_schema['required']):
            raise ResearchInputError(f'Invalid fields for {self.spec.name}')
        if self.alternatives and frozenset(arguments) not in self.alternatives:
            raise ResearchInputError(f'Choose exactly one field set for {self.spec.name}: {self.alternatives}')
        self.check(arguments)
        return OperationCall(self.spec.name, arguments)


class ExperimentOperations:
    """Domain operations and bindings are outside both default harnesses."""
    PREVIEW_LIMIT = 8
    SELECTION_LIMIT = 256
    MAX_IDS = 4096
    MAX_TEXT = 65536

    def __init__(self, assembly, learner, converter, excluded):
        self.assembly, self.learner, self.converter = assembly, learner, converter
        self.excluded = (*assembly.task.check(), *excluded)
        self.refs = {ref.artifact_id: ref for ref in assembly.systems.get('initial').components.values()}
        self.train_ref = assembly.systems.get('initial').components['data']
        self.rows = samples(assembly.artifacts.get(self.train_ref))
        self.by_id = {row.id: row for row in self.rows}
        integer = lambda lo, hi: {'type': 'integer', 'minimum': lo, 'maximum': hi}
        self.definitions = (
            OperationDefinition('define_recipe', 'Define controlled SFT recipe; returns recipe_ref. Seed alone is not a mechanism.',
                learner.recipe_properties(), [], learner.validate_recipe),
            OperationDefinition('preview_training_data', 'Preview at most eight training rows; development is never selectable.',
                {'offset': {**integer(0, len(self.rows)), 'default': 0},
                 'limit': {**integer(1, self.PREVIEW_LIMIT), 'default': self.PREVIEW_LIMIT}},
                [], self._page_fields, read_only=True),
            OperationDefinition('choose_training_data', 'Create training artifact from explicit IDs or bounded offset/limit page.',
                {'ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
                         'maxItems': self.MAX_IDS, 'description': 'Use distinct IDs; duplicates are rejected.'},
                 'offset': integer(0, len(self.rows)), 'limit': integer(1, self.SELECTION_LIMIT)},
                [], self._selection_fields, alternatives=(('ids',), ('offset', 'limit'))),
            OperationDefinition('prepare_data', 'Store explicit domain JSONL after protected-overlap and token-length validation.',
                {'text': {'type': 'string', 'minLength': 1, 'maxLength': self.MAX_TEXT}}, ['text'], self._text_fields),
            OperationDefinition('read_artifact', 'Read visible evidence; training data returns summary/reference. Use paginated preview.',
                {'artifact_id': {'type': 'string', 'minLength': 1}}, ['artifact_id'], self._artifact_fields, read_only=True))
        self.bindings = {definition.spec.name: self._guard(getattr(self, definition.spec.name))
                         for definition in self.definitions}

    @staticmethod
    def _guard(handler):
        def execute(arguments):
            try:
                return handler(arguments)
            except ResearchInputError as exc:
                return OperationResult('Domain operation rejected', (), False, ExecutionStatus.FAILED, str(exc))
        return execute

    def _page_fields(self, args):
        offset, limit = args.get('offset', 0), args.get('limit', self.PREVIEW_LIMIT)
        if (type(offset) is not int or not 0 <= offset <= len(self.rows)
                or type(limit) is not int or not 1 <= limit <= self.PREVIEW_LIMIT):
            raise ResearchInputError(f'Preview requires offset in 0..{len(self.rows)} and limit in 1..{self.PREVIEW_LIMIT}')

    def _selection_fields(self, args):
        if 'ids' in args:
            ids = args['ids']
            if not isinstance(ids, list) or not 1 <= len(ids) <= self.MAX_IDS or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
                raise ResearchInputError(f'Provide unique string ids only, 1..{self.MAX_IDS}')
        elif (type(args['offset']) is not int or not 0 <= args['offset'] <= len(self.rows)
                or type(args['limit']) is not int or not 1 <= args['limit'] <= self.SELECTION_LIMIT):
            raise ResearchInputError(f'Choose offset in 0..{len(self.rows)} and limit in 1..{self.SELECTION_LIMIT}')

    def _text_fields(self, args):
        if not isinstance(args['text'], str) or not 1 <= len(args['text']) <= self.MAX_TEXT:
            raise ResearchInputError(f'text must contain 1..{self.MAX_TEXT} characters')

    @staticmethod
    def _artifact_fields(args):
        if not isinstance(args['artifact_id'], str) or not args['artifact_id']:
            raise ResearchInputError('artifact_id must be a nonempty string')

    def define_recipe(self, args):
        ref = self.learner.define_recipe(args)
        return OperationResult('Recipe defined', (), False,
            data={'recipe_ref': ref, **self.learner.recipes[ref], 'seed_is_mechanism_change': False})

    def preview_training_data(self, args):
        offset, limit = args.get('offset', 0), args.get('limit', self.PREVIEW_LIMIT)
        rows = self.rows[offset:offset+limit]
        return OperationResult('Training page', (self.train_ref,), False,
            data={'offset': offset, 'total': len(self.rows), 'rows': [row_json(row) for row in rows],
                  'next_offset': offset+len(rows) if offset+len(rows) < len(self.rows) else None})

    def _publish(self, rows):
        if not rows:
            raise ResearchInputError('Selection is empty')
        ref = self.assembly.artifacts.put(jsonl(rows), kind='training_data')
        self.refs[ref.artifact_id] = ref
        return OperationResult('Training data prepared', (ref,), False,
            data={'train_ref': asdict(ref), 'count': len(rows), 'benchmarks': dict(Counter(r.benchmark_id for r in rows))})

    def choose_training_data(self, args):
        if 'ids' in args:
            if any(ident not in self.by_id for ident in args['ids']):
                raise ResearchInputError('IDs must belong to the frozen effective training partition')
            rows = tuple(self.by_id[ident] for ident in args['ids'])
        else:
            rows = self.rows[args['offset']:args['offset']+args['limit']]
        return self._publish(rows)

    def prepare_data(self, args):
        try:
            rows = samples(args['text'].encode())
            validate_training(rows, self.excluded)
            kept, excluded = self.converter.filter_samples(rows)
            if excluded:
                raise ValueError('Some rows exceed max_seq_length; remove whole rows before retrying')
        except (ValueError, TypeError, KeyError) as exc:
            raise ResearchInputError(str(exc)) from exc
        return self._publish(kept)

    def read_artifact(self, args):
        visible = dict(self.refs)
        visible.update(self.assembly.evaluator.evidence)
        for candidate in self.assembly.candidates.values():
            visible.update({ref.artifact_id: ref for ref in candidate.changes.values()})
        ref = visible.get(args['artifact_id'])
        if ref is None:
            raise ResearchInputError('Unknown or nonvisible artifact')
        if not ref.uri.startswith(('artifact://', 'memory://')):
            return OperationResult('External component reference; an explicit resolver is required', (ref,), False,
                data={'reference': asdict(ref), 'requires_external_resolver': True, 'content': None})
        if ref.kind == 'training_data':
            rows = samples(self.assembly.artifacts.get(ref))
            return OperationResult('Training artifact summary; use preview_training_data for source rows', (ref,), False,
                data={'train_ref': asdict(ref), 'count': len(rows),
                      'benchmarks': dict(Counter(r.benchmark_id for r in rows))})
        content = self.assembly.artifacts.get(ref).decode()
        return OperationResult('Visible evidence excerpt' if len(content) > 12000 else 'Visible artifact', (ref,), False,
            data={'content': content[:12000], 'total_characters': len(content), 'truncated': len(content) > 12000})


class ResearchPolicy(MemoryPrefixDecisionPolicy):
    ENVELOPE_GUIDANCE = (
        'Return one JSON object with unique keys in every object, including nested objects. '
        'Canonical outer envelopes are '
        '{"kind":"invoke","operation":"<name from context.operations>","arguments":{}} or '
        '{"kind":"explore","mode":"<mode from context.directive_modes>",'
        '"strategy":"<your strategy>","input_ids":[],"arguments":{}}. '
        'Fill arguments according to the chosen operation or directive input_schema. '
        'Put operation parameters (including view, kind, limit) inside arguments, '
        'never in the top-level kind; top-level kind selects the decision envelope. '
        'train requires nonempty top-level input_ids selecting available training_data artifacts. '
        'Omit arguments.origin: initialization is host-owned and origin is not an accepted model argument. '
        'The executor binds the current experimental parent and fixed initial model automatically, '
        'using restart without a checkpoint. Training data belongs only in input_ids. '
        'Choose your own available action and research steps, or finish with a nonempty reason; '
        'these envelopes are syntax templates, not a required workflow.')

    def __init__(self, backend, model, catalog, space, monitor, learner, train_ref, *, run_mode='researcher',
                 project_context: dict | None = None):
        super().__init__(backend, model, directives=catalog, max_validation_attempts=1,
                         project_context=project_context)
        self.space, self.monitor, self.learner = space, monitor, learner
        self.train_ref = train_ref
        self.run_mode = run_mode
        self.finish_reason: str | None = None

    def _decision_request(self, context):
        request = super()._decision_request(context)
        cap = getattr(self.backend.serving, 'native_output_tokens',
                      8192 if self.backend.decision_thinking else 1024)
        return replace(request, max_output_tokens=cap)

    def decision_schema(self, context):
        schema = super().decision_schema(context)
        # Match this example's existing serial execution rules, without choosing an action.
        schema['oneOf'] = [branch for branch in schema['oneOf']
                           if branch['properties']['kind']['const'] != 'wait']
        for branch in schema['oneOf']:
            fields = branch['properties']
            if fields['kind']['const'] == 'finish':
                fields['reason']['minLength'] = 1
                branch['required'].append('reason')
            if fields.get('mode', {}).get('const') == 'train':
                fields.pop('review_after_seconds', None)
                fields['input_ids']['minItems'] = 1
                branch['required'].append('input_ids')
        return schema

    def _parse_decision(self, response, context):
        """Resolve envelope spelling only when an allowed action has one owner.

        This is a provider-facing parser adaptation, not another executor. The
        unchanged action-space validation/permission checks still run afterward.
        """
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f'Duplicate JSON key {key!r}; keys must be unique in every object')
                result[key] = value
            return result

        payload = json.loads(response.text, object_pairs_hook=unique_object)
        operations = {spec.name for spec in context.operations}
        directives = set(context.directive_modes)
        kind = payload.get('kind') if isinstance(payload, dict) else None
        if not isinstance(payload, dict):
            raise ValueError('Decision must be a JSON object')
        names = [payload[key] for key in ('action', 'operation', 'mode') if key in payload]
        if kind not in (None, 'invoke', 'explore', 'finish', 'wait'):
            names.append(kind)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError('Action names must be nonempty strings')
        if len(set(names)) > 1 or (names and kind in ('finish', 'wait')):
            raise ValueError('Conflicting action names; request exactly one action')
        name = names[0] if names else None
        normalized = None
        if name in operations ^ directives:
            if set(payload) - {'kind', 'action', 'mode', 'operation', 'strategy', 'arguments', 'input_ids',
                               'review_after_seconds', 'reason', 'depends_on_decision_ids', 'source_decision_ids',
                               'research_attempt_id', 'research_attempt_title'}:
                raise ValueError('Unexpected fields in action envelope')
            if name in operations:
                if payload.get('input_ids', []) != [] or payload.get('review_after_seconds') is not None:
                    raise ValueError('Operation envelope cannot consume directive inputs or scheduling fields')
                normalized = {'kind': 'invoke', 'operation': name, 'arguments': payload.get('arguments', {})}
            else:
                normalized = {'kind': 'explore', 'mode': name, 'arguments': payload.get('arguments', {}),
                              'strategy': payload.get('strategy', 'Model requested '+name)}
                normalized.update({key: payload[key] for key in ('input_ids', 'review_after_seconds') if key in payload})
            if 'reason' in payload:
                normalized['reason'] = payload['reason']
            normalized.update({key: payload[key] for key in ('depends_on_decision_ids', 'source_decision_ids',
                                                            'research_attempt_id', 'research_attempt_title')
                               if key in payload})
        if normalized is not None and normalized != payload:
            self.monitor.event('decision_envelope_normalized', original=payload, normalized=normalized)
            response = replace(response, text=json.dumps(normalized))
        # This immutable input is explicitly published in run_control. The core
        # serial harness clears operation_results after directive feedback; make
        # the fixed initial reference resolvable for an explicit later train.
        # This affects parsing only and never displaces the latest model receipt.
        parsed = normalized if normalized is not None else payload
        if (parsed.get('kind') == 'explore' and parsed.get('mode') == 'train'
                and self.train_ref.artifact_id in parsed.get('input_ids', ())
                and not any(self.train_ref in row.artifacts for row in context.operation_results)):
            context = replace(context, operation_results=(OperationResult(
                'Fixed initial training reference', (self.train_ref,), False), *context.operation_results))
        return super()._parse_decision(response, context)

    def decide(self, context):
        hint = '\n' + self.ENVELOPE_GUIDANCE
        for attempt in range(3):
            # Transport/model identity/external failures are deliberately outside retry.
            current = replace(context, objective=context.objective + hint,
                run_control={**context.run_control, 'mode': self.run_mode,
                    'researcher_finish_ends_run': self.run_mode == 'researcher',
                    'training_budget': {'limit': self.learner.max_trainings,
                        'reserved': len(self.learner.ledger), 'remaining': self.learner.remaining},
                    'model_call_budget': {'limit': self.backend.limit,
                        'used': self.backend.calls, 'remaining': self.backend.limit - self.backend.calls},
                    'initial_training_data': asdict(self.train_ref)})
            response = self._generate_decision(current)
            try:
                decision = self._parse_decision(response, current)
                space = authorized_space(self.space)
                if decision.kind == 'invoke':
                    space.validate(decision.operation)
                elif decision.kind == 'explore':
                    space.validate_directive(decision.directive)
                    if decision.directive.mode == 'train':
                        if decision.directive.review_after_seconds is not None:
                            raise ValueError('Serial training owns the allocated GPUs; omit review_after_seconds')
                        if not decision.directive.input_refs or any(ref.kind != 'training_data' for ref in decision.directive.input_refs):
                            raise ValueError('train requires explicit input_ids containing training_data artifacts only')
                elif decision.kind == 'wait':
                    raise ValueError('Serial research has no outstanding job here; choose an operation, directive, or finish')
                elif not decision.reason.strip():
                    raise ValueError('finish requires a nonempty reason')
            except (ValueError, TypeError, KeyError, PermissionError) as exc:
                self.monitor.event('model_decision_rejected', attempt=attempt+1, error=str(exc)[:1000])
                hint = '\nCorrect this invalid proposal before execution (error data): ' + json.dumps({
                    'previous_output': response.text[:2048], 'validation_error': str(exc)[:1000],
                    'envelope_guidance': self.ENVELOPE_GUIDANCE,
                    'available_candidate_ids': [candidate.candidate_id for candidate in current.candidates]})
                continue
            if decision.kind == 'finish':
                self.finish_reason = decision.reason
            if decision.directive:
                self.monitor.stage(decision.directive.mode)
            self.monitor.decision(decision, current, len(self.learner.ledger)+1)
            self.monitor.event('model_decision_validated', kind=decision.kind)
            return decision
        raise ValueError('Researcher failed three bounded schema/validation attempts')

    def review(self, context):
        # Resource scheduling, not a synthetic researcher/model decision.
        return ReviewDecision('continue', reason='Serial controller polls without inference while training owns GPUs')


def evaluator_factory(monitor, root):
    class ObservedEvaluator(MedicalEvaluator):
        def _measure(self, system):
            before = len(self._partial.get(system.version, ()))
            result = super()._measure(system)
            predictions = self._partial.get(system.version, ())
            if len(predictions) > before:
                row = predictions[-1]
                monitor.event('job', kind='evaluate', job_id='evaluation:'+system.version,
                    status='succeeded' if result else 'running', step=len(predictions),
                    total_steps=sum(len(b.data.read()) for b in self.task.development),
                    candidate_id=system.version, metrics={'correct': int(row['correct'])})
                write_json(root / 'evaluation_progress.json', self._partial)
            write_json(root / 'evaluations.json', {key: asdict(value) for key, value in self._outputs.items()})
            return result
    return ObservedEvaluator


def model_identity(path: Path) -> tuple[str, dict]:
    """Content pin, with HF snapshot commit used only when actually present."""
    from .transformers_runtime import _files_identity
    root = path.resolve()
    actual = _files_identity(root)
    revision = root.name if root.parent.name == 'snapshots' and len(root.name) == 40 else 'local-sha256:'+actual
    config = json.loads((root / 'config.json').read_text())
    model_id = config.get('_name_or_path') or str(root)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError('Model config _name_or_path must be a nonempty string when supplied')
    manifest = {'model_id': model_id, 'revision': revision, 'source_directory': str(root),
        'actual_weight_sha256': actual, 'files': [{'name': p.name, 'bytes': p.stat().st_size}
            for p in sorted(root.iterdir()) if p.is_file() and (p.suffix in ('.json', '.safetensors', '.bin'))]}
    return revision, manifest


def persist(assembly, root):
    assembly.runner.decision_making.harness.memory.backup(root / 'decision_memory.json')
    assembly.runner.exploration.harness.memory.backup(root / 'exploration_memory.json')
    write_json(root / 'experience.json', [asdict(record) for record in assembly.experience_store.records()])
    write_json(root / 'candidates.json', {key: asdict(value) for key, value in assembly.candidates.items()})
    write_json(root / 'evaluations.json', {key: asdict(value) for key, value in assembly.evaluator._outputs.items()})
    write_json(root / 'systems.json', {'current': asdict(assembly.systems.current)})
    backend = getattr(assembly.runner.decision_making.harness.policy, 'backend', None)
    if getattr(backend, 'last_compaction', None) is not None:
        write_json(root / 'dsh_context_budget.json', backend.last_compaction)


def run(settings: ExperimentSettings, *, serving_factory=None, converter_factory=None,
        process_backend=None, identity_factory=model_identity, research_started_callback=None,
        decision_model: ModelRef | None = None, decision_backend: ModelBackend | None = None,
        native_model_kwargs: dict | None = None, decision_harness_factory=None,
        dsh_runtime_factory=None) -> dict:
    """Run with an optional independent researcher backend and model identity.

    Without decision_backend, serving.generate owns researcher routing. With an
    injected backend, local serving only needs the actor's base model. Native
    model kwargs override recipe loading defaults for every researcher recipe.
    """
    if decision_model is not None and not isinstance(decision_model, ModelRef):
        raise TypeError('decision_model must be a ModelRef or None')
    if decision_backend is not None and not callable(getattr(decision_backend, 'generate', None)):
        raise TypeError('decision_backend must provide generate(request)')
    if settings.decision_harness == 'dsh' and decision_backend is not None:
        raise ValueError('DSH owns its session/provider; configure dsh_config instead of decision_backend')
    if settings.feedback_learning and decision_harness_factory is not None:
        raise ValueError('feedback_learning does not yet support custom decision harnesses')
    if settings.feedback_learning and decision_backend is None:
        raise ValueError('feedback_learning requires an explicit independent decision_backend')
    dsh_config = DSHConfig.load(settings.dsh_config) if settings.decision_harness == 'dsh' else None
    if dsh_config is not None and dsh_config.resume:
        raise ValueError('Medical experiment never resumes; use a fresh DSH session')
    root = settings.run_dir.resolve()
    source = RunSource(root.name, settings.parent_run_id)
    root.mkdir(parents=True, exist_ok=True)
    # This marker intentionally survives success, exceptions, SIGTERM and crashes.
    with (root / 'controller-started.json').open('x') as stream:
        stream.write(encoded({'schema_version': 1, 'started_at': time.time()}).decode())
    write_json(root / 'run.json', {**source.to_dict(), 'objective': 'Autonomous medical research'})
    serving = learner = assembly = monitor = backend = None
    result = {'status': 'failed', 'reason': 'initialization_failed', **source.to_dict(),
              'researcher_finish_reason': None, 'baseline_seconds': None, 'research_seconds': 0.0,
              'run_mode': settings.run_mode, 'research_rounds': 1}
    research_started = None
    started = time.monotonic()
    try:
        from .transformers_runtime import LocalTransformerServing, tokenize_factory
        original = prepare(json.loads(settings.task.read_text()), base_directory=settings.task.resolve().parent)
        private = tuple(row for path in settings.private_exclusions for row in samples(path.read_bytes()))
        validate_training(original.train.read(), (*original.check(), *private))
        revision, identity = identity_factory(settings.model_path)
        write_json(root / 'model_identity.json', identity)
        base = ModelRef('medical-local', identity['model_id'], revision)
        researcher = ((ModelRef('dsh', dsh_config.model, 'provider-managed') if dsh_config else base)
                      if decision_model is None else decision_model)
        write_json(root / 'decision_model.json', asdict(researcher))
        converter = (converter_factory or tokenize_factory)(str(settings.model_path.resolve()),
            max_length=settings.max_seq_length, padding=False, revision=revision if len(revision) == 40 else None)
        kept, excluded = converter.filter_samples(original.train.read())
        if not kept:
            raise ValueError('No training rows survive tokenizer preflight')
        atomic_write(root / 'effective-train.jsonl', jsonl(kept))
        task = replace(original, train=DataRef.freeze(root / 'effective-train.jsonl'))
        write_json(root / 'effective-task.json', asdict(task))
        write_json(root / 'data-manifest.json', {'original_task': asdict(original), 'effective_task': asdict(task),
            'original_train_rows': len(original.train.read()), 'effective_train_rows': len(kept),
            'excluded': excluded, 'conversion_id': converter.conversion_id, 'conversion_metadata': converter.metadata})
        artifacts = FileArtifactStore(root / 'artifacts')
        monitor = JournalMonitor(root, base, 'Autonomous medical research', rounds=settings.max_trainings,
                                 source=source, decision_model=researcher)
        model_dir = settings.model_path.resolve()
        model_label = (model_dir.parent.parent.name.removeprefix('models--').replace('--', '-')
                       if model_dir.parent.name == 'snapshots' else model_dir.name)
        monitor.data['experiment_name'] = settings.experiment_name or root.name
        monitor.data['project'] = {'project_id': settings.project_id, 'name': settings.project_name,
                                  'system_id': 'medical-' + model_label.lower()}
        write_json(root / 'parallelism.json', {'serving_backend': settings.serving_backend,
            'tensor_parallel_size': settings.tensor_parallel_size, 'training_processes': settings.training_processes,
            'training_device_type': settings.training_device_type})
        monitor.write()
        serving_options = {}
        if settings.serving_backend == 'sglang':
            from .sglang_runtime import SGLangServing
            default_serving = SGLangServing
            serving_options = {'tensor_parallel_size': settings.tensor_parallel_size, 'runtime_root': root / 'serving'}
        else:
            default_serving = LocalTransformerServing
        serving = (serving_factory or default_serving)(str(settings.model_path.resolve()), actor_model=base,
            decision_model=researcher if decision_backend is None and dsh_config is None else base,
            artifacts=artifacts, native_model_class='image_text_to_text',
            device='cuda:0', dtype='bfloat16', max_context_tokens=32768,
            decision_thinking=settings.decision_thinking, **serving_options)
        learner = RecipeLearner(root / 'training', artifacts, converter, base, settings.model_path.resolve(), monitor,
            serving.release_memory, max_trainings=settings.max_trainings, max_steps=settings.max_train_steps,
            process_backend=process_backend, training_processes=settings.training_processes,
            native_model_kwargs=native_model_kwargs, training_device_type=settings.training_device_type)
        backend = DecisionBackend(serving if decision_backend is None else decision_backend,
            root, monitor, settings.max_model_calls, decision_thinking=settings.decision_thinking)
        backend.training_active = learner.active
        def check_deadline():
            if research_started is not None and time.monotonic()-research_started >= settings.seconds:
                raise BudgetExhausted('deadline')
        backend.check_deadline = check_deadline
        write_json(root / 'decision_harness.json', {'kind': settings.decision_harness,
            'dsh': ({**asdict(dsh_config), 'home': str(dsh_config.home), 'workspace': str(dsh_config.workspace)}
                    if dsh_config else None),
            'model_call_count_mode': ('native_turns_and_compaction_reservations' if dsh_config
                                      else 'backend_generate_attempts'),
            'native_token_count_mode': 'sdk_reported_or_unknown' if dsh_config else None})
        def permitted(request):
            if request.kind != 'directive':
                return True
            if request.name == 'train':
                return learner.remaining > 0
            if request.name in ('evaluate', 'apply_candidate'):
                return assembly is not None and bool(assembly.candidates)
            return True
        permissions = ActionPermissions(check=permitted)
        assembly = build(task, artifacts=artifacts, decision_backend=backend, decision_model=researcher, initial_model=base,
            inference=serving, system_activator=serving, prepare_inference=serving.prepare_inference,
            learner=learner, bind_training_inputs=learner.bind_inputs, trusted_exclusions=private,
            run_policy=(TimeRunPolicy(settings.seconds, settings.restart_delay_seconds)
                        if settings.run_mode == 'time' else ExperimentRunPolicy(settings.seconds, backend)),
            action_permissions=permissions,
            evaluator_factory=evaluator_factory(monitor, root))
        decision = assembly.runner.decision_making.harness
        operations = ExperimentOperations(assembly, learner, converter, private)
        research_tools = decision.tools.research_tools
        tools = BoundDecisionTools({**research_tools.bindings, **operations.bindings})
        tools.research_tools, tools.artifacts = research_tools, artifacts
        decision.tools = tools
        space = ModularDecisionActionSpace((*research_tools.definitions, *operations.definitions),
            directives=assembly.runner.decision_making.action_space.directives, permissions=permissions)
        assembly.runner.decision_making.action_space = space
        project_context = build_project_context(settings, task, target_model=base, decision_model=researcher)
        write_json(root / 'project-context.json', project_context)
        def policy_factory(native_backend, model):
            if dsh_config is not None:
                backend.serving = native_backend
            return ResearchPolicy(backend, model, space.directives, space, monitor, learner, operations.train_ref,
                                  run_mode=settings.run_mode, project_context=project_context)
        dependencies = dict(tools=tools, experience=decision.experience, memory=decision.memory,
            research_agenda=decision.research_agenda, directives=space.directives, model=researcher,
            policy_factory=policy_factory, backend=backend, config=dsh_config,
            runtime_factory=dsh_runtime_factory, on_notification=backend.native_notification if dsh_config else None)
        if decision_harness_factory is not None:
            replacement = decision_harness_factory(**dependencies)
        elif dsh_config is not None:
            replacement = DeepSeekDecisionHarness(dsh_config, tools, decision.experience,
                directives=space.directives, model=researcher, policy_factory=policy_factory,
                memory=decision.memory, research_agenda=decision.research_agenda,
                runtime_factory=dsh_runtime_factory, on_notification=backend.native_notification)
        else:
            replacement = DefaultDecisionHarness(policy_factory(backend, researcher), tools, decision.experience,
                memory=decision.memory, research_agenda=decision.research_agenda)
        replacement.state.operation_results.extend(decision.state.operation_results)
        assembly.runner.decision_making.harness = decision = replacement
        if settings.feedback_learning:
            from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning
            enable_feedback_learning(assembly.runner.decision_making, path=root / 'feedback_learning.json')
            decision = assembly.runner.decision_making.harness
        policy = decision.policy
        backend.decision_pending = lambda: getattr(policy, 'future', None) is not None
        commands = assembly.runner.exploration.harness.commands
        commands.handlers = MappingProxyType({**commands.handlers,
            'train': CheckedTraining(commands.handlers['train'], learner)})
        monitor.stage('baseline')
        baseline_started = time.monotonic()
        anchor = assembly.evaluator.establish_anchor()
        result['baseline_seconds'] = time.monotonic()-baseline_started
        write_json(root / 'baseline.json', {'result': asdict(anchor), 'seconds': result['baseline_seconds']})
        persist(assembly, root)
        objective = ('Autonomously research and improve this model to increase both '
            'MedXpertQA and MedQA accuracy within the available resource budget. '
            'MedXpertQA has ten answer labels A-J; MedQA has four A-D. Grading requires exactly '
            'one legal label, with no period, prefix or explanation. Prompt changes apply to both benchmarks. '
            'The initial baseline has ALREADY been measured; its results and evidence IDs are below. '
            'Read its evidence with read_artifact if useful. The initial system is not a candidate. '
            'evaluate accepts only a candidate_id listed in context.candidates; a sole candidate and the '
            'frozen evaluation protocol default automatically, but multiple candidates require an explicit ID. '
            'Only evaluate and apply_candidate require an existing candidate. '
            'Creating a candidate does NOT require an existing candidate: configure_prompt proposes a new prompt candidate '
            'and train produces a new SFT candidate from the fixed initializer and selected training data. '
            'When context.candidates is empty, you can still use either creation directive, inspect or prepare data, '
            'define recipes, and read or update research notes, according to the available capabilities. '
            'An empty candidate list does not mean that research is blocked. '
            'Use returned candidate IDs after creation; do not invent IDs or repeat baseline measurement. '
            'You choose the research steps and may conduct multiple rounds, change training data/recipe or prompt, '
            'evaluate, apply an accepted candidate, or finish with your reason at any time. '
            'Choose data construction, SFT recipes, inference prompts or combinations according to your hypotheses. '
            'No particular method or training submission is required. Compare candidates using frozen development evidence. '
            'A few negative results do not exhaust the search; seed changes alone are not mechanism improvements. '
            f'Application follows frozen selection_rule={task.selection_rule}. '
            'The default pareto rule requires at least one accuracy improvement over the current incumbent and no decreases. '
            'Tradeoff candidates remain available for comparison. '
            'Only method=sft and restart from the fixed initializer are installed; no real RL. '
            'Use define_recipe for controlled steps/lr/seed/weight_decay/warmup_ratio, '
            'preview_training_data (at most 8 rows), choose_training_data for a subset, '
            'or prepare_data for validated domain JSONL. read_artifact returns summaries for training data. '
            'Never use development or protected examples for training. '
            'For train explicitly select input_ids of training_data only, apply_result=false, '
            'and omit review_after_seconds: training owns the allocated GPUs until the processes exit. '
            'configure_prompt/evaluate/apply_candidate require input_ids=[]. '
            f'Train budget={settings.max_trainings}, decision calls={settings.max_model_calls}, '
            f'research seconds={settings.seconds}; exhausting train budget removes train capability only. '
            f'Default recipe_ref={learner.default_recipe}; default steps={min(32, settings.max_train_steps)}, lr=0.0001. '
            f'Frozen train artifact_id={operations.train_ref.artifact_id}. '
            f'Evaluation protocol_ref={task.protocol_ref}. '
            'Fixed initial anchor: '+json.dumps(asdict(anchor)))
        if settings.run_mode == 'time':
            objective += (' Fixed-time research: finish ends only the current research cycle, not this run. '
                'While time remains, the host starts another cycle with the same history, candidates, '
                'incumbent and cumulative budgets. Use run_control for current remaining time and budgets. '
                'You still choose the hypotheses and actions; no particular experiment is required.')
        monitor.stage('researching')
        research_started = time.monotonic()
        write_json(root / 'research-started.json', {'started_at': time.time(),
            'max_seconds': settings.seconds, 'run_mode': settings.run_mode,
            'researcher_finish_ends_run': settings.run_mode == 'researcher'})
        if research_started_callback is not None:
            research_started_callback(settings.seconds)
        while True:
            step = assembly.runner.step(objective)
            monitor.sync_research_attempts(decision.state.decision_records)
            backend.flush_monitor_events()
            monitor.advance(step)
            if step.event.get('run_control') == 'research_restarted':
                result['research_rounds'] = step.event['research_round']
                monitor.event('research_restarted', research_round=step.event['research_round'])
                monitor.stage('researching')
            for key, candidate in assembly.candidates.items():
                monitor.candidates[key] = {'candidate_id': key, 'parent_system': candidate.parent_system,
                    'status': 'applied' if assembly.systems.current.version == key else 'proposed',
                    'metrics': dict(assembly.evaluator.measurement(key).metrics) if assembly.evaluator.measurement(key) else {}}
            persist(assembly, root)
            with (root / 'steps.jsonl').open('ab') as stream:
                stream.write(encoded(asdict(step))+b'\n')
            result['researcher_finish_reason'] = policy.finish_reason
            if step.kind == 'finished':
                stop = step.event.get('run_stop', {})
                result.update(status='completed' if stop.get('reason') == 'researcher_finished' else 'stopped',
                              reason=stop.get('reason', 'runner_finished'), run_stop=stop)
                break
            if step.kind == 'waiting':
                remaining = max(0.0, settings.seconds - (time.monotonic() - research_started))
                time.sleep(min(.5, step.event.get('retry_after_seconds', .5), remaining))
        result['model_calls'] = backend.calls
    except BudgetExhausted as exc:
        result.update(status='stopped', reason=str(exc))
    except KeyboardInterrupt:
        result.update(status='stopped', reason='interrupted')
    except Exception as exc:
        result.update(status='failed', reason=type(exc).__name__, error=str(exc))
        atomic_write(root / 'exception.txt', traceback.format_exc().encode())
    finally:
        # No guessed PIDs, process-wide kill, unrelated jobs or automatic retries.
        cleanup_errors = []
        try:
            if assembly is not None:
                close = getattr(assembly.runner.decision_making.harness, 'close', None)
                if callable(close):
                    close()
        except Exception as exc:
            cleanup_errors.append(f'decision harness close: {type(exc).__name__}: {exc}')
        try:
            if backend is not None:
                backend.flush_monitor_events()
        except Exception as exc:
            cleanup_errors.append(f'monitor event flush: {type(exc).__name__}: {exc}')
        try:
            if learner is not None:
                result['cleanup'] = learner.cleanup()
                if not result['cleanup']['confirmed']:
                    result.update(status='failed', reason='training_cleanup_unconfirmed')
        except Exception as exc:
            cleanup_errors.append(f'training cleanup: {type(exc).__name__}: {exc}')
        finally:
            try:
                if serving is not None:
                    serving.release_memory()
            except Exception as exc:
                cleanup_errors.append(f'serving release: {type(exc).__name__}: {exc}')
        try:
            if assembly is not None:
                persist(assembly, root)
                result['researcher_finish_reason'] = assembly.runner.decision_making.harness.policy.finish_reason
            if serving is not None:
                write_json(root / 'serving-weight-identities.json', [
                    {'model': asdict(model), 'metadata': metadata} for model, metadata in serving.weight_metadata.items()])
        except Exception as exc:
            cleanup_errors.append(f'final persistence: {type(exc).__name__}: {exc}')
        if cleanup_errors:
            result.update(status='failed', reason='cleanup_or_persistence_failed', cleanup_errors=cleanup_errors)
        result['research_seconds'] = time.monotonic()-research_started if research_started is not None else 0.0
        result['duration_seconds'] = time.monotonic()-started
        result['trainings_reserved'] = len(learner.ledger) if learner is not None else 0
        result['model_calls'] = backend.calls if backend is not None else 0
        if monitor is not None:
            monitor.data['result'] = result
            monitor.stage(result['reason'], status=result['status'])
        write_json(root / 'result.json', result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--task', type=Path, required=True)
    result.add_argument('--run-dir', type=Path, required=True)
    result.add_argument('--project-id', default='medical-rsi')
    result.add_argument('--project-name', default='医疗自主研究')
    result.add_argument('--experiment-name')
    result.add_argument('--parent-run-id', help='Declare ancestry only; does not restore model or memory state')
    result.add_argument('--model-path', type=Path, required=True)
    result.add_argument('--seconds', type=float, default=5400)
    result.add_argument('--run-mode', choices=('researcher', 'time'), default='researcher')
    result.add_argument('--restart-delay-seconds', type=float, default=10.0)
    result.add_argument('--max-trainings', type=int, default=3)
    result.add_argument('--max-model-calls', type=int, default=96)
    result.add_argument('--max-seq-length', type=int, default=2048)
    result.add_argument('--max-train-steps', type=int, default=64)
    result.add_argument('--private-exclusions', type=Path, action='append', default=[])
    result.add_argument('--decision-thinking', action='store_true',
                        help='Enable researcher thinking and sampling; actor evaluation stays greedy/non-thinking')
    result.add_argument('--decision-harness', choices=('default', 'dsh'), default='default')
    result.add_argument('--dsh-config', type=Path, help='Explicit DSH JSON session/provider/compaction configuration')
    result.add_argument('--training-processes', type=int, default=1)
    result.add_argument('--training-device-type', choices=('cuda', 'npu'), default='cuda')
    result.add_argument('--serving-backend', choices=('transformers', 'sglang'), default='transformers')
    result.add_argument('--tensor-parallel-size', type=int, default=1)
    result.add_argument('--resource-allocation', type=json.loads,
                        help='Explicit bounded JSON allocation declaration; omitted means unknown')
    result.add_argument('--feedback-learning', action='store_true',
                        help='Opt in to proposal-only feedback learning with the default decision harness')
    return result


def main(argv=None):
    args = vars(parser().parse_args(argv))
    args['private_exclusions'] = tuple(args['private_exclusions'])
    settings = ExperimentSettings(**args)
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    previous = signal.signal(signal.SIGTERM, terminate)
    def expired(signum, frame):
        raise BudgetExhausted('deadline')
    previous_alarm = signal.signal(signal.SIGALRM, expired)
    try:
        result = run(settings, research_started_callback=lambda seconds: signal.setitimer(signal.ITIMER_REAL, seconds))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1 if result['status'] == 'failed' else 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
