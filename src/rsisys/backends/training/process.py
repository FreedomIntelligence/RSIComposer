"""Trusted POSIX process bridge and optional Transformers/PEFT worker; no shell."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.backend import BackendCapabilities, validate_training_capabilities
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus, JobProgress
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.metrics import validated_metrics
from rsisys.loops.contracts.training import ResumeMode, TrainingResult
from rsisys.runtime.local_process import _kill_process_group, _wait_for_process_exit
from rsisys.runtime.process_store import _json, _write, _lock

_WORKER_ENV = ('PATH', 'HOME', 'LANG', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_HOME',
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
    'NVIDIA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
    'ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH', 'ASCEND_AICPU_PATH',
    'ASCEND_CUSTOM_OPP_PATH', 'TOOLCHAIN_HOME', 'TBE_IMPL_PATH', 'VIRTUAL_ENV',
    'PYTORCH_NPU_ALLOC_CONF', 'ASCEND_LAUNCH_BLOCKING', 'TASK_QUEUE_ENABLE',
    'HCCL_CONNECT_TIMEOUT', 'HCCL_EXEC_TIMEOUT', 'HCCL_SOCKET_IFNAME', 'HCCL_IF_IP',
    'HCCL_WHITELIST_DISABLE', 'HCCL_INTRA_ROCE_ENABLE', 'HCCL_INTRA_PCIE_ENABLE',
    'ROCR_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
    'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_SOCKET_IFNAME', 'NCCL_SOCKET_FAMILY',
    'NCCL_IB_HCA', 'NCCL_IB_GID_INDEX', 'NCCL_IB_QPS_PER_CONNECTION', 'NCCL_IB_DISABLE', 'NCCL_IB_TIMEOUT', 'NCCL_IB_RETRY_CNT',
    'NCCL_P2P_DISABLE', 'NCCL_P2P_LEVEL', 'NCCL_SOCKET_NTHREADS', 'NCCL_NSOCKS_PERTHREAD',
    'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_CUMEM_ENABLE',
    'NCCL_SHM_DISABLE', 'NCCL_NET_GDR_LEVEL', 'NCCL_CROSS_NIC',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING', 'TORCH_NCCL_BLOCKING_WAIT',
    'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'TOKENIZERS_PARALLELISM',
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'XDG_CACHE_HOME')


def _identity(pid):
    """Linux boot/start ticks, or POSIX ps start time; zombies count as gone."""
    try:
        if sys.platform.startswith('linux'):
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                return None
            return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + fields[19]
        value = subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'stat='],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return ' '.join(value.split()[:-1]) if value and not value.split()[-1].startswith('Z') else None
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


@dataclass(frozen=True)
class TrainingCommand:
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 3600

    def __post_init__(self):
        if not self.argv or not all(isinstance(arg, str) and '\0' not in arg for arg in self.argv):
            raise ValueError('Explicit argv strings are required')
        if not Path(self.cwd).is_dir():
            raise ValueError('cwd must be an existing directory')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive and finite')
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.env.items()):
            raise ValueError('env must contain string pairs')


def _num_processes(value):
    if type(value) is not int or value < 1:
        raise ValueError('num_processes must be a positive integer')
    return value


def transformers_peft_command(config_ref, *, cwd, python=sys.executable, num_processes=1):
    """Native single-node Trainer launch; one complete replica per accelerator in DDP.

    Ascend recipes explicitly set device_type='npu'. Select allocated physical NPUs
    with ASCEND_RT_VISIBLE_DEVICES before launch (e.g. '13' for one worker); each
    worker uses its logical local rank. The supplied Python must have torch_npu.
    """
    count = _num_processes(num_processes)
    launcher = (python,) if count == 1 else (python, '-m', 'torch.distributed.run',
        '--standalone', '--nnodes=1', f'--nproc-per-node={count}')
    return TrainingCommand((*launcher, '-m', 'rsisys.backends.training.process', 'transformers-peft',
                            str(config_ref)), str(cwd))


def llamafactory_command(config_ref, *, cwd, executable='llamafactory-cli'):
    """Native CLI only. A caller-supplied result reader must publish its output."""
    return TrainingCommand((executable, 'train', str(config_ref)), str(cwd))


def _result(data):
    candidate = dict(data['candidate'])
    candidate['changes'] = {k: ArtifactRef(**v) for k, v in candidate['changes'].items()}
    candidate['affected_roles'] = tuple(candidate['affected_roles'])
    metrics = validated_metrics(data.get('metrics', {}), 'Training metrics')
    if type(data.get('simulated', False)) is not bool:
        raise ValueError('simulated must be boolean')
    return TrainingResult(ModelRef(**data['base']), ModelRef(**data['model']),
                          Candidate(**candidate), metrics, data.get('simulated', False))


def _native_inputs(request, config_ref):
    """Validate and fingerprint native inputs without importing the frameworks."""
    config_path = Path(config_ref).resolve()
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    _num_processes(config.get('num_processes', 1))
    if config.get('device_type', 'cuda') not in ('cuda', 'npu'):
        raise ValueError('device_type must be cuda or npu')
    if config.get('model_auto_class', 'causal_lm') not in ('causal_lm', 'image_text_to_text'):
        raise ValueError('model_auto_class must be causal_lm or image_text_to_text')
    if 'use_cache' in config and type(config['use_cache']) is not bool:
        raise ValueError('use_cache must be boolean')
    if config.get('initial_model') != request['base']:
        raise ValueError('Recipe initial_model must explicitly match request.base')
    data_path = Path(config['train_dataset'])
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data_path = data_path.resolve()
    raw_data = data_path.read_bytes()
    digest = hashlib.sha256(raw_data).hexdigest()
    from urllib.parse import urlsplit, unquote
    for ref in request.get('data_refs', []):
        parsed = urlsplit(ref['uri'])
        if parsed.scheme not in ('', 'file') or parsed.netloc not in ('', 'localhost'):
            raise ValueError('Built-in worker data_refs must identify the native local tokenized file')
        if Path(unquote(parsed.path)).resolve() != data_path:
            raise ValueError('Request data_refs differ from recipe train_dataset; convert explicitly')
        if ref.get('digest') and ref['digest'].removeprefix('sha256:') != digest:
            raise ValueError('Native training data digest mismatch')
    rows = [json.loads(line) for line in raw_data.splitlines() if line.strip()]
    if not rows or any('input_ids' not in row or 'labels' not in row for row in rows):
        raise ValueError('train_dataset must be native tokenized JSONL with input_ids and labels')
    evidence = {'recipe': str(config_path), 'recipe_sha256': hashlib.sha256(raw_config).hexdigest(),
        'data': str(data_path), 'data_sha256': digest, 'dataset_rows': len(rows),
        'initial_model_binding': request['base'],
        'initialization': 'random_from_config' if 'model_config' in config else 'pretrained',
        'model_config': config.get('model_config'), 'model_name_or_path': config.get('model_name_or_path'),
        'model_kwargs': config.get('model_kwargs', {}),
        'model_auto_class': config.get('model_auto_class', 'causal_lm'),
        'device_type': config.get('device_type', 'cuda'),
        'use_cache': config.get('use_cache'),
        'weight_identity_verified': False}
    return config, rows, evidence


class ProcessTrainingBackend:
    """Durable at-most-once process launch on one trusted POSIX host/filesystem.

    command(request, job_dir) adapts native configs. None selects our SFT worker.
    result_reader(request, job_dir) may read native CLI outputs after exit zero.
    Neither polling nor reattach launches work. A lost supervisor is UNKNOWN.
    """

    def __init__(self, root, command=None, *, capabilities=None, result_reader=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = 'process:' + hashlib.sha256(str(self.root).encode()).hexdigest()
        self.command = command
        self.result_reader = result_reader
        self._capabilities = capabilities or BackendCapabilities(methods=('sft',),
            resume_modes=(ResumeMode.RESTART, ResumeMode.WEIGHTS_ONLY) if command is None else
            (ResumeMode.RESTART,), supports_reattach=True)
        if self._capabilities.update_fields or self._capabilities.supports_checkpoint:
            raise ValueError('Process bridge does not implement live updates or checkpoint export')

    def capabilities(self):
        return self._capabilities

    def _directory(self, job):
        if job.backend != self.backend or len(job.job_id) != 64 or any(c not in '0123456789abcdef' for c in job.job_id):
            raise ValueError('Foreign job reference')
        directory = self.root / job.job_id
        if not (directory / 'receipt.json').exists():
            raise ValueError('Unknown job')
        return directory

    def submit(self, request, *, idempotency_key):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError('An idempotency key is required')
        validate_training_capabilities(self.capabilities(), method=request.method,
                                       origin=getattr(request, 'origin', None))
        origin = getattr(request, 'origin', None)
        if origin is not None and origin.initial_model != request.base:
            raise ValueError('Training origin initial_model must match request.base')
        payload = {'request_type': type(request).__name__, 'request': asdict(request)}
        job = JobRef(hashlib.sha256(idempotency_key.encode()).hexdigest(), self.backend)
        directory = self.root / job.job_id
        with _lock(self.root / '.submit.lock'):
            directory.mkdir(exist_ok=True)
            receipt_path = directory / 'receipt.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt['payload'] != json.loads(_json(payload)):
                    raise ValueError('Idempotency key reused with a different request')
                if (directory / 'state.json').exists():
                    return job
            else:
                if self.command is None:
                    config, _, evidence = _native_inputs(payload['request'], request.recipe_ref)
                    _write(directory / 'native-inputs.json', evidence)
                    command = transformers_peft_command(str(Path(request.recipe_ref).resolve()),
                        cwd=directory, num_processes=config.get('num_processes', 1))
                else:
                    command = self.command(request, directory)
                if not isinstance(command, TrainingCommand):
                    raise TypeError('command must return TrainingCommand')
                _write(directory / 'request.json', payload)
                _write(receipt_path, {'payload': payload, 'command': asdict(command)})
            # A supervisor also holds execution.lock for its lifetime, including exit
            # persistence. Concurrent retries cannot start the native command twice.
            environment = dict(os.environ)
            environment['PYTHONPATH'] = os.pathsep.join(str(p) for p in sys.path if p)
            with (directory / 'supervisor.log').open('ab') as log:
                process = subprocess.Popen([sys.executable, '-m', 'rsisys.backends.training.process',
                    'supervise', str(directory)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    env=environment, start_new_session=True)
            # Reap in the live controller; the detached child survives controller loss.
            import threading
            threading.Thread(target=process.wait, daemon=True).start()
        return job

    def poll(self, job):
        directory = self._directory(job)
        state_path = directory / 'state.json'
        if not state_path.exists():
            return JobOutcome(job, JobStatus.QUEUED)
        state = json.loads(state_path.read_text())
        status = JobStatus(state['status'])
        progress = None
        if (directory / 'progress.json').exists():
            progress = JobProgress(**json.loads((directory / 'progress.json').read_text()))
        if status == JobStatus.RUNNING and _identity(state['pid']) != state['identity']:
            # The supervisor may have persisted its terminal receipt and exited
            # between our first read and the identity check.
            state = json.loads(state_path.read_text())
            status = JobStatus(state['status'])
            if status == JobStatus.RUNNING:
                return JobOutcome(job, JobStatus.UNKNOWN, error='Supervisor identity lost; no restart attempted', progress=progress)
        if status != JobStatus.SUCCEEDED:
            return JobOutcome(job, status, error=state.get('error'), progress=progress)
        # Validate once and retain an immutable terminal result across reattachments.
        with _lock(directory / 'result.lock'):
            cached = directory / 'validated-result.json'
            failure = directory / 'invalid-result.json'
            if failure.exists():
                return JobOutcome(job, JobStatus.FAILED,
                    error=json.loads(failure.read_text())['error'], progress=progress)
            try:
                if cached.exists():
                    result = _result(json.loads(cached.read_text()))
                else:
                    payload = json.loads((directory / 'request.json').read_text())
                    if self.result_reader is not None:
                        from types import SimpleNamespace
                        request = SimpleNamespace(**payload['request'])
                        request.base = ModelRef(**request.base)
                        result = self.result_reader(request, directory)
                        result = _result(asdict(result))
                    else:
                        result = _result(json.loads((directory / 'result.json').read_text()))
                    request = payload['request']
                    if result.base != ModelRef(**request['base']):
                        raise ValueError('Training result base mismatch')
                    role = 'decision_maker' if payload['request_type'] == 'DecisionTrainingRequest' else 'actor'
                    if role not in result.candidate.affected_roles:
                        raise ValueError('Training result role mismatch')
                    origin = request.get('origin')
                    if origin and result.candidate.parent_system != origin['parent_system']:
                        raise ValueError('Training result parent_system mismatch')
                    _write(cached, asdict(result))
            except Exception as exc:
                error = f'Invalid training result: {exc}'
                _write(failure, {'error': error})
                return JobOutcome(job, JobStatus.FAILED, error=error, progress=progress)
        return JobOutcome(job, JobStatus.SUCCEEDED, result, simulated=result.simulated, progress=progress)

    def reattach(self, job):
        return self.poll(job)

    def cancel(self, job):
        directory = self._directory(job)
        _write(directory / 'cancel.json', {'requested': True})

    def result(self, job):
        path = self._directory(job) / 'validated-result.json'
        if not path.exists():
            raise RuntimeError('Job has no validated successful result; poll first')
        return _result(json.loads(path.read_text()))


def _supervise(directory):
    with (directory / 'execution.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (directory / 'state.json').exists():
            return  # Including UNKNOWN: never replay an uncertain external launch.
        command = json.loads((directory / 'receipt.json').read_text())['command']
        state = {'status': 'running', 'pid': os.getpid(), 'identity': _identity(os.getpid())}
        _write(directory / 'state.json', state)
        process = None
        try:
            if (directory / 'cancel.json').exists():
                state['status'] = 'cancelled'
                return
            environment = {k: os.environ[k] for k in _WORKER_ENV if k in os.environ}
            environment.update(command['env'])
            environment.update(RSISYS_REQUEST_FILE=str(directory / 'request.json'),
                               RSISYS_RESULT_FILE=str(directory / 'result.json'),
                               RSISYS_JOB_DIR=str(directory))
            started = time.monotonic()
            with (directory / 'stdout.log').open('ab') as stdout, (directory / 'stderr.log').open('ab') as stderr:
                process = subprocess.Popen(command['argv'], cwd=command['cwd'], env=environment,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
                state.update(child_pid=process.pid, child_identity=_identity(process.pid))
                _write(directory / 'state.json', state)
                while True:
                    try:
                        _wait_for_process_exit(process, .05)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if (directory / 'cancel.json').exists():
                        state['status'] = 'cancelled'
                        break
                    if time.monotonic() - started > command['timeout_seconds']:
                        state.update(status='failed', error='Training command timed out')
                        break
                if state['status'] == 'running':
                    _kill_process_group(process)
                    code = process.wait()
                    state.update(status='succeeded' if code == 0 else 'failed', exit_code=code)
                    if code:
                        state['error'] = f'Training command exited {code}; see stderr.log'
        except Exception as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            if process is not None:
                try:
                    # The direct child is still unreaped when running. Never
                    # signal a recycled leader PID after wait/poll reaped it.
                    if process.returncode is None:
                        _kill_process_group(process)
                except ProcessLookupError:
                    pass
                process.wait()
            _write(directory / 'state.json', state)


def _worker_ranks(config):
    """Accept only a single torchrun node, never infer ranks from device visibility."""
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world < 1 or not 0 <= rank < world or not 0 <= local < world:
        raise ValueError('Invalid distributed worker ranks')
    if world > 1 and (any(key not in os.environ for key in
            ('RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')) or
            int(os.environ['LOCAL_WORLD_SIZE']) != world or rank != local):
        raise ValueError('Built-in worker supports single-node torchrun only')
    if 'num_processes' in config and _num_processes(config['num_processes']) != world:
        raise ValueError('Recipe num_processes differs from launcher WORLD_SIZE')
    return rank, local, world


@contextmanager
def _worker_group(local_rank, world_size, device_type='cuda'):
    if device_type not in ('cuda', 'npu'):
        raise ValueError('device_type must be cuda or npu')
    if world_size == 1 and device_type != 'npu':
        yield None
        return
    import torch
    if device_type == 'npu':
        # Register the optional backend only on explicit opt-in, before importing
        # Transformers/Accelerate or seeding, loading weights, and creating HCCL.
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('device_type=npu requires torch_npu in the worker Python') from exc
        if not torch.npu.is_available() or torch.npu.device_count() != world_size:
            raise ValueError('NPU SFT requires exactly WORLD_SIZE visible NPUs; '
                             'select allocated devices with ASCEND_RT_VISIBLE_DEVICES')
        torch.npu.set_device(local_rank)
    else:
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise ValueError('Multi-process SFT requires one visible CUDA GPU per local rank')
        torch.cuda.set_device(local_rank)  # Before NCCL, weight loading, or Trainer setup.
    if world_size == 1:
        yield None
        return
    distributed = torch.distributed
    distributed.init_process_group(backend='hccl' if device_type == 'npu' else 'nccl')
    try:
        yield distributed
    finally:
        distributed.destroy_process_group()


def _training_exposure(trainer, distributed):
    local = {'presented_rows': trainer.presented_rows,
             'presented_input_tokens': trainer.presented_tokens}
    counts = [local]
    if distributed is not None:
        counts = [None] * distributed.get_world_size()
        distributed.all_gather_object(counts, local)
    return {'presented_rows': sum(count['presented_rows'] for count in counts),
        'presented_input_tokens': sum(count['presented_input_tokens'] for count in counts),
        'per_rank_exposure': [dict(rank=rank, **count) for rank, count in enumerate(counts)]}


def _transformers_peft(config_ref):
    directory = Path(os.environ['RSISYS_JOB_DIR'])
    payload = json.loads(Path(os.environ['RSISYS_REQUEST_FILE']).read_text())
    request = payload['request']
    config, rows, evidence = _native_inputs(request, config_ref)
    if (directory / 'native-inputs.json').exists():
        if json.loads((directory / 'native-inputs.json').read_text()) != evidence:
            raise ValueError('Native recipe/data changed after submission')
    arguments = dict(config.get('training_args', {}))
    output = Path(arguments.setdefault('output_dir', str(directory / 'adapter'))).resolve()
    if arguments.get('resume_from_checkpoint'):
        raise ValueError('Resume must be declared through TrainingOrigin')
    rank, local_rank, world_size = _worker_ranks(config)
    device_type = config.get('device_type', 'cuda')
    if world_size > 1 or device_type == 'npu':
        if (any(arguments.get(key) for key in
                ('deepspeed', 'fsdp', 'parallelism_config', 'use_cpu', 'no_cuda')) or
                config.get('model_kwargs', {}).get('tp_plan') is not None):
            raise ValueError(f'Worker supports native {"NPU" if device_type == "npu" else "CUDA"} Trainer DDP only')
    if device_type == 'npu':
        # Transformers 5 defaults to a CUDA-oriented fused AdamW optimizer.
        arguments.setdefault('optim', 'adamw_torch')
        arguments.setdefault('dataloader_pin_memory', False)
    if world_size > 1:
        arguments['local_rank'] = local_rank
        arguments['ddp_backend'] = 'hccl' if device_type == 'npu' else 'nccl'
        arguments['save_on_each_node'] = False
        if arguments.get('gradient_checkpointing') and arguments.get('ddp_find_unused_parameters') is None:
            # PEFT wrappers do not inherit Trainer's PreTrainedModel checkpointing default.
            arguments['ddp_find_unused_parameters'] = False
    with _worker_group(local_rank, world_size, device_type) as distributed:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)  # Never overwrite existing weights.
        if distributed is not None:
            distributed.barrier()
        _train_transformers_peft(directory, payload, config, rows, evidence, arguments,
                                 output, rank, local_rank, world_size, distributed)


def _train_transformers_peft(directory, payload, config, rows, evidence, arguments,
                             output, rank, local_rank, world_size, distributed):
    request = payload['request']
    # Importing this adapter never imports torch or optional training packages.
    from transformers import AutoConfig, Trainer, TrainingArguments, TrainerCallback, default_data_collator
    if config.get('model_auto_class', 'causal_lm') == 'image_text_to_text':
        from transformers import AutoModelForImageTextToText as ModelAutoClass
    else:
        from transformers import AutoModelForCausalLM as ModelAutoClass
    from peft import LoraConfig, get_peft_model, PeftModel
    import torch

    origin = request.get('origin') or {}
    mode = origin.get('resume_mode', 'restart')
    if mode == 'full_state':
        raise NotImplementedError('Built-in worker does not certify full optimizer/RNG state resume')
    torch.manual_seed(config.get('seed', 0))
    if 'model_config' in config:
        native = dict(config['model_config'])
        model = ModelAutoClass.from_config(AutoConfig.for_model(native.pop('model_type'), **native))
    else:
        model_kwargs = dict(config.get('model_kwargs', {}))
        if config.get('device_type') == 'npu':
            # Logical rank zero maps to physical 13 when visibility is '13'.
            # Even a single worker must not infer placement across other devices.
            model_kwargs['device_map'] = {'': f'npu:{local_rank}'}
        elif distributed is not None:
            # Inference auto-sharding must never span GPUs during DDP training.
            model_kwargs['device_map'] = {'': local_rank}
        model = ModelAutoClass.from_pretrained(config['model_name_or_path'], **model_kwargs)
    if 'use_cache' in config:
        model.config.use_cache = config['use_cache']
        if getattr(model.config, 'text_config', None) is not None:
            model.config.text_config.use_cache = config['use_cache']
    checkpoint = origin.get('checkpoint')
    if mode == 'weights_only' and checkpoint:
        uri = checkpoint['uri']
        if '://' in uri:
            if not uri.startswith('file://'):
                raise ValueError('Built-in worker requires a local PEFT checkpoint path')
            uri = uri[7:]
        model = PeftModel.from_pretrained(model, uri, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(**config['lora_config']))
    # The explicit recipe controls native Trainer behavior, including checkpoints.
    arguments.setdefault('report_to', 'none')
    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if rank != 0:
                return
            metrics = {k: v for k, v in (logs or {}).items() if type(v) in (int, float) and math.isfinite(v)}
            _write(directory / 'progress.json', asdict(JobProgress(step=state.global_step,
                total_steps=state.max_steps or None, metrics=metrics, observed_at=time.time())))
    class ObservedTrainer(Trainer):
        presented_rows = 0
        presented_tokens = 0
        def compute_loss(self, model, inputs, *args, **kwargs):
            if model.training:
                self.presented_rows += inputs['input_ids'].shape[0]
                self.presented_tokens += int(inputs['attention_mask'].sum()) if 'attention_mask' in inputs else inputs['input_ids'].numel()
            return super().compute_loss(model, inputs, *args, **kwargs)
    trainer = ObservedTrainer(model=model, args=TrainingArguments(**arguments), train_dataset=rows,
                      data_collator=default_data_collator, callbacks=[Progress()])
    trained = trainer.train()
    if distributed is not None:
        distributed.barrier()
    # Every rank participates; native Trainer gates actual writes to rank zero.
    trainer.save_model(str(output))
    if distributed is not None:
        distributed.barrier()
    exposure = _training_exposure(trainer, distributed)
    if rank != 0:
        return
    evidence.update(exposure, optimizer_steps=trainer.state.global_step, world_size=world_size,
        exposure_scope='all_worker_ranks' if distributed is not None else 'this_worker_rank',
        output_dir=str(output))
    _write(directory / 'training-evidence.json', evidence)
    base = ModelRef(**request['base'])
    model_ref = ModelRef(**config['result_model']) if 'result_model' in config else ModelRef(
        base.backend_id, base.model_id, 'peft-' + directory.name)
    artifact = ArtifactRef(directory.name, 'peft_adapter', output.as_uri(), version=model_ref.revision)
    evidence_artifact = ArtifactRef(directory.name + '-evidence', 'training_evidence',
        (directory / 'training-evidence.json').as_uri(), digest=hashlib.sha256(
            (directory / 'training-evidence.json').read_bytes()).hexdigest())
    role = 'decision_maker' if payload['request_type'] == 'DecisionTrainingRequest' else 'actor'
    candidate = Candidate(directory.name, base.revision, {'model': artifact, 'training_evidence': evidence_artifact}, (role,),
                          **({'parent_system': origin['parent_system']} if origin else {}))
    metrics = validated_metrics(trained.metrics, 'Training metrics')
    _write(Path(os.environ['RSISYS_RESULT_FILE']), asdict(TrainingResult(base, model_ref, candidate, metrics)))


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('supervise', 'transformers-peft'):
        raise SystemExit('Usage: python -m rsisys.backends.training.process {supervise JOB_DIR|transformers-peft RECIPE.json}')
    if sys.argv[1] == 'supervise':
        _supervise(Path(sys.argv[2]).resolve())
    else:
        _transformers_peft(sys.argv[2])
