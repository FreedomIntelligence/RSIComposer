"""A CPU arithmetic fixture and a hosted Transformers LoRA backend for MetaRSI."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from operator import add, mul, sub
from pathlib import Path
import re
from shutil import copyfile
from tempfile import TemporaryDirectory

from rsicomposer.api import HostedLoRAEngine, execution_capabilities, file_hash, json_bytes
from rsicomposer.runtime.storage.files import atomic_write, sync_directory, sync_tree


OPERATIONS = {'+': add, '-': sub, '*': mul}
EXPRESSION = re.compile(r'Compute ([0-9]{1,6}) ([+*-]) ([0-9]{1,6})\.')
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors')


def digest(value):
    return sha256(json_bytes(value)).hexdigest()


def model_stamp(path):
    if path.is_symlink() or not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError('MetaRSI base_model must be a local Transformers model directory')
    files = {}
    for item in sorted(path.rglob('*')):
        if item.is_symlink() or not (item.is_dir() or item.is_file()):
            raise ValueError('MetaRSI base_model must be a regular local tree without symlinks')
        if item.is_file():
            stat = item.stat()
            files[item.relative_to(path).as_posix()] = (stat.st_dev, stat.st_ino, stat.st_size,
                                                        stat.st_mtime_ns, stat.st_ctime_ns)
    return files


def model_identity(path):
    """Stream a fingerprint of pinned local weights and their tokenizer/config."""
    stamp = model_stamp(path)
    identity = digest({name: file_hash(path / name) for name in stamp})
    if model_stamp(path) != stamp:
        raise ValueError('Pinned MetaRSI base model changed while computing its identity')
    return identity


class FixtureEngine:
    """Intentionally limited arithmetic, with no answer-key access or learned weights."""
    name = 'meta-rsi-arithmetic-fixture-v1'
    simulated = True

    def generate(self, prompts, count, *, model, harness, round_number, sample, temperature):
        result = []
        for prompt in prompts:
            match = EXPRESSION.fullmatch(prompt['question'])
            if not match:
                result.append(['Unsupported fixture question.'] * count)
                continue
            a, operation, b = match.groups()
            allowed = ('+', '-', '*') if model['checkpoint'] is not None else (
                ('+', '-') if harness['instruction'] != 'seed' else ('+',))
            if operation not in allowed:
                result.append(['Unsupported operation in this fixture mode.'] * count)
                continue
            answer = OPERATIONS[operation](int(a), int(b))
            completion = f'\\boxed{{{answer}}}'
            # Deliberate invalid and duplicate examples exercise verifier and deduplication.
            variations = [completion, 'No final answer.', completion,
                          f'{a} {operation} {b} = {answer}. Final: {completion}']
            result.append([variations[i % len(variations)] for i in range(count)])
        return result

    def fit(self, rows, *, output, round_number, steps):
        # The immutable checkpoint manifest records the simulated transition, not a weight file.
        return {'steps': 0, 'samples': len(rows)}


_WIDE_ROLLOUT_ROLES = tuple(f'rollout_{index}' for index in range(4))


@dataclass(frozen=True)
class _RolloutLane:
    """Route an unmodified hosted worker's rollout request to a declared component lane."""
    component: object
    role: str

    @property
    def owner(self):
        return self.component.owner

    def acquire(self, role):
        if role != 'rollout':
            raise ValueError('MetaRSI rollout lanes cannot acquire training resources')
        return self.component.acquire(self.role)


def _validate_rollout_placement(resources):
    config = resources.owner.config
    if config.get('executor', 'local') != 'local':
        raise ValueError('MetaRSI rollout_shards=4 requires the local Host executor')
    placement, pools = config.get('placement', {}), config.get('pools', {})
    devices = set()
    for lane in _WIDE_ROLLOUT_ROLES:
        role = f'{resources.name}.{lane}'
        entry = placement.get(role, {})
        pool = pools.get(entry.get('pool'), {})
        inventory = pool.get('gpus', list(range(pool.get('gpus_per_node', 0))))
        if entry.get('gpus', 1) != 1 or len(inventory) != 1 or inventory[0] in devices:
            raise ValueError(f'MetaRSI role {role} requires explicit placement in a distinct singleton GPU pool')
        devices.add(inventory[0])


class TransformersEngine:
    """Use hosted LoRA workers, optionally sharding each generation batch across four lanes."""
    name = 'transformers-lora-v1'
    simulated = False

    def __init__(self, context, base, options):
        self.base = Path(base).resolve(strict=True)
        config = dict(options or {})
        shards = config.pop('rollout_shards', 1)
        if type(shards) is not int or shards not in (1, 4) or (shards == 4 and config.get('device') != 'cuda'):
            raise ValueError('MetaRSI rollout_shards must be 1 or be 4 on CUDA')
        self.backend = HostedLoRAEngine(config)
        self._rollouts = ()
        if shards == 1:
            self.backend.bind_execution(context)
            return

        resources, self.backend.executor = execution_capabilities(context)
        self.backend.resources = resources
        _validate_rollout_placement(resources)
        self._rollouts = tuple(self._rollout_backend(resources, role, index)
                               for index, role in enumerate(_WIDE_ROLLOUT_ROLES))
        # The normal LabGroup registers the Lab's resource_roles hook. Standalone
        # users of this engine have no such hook and own these same declarations here.
        from rsicomposer.component_context import LabContext
        if not isinstance(context, LabContext):
            resources.declare(self.resource_roles())

    def _rollout_backend(self, resources, role, index):
        backend = copy(self.backend)
        backend.resources = _RolloutLane(resources, role)
        # Avoid resetting each worker to the identical sampling stream. Lane zero
        # preserves the serial seed, including on a single-prompt generation.
        if index:
            seed = int(digest([backend.config.get('seed', 73), role])[:15], 16)
            backend.config = dict(backend.config, seed=seed)
        return backend

    def resource_roles(self):
        roles = self.backend.resource_roles()
        if self._rollouts:
            rollout = roles.pop('rollout')
            roles.update(dict.fromkeys(_WIDE_ROLLOUT_ROLES, rollout))
        return roles

    def generate(self, prompts, count, *, model, harness, round_number, sample, temperature):
        rows = [{'prompt': row['prompt']} for row in prompts]

        def generate(backend, batch):
            return backend.generate(self.base, batch, count, round_number, sample,
                                    adapter=model['checkpoint'], temperature=temperature)

        if not self._rollouts:
            return generate(self.backend, rows)
        if not rows:
            return []
        width, remainder = divmod(len(rows), len(self._rollouts))
        results = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=min(len(rows), len(self._rollouts)), thread_name_prefix='meta-rsi-rollout') as pool:
            pending, start = {}, 0
            for index, backend in enumerate(self._rollouts):
                stop = start + width + (index < remainder)
                if start == stop:
                    break
                # Preserve the host observation/accounting scope in each worker thread.
                pending[pool.submit(copy_context().run, generate, backend, rows[start:stop])] = (start, stop)
                start = stop
            for future in as_completed(pending):
                start, stop = pending[future]
                try:
                    batch = future.result()
                    if (not isinstance(batch, list) or len(batch) != stop - start
                            or any(not isinstance(group, list) or len(group) != count for group in batch)):
                        raise RuntimeError('MetaRSI rollout worker returned an incomplete generation shard')
                except Exception as exc:
                    exc.add_note(f'While generating MetaRSI rollout positions [{start}, {stop})')
                    raise
                results[start:stop] = batch
        return results

    def generate_policy(self, messages, *, checkpoint, round_number):
        # Same pinned target and adapter, with a grammar-friendly decode distinct from blind reasoning.
        lane = self._rollouts[0] if self._rollouts else self.backend
        worker = HostedLoRAEngine(dict(lane.config, enable_thinking=False, max_new_tokens=768))
        worker.resources, worker.executor = lane.resources, lane.executor
        return worker.generate(self.base, [{'prompt': messages}], 1, round_number, False,
                               adapter=checkpoint, temperature=0.)

    def fit(self, rows, *, output, round_number, steps, overrides=None):
        backend = self.backend
        if overrides is not None:
            rate = overrides.get('learning_rate') if isinstance(overrides, dict) else None
            if (not isinstance(overrides, dict) or set(overrides) != {'learning_rate'}
                    or type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0):
                raise ValueError('MetaRSI fit overrides must specify only a finite, positive learning_rate')
            backend = HostedLoRAEngine(dict(self.backend.config, learning_rate=rate))
            backend.resources, backend.executor = self.backend.resources, self.backend.executor
        backend.train(self.base, rows, output, round_number, steps=steps)
        training = json.loads((output / 'training.json').read_bytes())
        if (type(training.get('steps')) is not int or training['steps'] < 1
                or training.get('samples') != len(rows)):
            raise RuntimeError('MetaRSI model training did not report completed optimizer steps')
        # Keep elapsed time out of the identity of the published checkpoint.
        return {key: training[key] for key in ('steps', 'samples')}


def _checkpoint_files(path, simulated):
    names = () if simulated else tuple(f'adapter/{name}' for name in ADAPTER_FILES)
    for name in names:
        file = path / name
        if file.is_symlink() or file.parent.is_symlink() or not file.is_file():
            raise ValueError('MetaRSI adapter payload must consist of regular files')
    return {name: file_hash(path / name) for name in names}


def read_checkpoint(root, key, *, simulated):
    """Recover only a completed immutable checkpoint with verified adapter files."""
    if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{64}', key):
        raise ValueError('Invalid MetaRSI checkpoint identity')
    path = root / key
    if root.is_symlink() or path.is_symlink() or any(item.is_symlink() for item in path.rglob('*')):
        raise ValueError('MetaRSI checkpoint contains symlinks')
    manifest = path / 'checkpoint.json'
    if not manifest.is_file():
        raise ValueError('MetaRSI requires a regular checkpoint manifest')
    payload = json.loads(manifest.read_bytes())
    if (digest(payload.get('recipe')) != key or payload.get('simulated') is not simulated
            or payload.get('training_performed') is not (not simulated)
            or payload.get('adapter_files') != _checkpoint_files(path, simulated)
            or type(payload.get('steps')) is not int or not (payload['steps'] == 0 if simulated else payload['steps'] > 0)
            or type(payload.get('samples')) is not int or payload['samples'] < 1):
        raise ValueError('Published MetaRSI checkpoint does not match its recipe or adapter files')
    return payload


def inherit_checkpoint(source, target, model, *, simulated, validate):
    """Copy only the verified checkpoint manifest and fixed adapter filenames."""
    key = model['checkpoint']
    if source.parent.is_symlink() or target.parent.is_symlink() or target.is_symlink():
        raise ValueError('MetaRSI model roots cannot use symlinks')
    payload = read_checkpoint(source, key, simulated=simulated)
    if digest(payload) != model['checkpoint_sha256']:
        raise ValueError('MetaRSI inherited checkpoint disagrees with its candidate')
    validate(payload)
    target.mkdir(parents=True, exist_ok=True)
    if not (target / key).exists():
        with TemporaryDirectory(prefix='.inherit-', dir=target) as temporary:
            stage = Path(temporary) / key
            stage.mkdir()
            if not simulated:
                (stage / 'adapter').mkdir()
                for name in ADAPTER_FILES:
                    copyfile(source / key / 'adapter' / name, stage / 'adapter' / name)
            atomic_write(stage / 'checkpoint.json', json_bytes(payload))
            if read_checkpoint(stage.parent, key, simulated=simulated) != payload:
                raise ValueError('MetaRSI checkpoint changed during inheritance')
            sync_tree(stage)
            stage.rename(target / key)
            sync_directory(target)
    if read_checkpoint(target, key, simulated=simulated) != payload:
        raise ValueError('MetaRSI local checkpoint disagrees with its inherited identity')
    return payload


def fit_checkpoint(root, engine, recipe, rows):
    """Train in a private staging tree, then publish by recipe identity in one rename."""
    key = digest(recipe)
    target = root / key
    if root.is_symlink() or root.parent.is_symlink() or target.is_symlink():
        raise ValueError('MetaRSI model outputs cannot use symlinks')
    root.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with TemporaryDirectory(prefix='.fit-', dir=root) as temporary:
            stage = Path(temporary) / 'checkpoint'
            stage.mkdir()
            overrides = {'overrides': recipe['fit_overrides']} if 'fit_overrides' in recipe else {}
            metrics = engine.fit(rows, output=stage, round_number=recipe.get('worker_round', recipe['round']),
                                 steps=recipe['steps'], **overrides)
            payload = {'recipe': recipe, 'simulated': engine.simulated, 'training_performed': not engine.simulated,
                       **metrics, 'adapter_files': _checkpoint_files(stage, engine.simulated)}
            atomic_write(stage / 'checkpoint.json', json_bytes(payload))
            sync_tree(stage)
            stage.rename(target)
            sync_directory(root)
    payload = read_checkpoint(root, key, simulated=engine.simulated)
    return {'checkpoint': key, 'checkpoint_sha256': digest(payload)}
