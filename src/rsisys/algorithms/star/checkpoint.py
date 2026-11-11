"""FSDP2 initialization and Hugging Face checkpoint I/O for the local Qwen3.5-9B."""
from collections import defaultdict
import json
from pathlib import Path
import shutil

from .records import shard_rows, WORLD_SIZE

TEXT_PREFIXES = ('model.language_model.', 'lm_head.')
FROZEN_PREFIXES = ('model.visual.', 'mtp.')
ASSET_FILES = (
    'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja', 'vocab.json',
    'merges.txt', 'preprocessor_config.json', 'video_preprocessor_config.json', 'LICENSE',
)
INDEX_NAME = 'model.safetensors.index.json'


def shard_index(base):
    weight_map = json.loads((base / INDEX_NAME).read_text(encoding='utf-8'))['weight_map']
    if not weight_map or any(not key.startswith(TEXT_PREFIXES + FROZEN_PREFIXES)
                             for key in weight_map):
        raise ValueError('Checkpoint must be the original Qwen3.5 safetensors format')
    shards = defaultdict(list)
    for key, filename in sorted(weight_map.items()):
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith('.safetensors') or not (base / filename).is_file()):
            raise ValueError(f'Invalid weight shard in checkpoint index: {filename!r}')
        shards[filename].append(key)
    return dict(sorted(shards.items())), weight_map


def initialize(base, config, shards, *, rank, device):
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor, Shard
    from transformers import Qwen3_5ForConditionalGeneration

    # Keep small, computed nonpersistent RoPE buffers real while deferring all
    # parameters. FP32 optimizer weights are distributed before allocating them.
    # Qwen's optional fused RMSNorm uses text_config.dtype at construction.
    config.text_config.dtype = torch.float32
    with init_empty_weights(include_buffers=False):
        model = Qwen3_5ForConditionalGeneration(config)
    model.model.visual.requires_grad_(False)
    text = model.model.language_model
    mesh = init_device_mesh('cuda', (WORLD_SIZE,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    groups = [text.embed_tokens, *text.layers, model.lm_head, model.model.visual, model]
    for module in groups:
        fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    buffers = [(module, name, buffer) for module in model.modules()
               for name, buffer in module.named_buffers(recurse=False)]
    if any(buffer.is_meta for _, _, buffer in buffers):
        raise RuntimeError('Qwen3.5 created an uninitialized runtime buffer')
    model.to_empty(device=device)
    for module, name, buffer in buffers:
        setattr(module, name, buffer.to(device))
    parameters = dict(model.named_parameters())
    indexed = {name for names in shards.values() for name in names if not name.startswith('mtp.')}
    if indexed != set(parameters):
        raise RuntimeError(f'Checkpoint/model key mismatch: missing={sorted(set(parameters) - indexed)[:8]}, '
                           f'unexpected={sorted(indexed - set(parameters))[:8]}')
    with torch.no_grad():
        for filename, names in shards.items():
            with safe_open(base / filename, framework='pt', device='cpu') as original:
                for name in names:
                    if name.startswith('mtp.'):
                        continue  # Transformers 5.2 does not instantiate Qwen MTP.
                    param = parameters[name]
                    if (not isinstance(param, DTensor) or param.placements != (Shard(0),)
                            or param.dtype != torch.float32):
                        raise RuntimeError(f'Unexpected FSDP2 placement or master dtype for {name}')
                    source = original.get_slice(name)
                    if tuple(source.get_shape()) != tuple(param.shape):
                        raise RuntimeError(f'Checkpoint shape mismatch for {name}')
                    start, end = shard_rows(param.shape[0], rank, WORLD_SIZE)
                    local = param.to_local()
                    if local.shape[0] != end - start:
                        raise RuntimeError(f'FSDP2 shard shape mismatch for {name}')
                    if end > start:
                        local.copy_(source[start:end])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def export(model, base, destination, shards, weight_map, *, rank):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Keep the original architecture, tokenizer/processor, shard layout, and
    # frozen vision/MTP tensors. Only gather one updated text tensor at a time.
    dtypes = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}
    parameters = dict(model.named_parameters())
    total_bytes = 0
    changed = 0
    if rank == 0:
        destination.mkdir()
        for filename in ASSET_FILES:
            if (base / filename).is_file():
                shutil.copyfile(base / filename, destination / filename)
    with torch.no_grad():
        for filename, names in shards.items():
            shard = {}
            with safe_open(base / filename, framework='pt', device='cpu') as original:
                for name in names:
                    if name.startswith(FROZEN_PREFIXES):
                        if rank == 0:
                            shard[name] = original.get_tensor(name)
                        continue
                    full = parameters[name].full_tensor()
                    if rank == 0:
                        dtype = dtypes[original.get_slice(name).get_dtype()]
                        updated = full.to(device='cpu', dtype=dtype).contiguous()
                        changed += not torch.equal(updated, original.get_tensor(name))
                        shard[name] = updated
                    del full
            if rank == 0:
                total_bytes += sum(tensor.numel() * tensor.element_size() for tensor in shard.values())
                save_file(shard, destination / filename, metadata={'format': 'pt'})
            del shard
    if rank == 0:
        index = {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}
        (destination / INDEX_NAME).write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
        return changed
    return None
