"""CPU-only input contracts and scheduling for the four-GPU training worker."""
from dataclasses import dataclass
import json
import random

WORLD_SIZE = 4
ACCUMULATION = 2
GLOBAL_BATCH = WORLD_SIZE * ACCUMULATION


def load_records(path, field, *, unique_prompts=True):
    records = []
    seen = set()
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSONL') from exc
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ('prompt', field)
            ):
                raise ValueError(f'{path}:{line_number}: expected nonempty prompt and {field}')
            if unique_prompts and row['prompt'] in seen:
                raise ValueError(f'{path}:{line_number}: duplicate held-out prompt')
            seen.add(row['prompt'])
            records.append((row['prompt'], row[field]))
    if not records:
        raise ValueError(f'{path}: expected at least one record')
    return records


@dataclass(frozen=True)
class TrainingExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    supervised_tokens: int


def training_example(tokenizer, prompt, completion, *, max_tokens):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = tokenizer.encode(completion, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if not prefix or not suffix or not isinstance(eos, int):
        raise ValueError('Expected tokenized prompt, completion, and EOS')
    if eos in suffix[:-1]:
        raise ValueError('Completion contains EOS before its final token')
    if suffix[-1] != eos:
        suffix.append(eos)
    if len(prefix) + len(suffix) > max_tokens:
        raise ValueError(f'Prompt plus completion exceeds {max_tokens} tokens')
    return TrainingExample(tuple(prefix + suffix), tuple([-100] * len(prefix) + suffix), len(suffix))


def optimizer_batches(size, *, epochs, seed):
    if size < GLOBAL_BATCH:
        raise ValueError('Training requires at least eight accepted completions')
    rng = random.Random(seed)
    for _ in range(epochs):
        indices = list(range(size))
        rng.shuffle(indices)
        for start in range(0, size - GLOBAL_BATCH + 1, GLOBAL_BATCH):
            yield indices[start:start + GLOBAL_BATCH]


def shard_rows(rows, rank, world_size):
    chunk = (rows + world_size - 1) // world_size
    return min(rank * chunk, rows), min((rank + 1) * chunk, rows)
