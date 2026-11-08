"""Example-owned JSONL task format. Preparation reads files without writing or serving."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

LABELS = {
    'medxpertqa': tuple('ABCDEFGHIJ'), 'medqa': tuple('ABCD'),
    'medmcqa': tuple('ABCD'), 'pubmedqa': ('yes', 'no', 'maybe'),
}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def normalized(question):
    return ' '.join(question.casefold().split())


@dataclass(frozen=True)
class Sample:
    id: str
    benchmark_id: str
    question: str
    options: tuple[tuple[str, str], ...]
    gold: str

    @property
    def prompt(self):
        return self.question + ''.join(f'\n{k}. {v}' for k, v in self.options)


def samples(content: bytes) -> tuple[Sample, ...]:
    rows, ids = [], set()
    for line in content.decode('utf-8').splitlines():
        raw = json.loads(line)
        if set(raw) != {'id', 'benchmark_id', 'question', 'options', 'gold'}:
            raise ValueError('JSONL requires exactly id, benchmark_id, question, options, gold')
        ident, benchmark, question, options, gold = (raw[k] for k in
            ('id', 'benchmark_id', 'question', 'options', 'gold'))
        if (not isinstance(ident, str) or not ident or ident in ids
                or not isinstance(question, str) or not question.strip()
                or benchmark not in LABELS or gold not in LABELS[benchmark]
                or not isinstance(options, dict)):
            raise ValueError('Invalid sample identity, benchmark, question, or gold')
        expected = () if benchmark == 'pubmedqa' else LABELS[benchmark]
        if set(options) != set(expected) or any(not isinstance(v, str) or not v.strip() for v in options.values()):
            raise ValueError('Options must match the benchmark legal labels; PubMedQA uses {}')
        ids.add(ident)
        rows.append(Sample(ident, benchmark, question, tuple((k, options[k]) for k in expected), gold))
    if not rows:
        raise ValueError('Empty JSONL data')
    return tuple(rows)


@dataclass(frozen=True)
class DataRef:
    path: str
    digest: str

    @classmethod
    def freeze(cls, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        samples(content)
        return cls(str(path), sha256(content).hexdigest())

    def read(self):
        content = Path(self.path).read_bytes()
        if sha256(content).hexdigest() != self.digest:
            raise ValueError('Frozen data digest changed')
        return samples(content)


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    role: str
    data: DataRef


@dataclass(frozen=True)
class MedicalTask:
    train: DataRef
    development: tuple[Benchmark, ...]
    guardrails: bool = True
    max_guardrail_drop: float = 0.02
    max_output_tokens: int = 32
    selection_rule: str = "pareto"

    def __post_init__(self):
        if not isinstance(self.train, DataRef) or not isinstance(self.development, tuple):
            raise TypeError('Task requires frozen train/development references')
        if self.selection_rule not in ('pareto', 'primary_guardrail'):
            raise ValueError('Unknown frozen selection_rule')
        ids = [b.benchmark_id for b in self.development]
        if (not ids or len(ids) != len(set(ids)) or any(b not in LABELS for b in ids)
                or sum(b.role == 'primary' for b in self.development) != 1
                or any(b.role not in ('primary', 'guardrail') for b in self.development)):
            raise ValueError('One primary and distinct legal primary/guardrail benchmarks required')
        if (type(self.guardrails) is not bool or type(self.max_guardrail_drop) not in (int, float)
                or not math.isfinite(self.max_guardrail_drop) or not 0 <= self.max_guardrail_drop <= 1
                or type(self.max_output_tokens) is not int or self.max_output_tokens < 1):
            raise ValueError('Invalid fixed acceptance/generation settings')

    @property
    def protocol_ref(self):
        return 'medical-exact-v1:' + sha256(encoded(asdict(self))).hexdigest()

    def check(self):
        development, ids, stems = [], set(), set()
        for benchmark in self.development:
            for row in benchmark.data.read():
                if row.benchmark_id != benchmark.benchmark_id or row.id in ids or normalized(row.question) in stems:
                    raise ValueError('Development membership, ID, or normalized question overlap')
                ids.add(row.id)
                stems.add(normalized(row.question))
                development.append(row)
        validate_training(self.train.read(), development)
        return tuple(development)


def validate_training(rows, excluded):
    """Trusted-side exact overlap check; excluded may include private operator rows.

    This proves no normalized exact question overlap, not semantic separation or
    label correctness. Private exclusions are never serialized into a task.
    """
    forbidden = {normalized(row.question) for row in excluded}
    forbidden_ids = {row.id for row in excluded}
    if any(normalized(row.question) in forbidden or row.id in forbidden_ids for row in rows):
        raise ValueError('Training overlaps a protected evaluation sample')


def prepare(config, *, base_directory=Path('.')):
    """Freeze local paths from a small explicit config; no downloads or conversions."""
    allowed = {'train', 'development', 'guardrails', 'max_guardrail_drop', 'max_output_tokens', 'selection_rule'}
    if set(config) - allowed:
        raise ValueError('Unknown medical task fields')
    def ref(path):
        return DataRef.freeze(Path(base_directory) / path)
    development = tuple(Benchmark(b['benchmark_id'], b['role'], ref(b['path'])) for b in config['development'])
    task = MedicalTask(ref(config['train']), development,
        **{k: v for k, v in config.items() if k not in ('train', 'development')})
    task.check()
    return task
