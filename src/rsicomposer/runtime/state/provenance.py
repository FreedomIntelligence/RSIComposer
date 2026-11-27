"""Run identity and declared ancestry, independent of state restoration."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class RunSource:
    run_id: str
    parent_run_id: str | None = None

    def __post_init__(self):
        for name in ('run_id', 'parent_run_id'):
            value = getattr(self, name)
            if name == 'parent_run_id' and value is None:
                continue
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f'{name} must be a nonblank, trimmed string')
        if self.run_id == self.parent_run_id:
            raise ValueError('A run cannot be its own parent')

    def child(self, run_id: str) -> 'RunSource':
        """Declare a new run derived from this one; does not restore any state."""
        return RunSource(run_id, self.run_id)

    @classmethod
    def from_dict(cls, value: dict) -> 'RunSource':
        return cls(value['run_id'], value.get('parent_run_id'))

    def to_dict(self):
        return asdict(self)


_REVISION_SUFFIXES = ('.py', '.json', '.jsonl', '.txt', '.toml', '.yml', '.yaml', '.md')
_REVISION_SKIP = {'__pycache__', 'build', 'dist'}


def code_revision() -> str:
    """Hash the shipped package source, including the data a run actually reads.

    Python-only hashing let a prompt, dataset manifest or profile table change
    without invalidating snapshots, so a restore could silently pair old
    checkpoints with new packaged data.
    """
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in _REVISION_SUFFIXES:
            continue
        if _REVISION_SKIP.intersection(path.relative_to(root).parts):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
