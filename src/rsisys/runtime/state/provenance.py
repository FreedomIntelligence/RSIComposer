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


def code_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
