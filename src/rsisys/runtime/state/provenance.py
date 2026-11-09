"""Run identity and declared ancestry, independent of state restoration."""

from dataclasses import asdict, dataclass


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
