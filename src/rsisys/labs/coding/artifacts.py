"""Content-addressed method artifacts; no proposal or selection policy."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from rsisys.loops.contracts.domain_agent import path_digest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _entries(path):
    """The files of a tree with their hashes, or None for a single file (see path_digest)."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Coding artifacts must not contain symlinks')
    if path.is_file():
        return None
    if not path.is_dir():
        raise ValueError('Coding artifact must be a file or directory')
    path_digest(path)   # Validates the whole tree with the contract's rules.
    return {p.relative_to(path).as_posix(): None for p in sorted(path.rglob('*'))
            if p.is_file() and not any(part in {'.git', '__pycache__'} for part in p.relative_to(path).parts)
            and p.suffix != '.pyc'}


@dataclass(frozen=True)
class CodingArtifact:
    """A method-owned instruction, configuration, model reference or agent tree."""
    path: str
    sha256: str

    @classmethod
    def from_path(cls, path):
        path = Path(path).absolute()
        entries = _entries(path)
        sha = path_digest(path)
        return cls(str(path), sha)

    def to_json(self):
        return {'path': self.path, 'sha256': self.sha256}

    def verify(self):
        if self != self.from_path(self.path):
            raise ValueError('Coding artifact changed after it was referenced')


def snapshot(artifact, root):
    if isinstance(artifact, dict) and set(artifact) == {'path', 'sha256'}:
        artifact = CodingArtifact(**artifact)
    if not isinstance(artifact, CodingArtifact):
        raise ValueError('Expected a CodingArtifact path and sha256')
    artifact.verify()
    source = Path(artifact.path)
    if source.is_dir() and Path(root).resolve().is_relative_to(source.resolve()):
        raise ValueError('Artifact storage must be outside the source tree')
    store = Path(root) / artifact.sha256
    target = store / 'input'
    if not store.exists():
        temp = store.with_name('.' + store.name + '-' + uuid4().hex)
        temp.mkdir(parents=True)
        try:
            entries = _entries(source)
            if entries is None:
                shutil.copy2(source, temp / 'input')
            else:
                for relative in entries:
                    destination = temp / 'input' / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / relative, destination)
            if CodingArtifact.from_path(temp / 'input').sha256 != artifact.sha256:
                raise ValueError('Coding artifact changed while being copied')
            temp.rename(store)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    pinned = CodingArtifact(str(target), artifact.sha256)
    pinned.verify()
    return pinned
