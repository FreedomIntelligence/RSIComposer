"""Content-addressed method artifacts; no proposal or selection policy."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _entries(path):
    if path.is_symlink():
        raise ValueError('Coding artifacts must not contain symlinks')
    if path.is_file():
        return None
    if not path.is_dir():
        raise ValueError('Coding artifact must be a file or directory')
    entries = {}
    for p in sorted(path.rglob('*')):
        relative = p.relative_to(path)
        if any(part in {'.git', '__pycache__'} for part in relative.parts) or p.suffix == '.pyc':
            continue
        if p.is_symlink():
            raise ValueError('Coding artifacts must not contain symlinks: ' + str(relative))
        if p.is_file():
            entries[relative.as_posix()] = {
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'executable': bool(p.stat().st_mode & 0o111),
            }
        elif not p.is_dir():
            raise ValueError('Unsupported artifact entry: ' + str(relative))
    if not entries:
        raise ValueError('Coding artifact directory is empty')
    return entries


@dataclass(frozen=True)
class CodingArtifact:
    """A method-owned instruction, configuration, model reference or agent tree."""
    path: str
    sha256: str

    @classmethod
    def from_path(cls, path):
        path = Path(path).absolute()
        entries = _entries(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest() if entries is None else digest(entries)
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
