from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from rsisys.loops.contracts.artifacts import ArtifactRef


class ArtifactStore(Protocol):
    def put(self, content: bytes, *, kind: str) -> ArtifactRef: ...

    def get(self, ref: ArtifactRef) -> bytes: ...


class InMemoryArtifactStore:
    """Immutable demo objects. References expire with the Python process."""

    def __init__(self):
        self._items: dict[str, tuple[ArtifactRef, bytes]] = {}

    def put(self, content: bytes, *, kind: str) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        ident = str(uuid4())
        ref = ArtifactRef(ident, kind, f"memory://{ident}", "1", sha256(content).hexdigest())
        self._items[ident] = (ref, content)
        return ref

    def get(self, ref: ArtifactRef) -> bytes:
        stored_ref, content = self._items[ref.artifact_id]
        if stored_ref != ref or sha256(content).hexdigest() != ref.digest:
            raise ValueError("Artifact reference or digest mismatch")
        return content


class FileArtifactStore:
    """Immutable local artifacts with portable references and verified content digests."""

    def __init__(self, root):
        from pathlib import Path
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, artifact_id: str):
        import re
        if not isinstance(artifact_id, str) or not re.fullmatch(r'[0-9a-f]{32}', artifact_id):
            raise ValueError('Invalid artifact identity')
        directory = self.root / artifact_id
        if directory.is_symlink():
            raise ValueError('Artifact directories may not be symlinks')
        return directory

    def put(self, content: bytes, *, kind: str) -> ArtifactRef:
        from io import BytesIO
        if not isinstance(content, bytes):
            raise TypeError('artifact content must be bytes')
        return self._put_stream(BytesIO(content), kind=kind)

    def put_file(self, path, *, kind: str) -> ArtifactRef:
        from pathlib import Path
        with Path(path).open('rb') as stream:
            return self._put_stream(stream, kind=kind)

    def _put_stream(self, stream, *, kind: str) -> ArtifactRef:
        from dataclasses import asdict
        import os
        from pathlib import Path
        import shutil
        import tempfile
        from .files import atomic_write, json_bytes, sync_directory
        if not isinstance(kind, str) or not kind:
            raise ValueError('Artifact kind is required')
        ident = uuid4().hex
        staging = Path(tempfile.mkdtemp(prefix='.artifact-', dir=self.root))
        try:
            digest = sha256()
            with (staging / 'data').open('wb') as output:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            ref = ArtifactRef(ident, kind, f'artifact://{ident}', '1', digest.hexdigest())
            atomic_write(staging / 'ref.json', json_bytes(asdict(ref)))
            staging.rename(self._directory(ident))
            sync_directory(self.root)
            return ref
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def resolve(self, ref: ArtifactRef):
        import json
        directory = self._directory(ref.artifact_id)
        if not directory.is_dir():
            raise KeyError(ref.artifact_id)
        metadata, content = directory / 'ref.json', directory / 'data'
        if metadata.is_symlink() or content.is_symlink():
            raise ValueError('Artifact files may not be symlinks')
        stored = ArtifactRef(**json.loads(metadata.read_text()))
        if stored != ref:
            raise ValueError('Artifact reference does not match stored metadata')
        digest = sha256()
        with content.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != ref.digest:
            raise ValueError('Artifact digest mismatch')
        return content

    def get(self, ref: ArtifactRef) -> bytes:
        return self.resolve(ref).read_bytes()

    def refs(self) -> tuple[ArtifactRef, ...]:
        import json
        refs = []
        for directory in sorted(self.root.iterdir()):
            if directory.name.startswith('.'):
                continue
            self._directory(directory.name)
            ref = ArtifactRef(**json.loads((directory / 'ref.json').read_text()))
            if ref.artifact_id != directory.name:
                raise ValueError('Artifact directory and identity do not match')
            self.resolve(ref)
            refs.append(ref)
        return tuple(refs)
