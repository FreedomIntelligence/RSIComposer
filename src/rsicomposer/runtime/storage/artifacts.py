from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from rsicomposer.loops.contracts.artifacts import ArtifactRef


class ArtifactStore(Protocol):
    """Content access for store-managed references, not arbitrary external URIs."""

    def put(self, content: bytes, *, kind: str, access=None) -> ArtifactRef: ...

    def read(self, ref: ArtifactRef) -> bytes: ...


class InMemoryArtifactStore:
    """Immutable demo objects; only references managed by this instance are accepted."""

    def __init__(self):
        self._items: dict[str, tuple[ArtifactRef, bytes]] = {}
        self._access = {}

    def put(self, content: bytes, *, kind: str, access=None) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        ident = str(uuid4())
        ref = ArtifactRef(ident, kind, f"memory://{ident}", "1", sha256(content).hexdigest())
        self._items[ident] = (ref, content)
        self._access[ident] = artifact_access(kind, access)
        return ref

    def read(self, ref: ArtifactRef) -> bytes:
        stored_ref, content = self._items[ref.artifact_id]
        if (ref.uri != f'memory://{ref.artifact_id}' or ref.version != '1'
                or ref.digest is None):
            raise ValueError('Artifact reference is not managed by this store')
        if stored_ref != ref or sha256(content).hexdigest() != ref.digest:
            raise ValueError("Artifact reference or digest mismatch")
        return content


    def refs(self):
        return tuple(ref for ref, _ in self._items.values())

    def access(self, ref):
        self.read(ref)
        return dict(self._access[ref.artifact_id])

    def restrict(self, ref, *, owner, split):
        self.read(ref)
        if not self._access[ref.artifact_id]['sealed']:
            self._access[ref.artifact_id] = artifact_access(ref.kind,
                {'owner': owner, 'split': split, 'sealed': True})


class FileArtifactStore:
    """Immutable local artifacts; only references managed under this root are accepted."""

    def __init__(self, root, *, uri_scheme="artifact"):
        from pathlib import Path
        if uri_scheme not in {"artifact", "file"}:
            raise ValueError("File artifact store supports artifact:// and file:// only")
        self.uri_scheme = uri_scheme
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

    def put(self, content: bytes, *, kind: str, access=None) -> ArtifactRef:
        from io import BytesIO
        if not isinstance(content, bytes):
            raise TypeError('artifact content must be bytes')
        return self._put_stream(BytesIO(content), kind=kind, access=access)

    def put_file(self, path, *, kind: str, access=None) -> ArtifactRef:
        from pathlib import Path
        with Path(path).open('rb') as stream:
            return self._put_stream(stream, kind=kind, access=access)

    def _put_stream(self, stream, *, kind: str, access=None) -> ArtifactRef:
        from dataclasses import asdict
        import os
        from pathlib import Path
        import shutil
        import tempfile
        from rsicomposer.runtime.storage.files import atomic_write, json_bytes, sync_directory
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
            uri = ((self.root / ident / 'data').as_uri() if self.uri_scheme == 'file' else f'artifact://{ident}')
            ref = ArtifactRef(ident, kind, uri, '1', digest.hexdigest())
            atomic_write(staging / 'ref.json', json_bytes(asdict(ref)))
            atomic_write(staging / 'access.json', json_bytes(artifact_access(kind, access)))
            staging.rename(self._directory(ident))
            sync_directory(self.root)
            return ref
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def resolve(self, ref: ArtifactRef):
        import json
        from urllib.parse import unquote, urlsplit
        from pathlib import Path
        uri = urlsplit(ref.uri)
        file_uri = (uri.scheme == 'file' and not uri.netloc and not uri.query and not uri.fragment
                    and Path(unquote(uri.path)).is_absolute()
                    and Path(unquote(uri.path)).parts[-2:] == (ref.artifact_id, 'data'))
        if ((ref.uri != f'artifact://{ref.artifact_id}' and not file_uri) or ref.version != '1'
                or ref.digest is None):
            raise ValueError('Artifact reference is not managed by this store')
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

    def read(self, ref: ArtifactRef) -> bytes:
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


    def access(self, ref):
        import json
        self.resolve(ref)
        path = self._directory(ref.artifact_id) / 'access.json'
        if path.is_symlink():
            raise ValueError('Artifact access metadata may not be a symlink')
        raw = json.loads(path.read_text())
        checked = artifact_access(ref.kind, raw)
        if checked != raw:
            raise ValueError('Invalid artifact access metadata')
        return checked

    def restrict(self, ref, *, owner, split):
        from rsicomposer.runtime.storage.files import atomic_write, json_bytes
        if self.access(ref)['sealed']:
            return
        atomic_write(self._directory(ref.artifact_id) / 'access.json', json_bytes(
            artifact_access(ref.kind, {'owner': owner, 'split': split, 'sealed': True})))


def artifact_access(kind, access=None):
    raw = dict(access) if access is not None else {'owner': 'host', 'split': None, 'sealed': False}
    raw.setdefault('schema', 1)
    if (set(raw) != {'schema', 'owner', 'split', 'sealed'} or type(raw['schema']) is not int or raw['schema'] != 1
            or not isinstance(raw['owner'], str) or not raw['owner'] or type(raw['sealed']) is not bool
            or (raw['split'] is not None and (not isinstance(raw['split'], str) or not raw['split']))):
        raise ValueError('Invalid artifact access metadata')
    if kind == 'sealed-evaluation':
        raw['sealed'] = True
    return raw
