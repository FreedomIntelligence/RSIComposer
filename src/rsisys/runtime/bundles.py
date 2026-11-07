"""Content identities for explicitly selected local artifacts; no discovery."""
import hashlib
from pathlib import Path


def file_manifest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Artifacts must be regular files without symlinks')
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return {'sha256': sha.hexdigest(), 'bytes': path.stat().st_size}


def tree_manifest(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Artifact root must be a directory without symlinks')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Artifact trees cannot contain symlinks')
        if not path.is_dir():
            result[path.relative_to(root).as_posix()] = file_manifest(path)
    return result


def workspace_path(root, relative):
    root = Path(root).resolve()
    if (not isinstance(relative, str) or not relative or '\x00' in relative
            or Path(relative).is_absolute() or '..' in Path(relative).parts):
        raise ValueError('Use a relative path inside the workspace')
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes workspace')
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError('Workspace paths cannot traverse symlinks')
        current = current.parent
    return path
