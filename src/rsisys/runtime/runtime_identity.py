"""Identity for installed candidate runtimes without copying installed libraries."""
import hashlib
from pathlib import Path
import platform

from .files import json_bytes
from .isolation import ExecutionPolicy, system_runtime_paths


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inference_policy(python, *, read_only=(), devices=(), environment=()):
    # OS libraries only, not /usr/local, host homes, repository source, research
    # data or credentials. All non-runtime dependencies must be in the bundle.
    paths = [p for p in system_runtime_paths(python) if p not in ('/usr', '/bin', '/sbin', '/System')]
    paths += [p for p in ('/usr/lib', '/System/Library') if Path(p).exists()]
    return ExecutionPolicy(read_only=(*paths, *read_only), devices=devices, environment=environment)


def runtime_identity(python, policy):
    """Hash installed non-OS files; OS libraries bind the host OS build.

    OS roots are part of the trusted host image. Operators must not mutate that
    image during a run. Mutable application runtime mounts are SHA-bound before
    and after inference, including package bytes, not just pip package versions.
    """
    os_roots = ('/usr/lib', '/lib', '/lib64', '/System/Library', '/Library/Apple',
                '/private/var/db/dyld', '/etc/', '/private/var/db/timezone/')
    files = {}
    for name in policy.read_only:
        if any(name == p.rstrip('/') or name.startswith(p.rstrip('/') + '/') for p in os_roots):
            continue
        root = Path(name)
        for path in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if path.is_file():
                files[str(path)] = file_identity(path)
    value = {'protocol': 'installed-inference-runtime/v1', 'python': str(Path(python).resolve()),
             'python_sha256': file_identity(Path(python).resolve()), 'files': files,
             'os_build': {k: getattr(platform.uname(), k) for k in ('system', 'release', 'version', 'machine')},
             'policy': policy.configuration()}
    return {**value, 'sha256': hashlib.sha256(json_bytes(value)).hexdigest()}
