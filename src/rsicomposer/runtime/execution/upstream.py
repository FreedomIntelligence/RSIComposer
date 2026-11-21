"""Verify an immutable, clean checkout for an optional external backend."""
import subprocess
from pathlib import Path


def checkout(path, revision):
    root = Path(path).resolve(strict=True)
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise ValueError(f'Expected upstream revision {revision}, got {actual}')
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Pinned upstream checkout has tracked modifications')
    return root
