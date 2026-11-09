"""Bounded descriptor-relative host reads of candidate-controlled files."""
import os
from pathlib import Path
import stat


def read_workspace_file(workspace, relative, *, maximum=1024 * 1024):
    if (not isinstance(relative, str) or not relative or '\0' in relative
            or Path(relative).is_absolute() or any(p in ('..', '.') for p in relative.split('/'))):
        raise ValueError('Candidate paths must be relative workspace files')
    parts = Path(relative).parts
    directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise ValueError('Candidate component must be a bounded regular file')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(maximum + 1)
            after = os.fstat(fd)
            if len(raw) > maximum or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Candidate file changed while freezing')
            return raw
        finally:
            os.close(fd)
    finally:
        os.close(directory)
