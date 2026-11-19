"""Local process identity and group liveness for supervised cleanup."""
import os
from pathlib import Path
import subprocess
import sys


def process_identity(pid):
    """Include zombies: a supervised, unreaped leader still owns its PID/group."""
    try:
        if sys.platform.startswith('linux'):
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            return boot + ':' + fields[19]
        result = subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart='], text=True).strip()
        return result or None
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def process_group_alive(pgid):
    """Zombies cannot execute or retain accelerator resources."""
    if sys.platform.startswith('linux'):
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():
                continue
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != 'Z':
                    return True
            except (OSError, IndexError, ValueError):
                continue
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
