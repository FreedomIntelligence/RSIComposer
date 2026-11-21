"""Host-selected filesystem, process, network and device execution boundaries.

Research code can use every installed program in the mounted runtime and write
its workspace. It cannot mount more host files, contact external services or
gain devices by changing environment variables. Never falls back to local exec.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys



class IsolationUnavailable(RuntimeError):
    pass


def _paths(values):
    result = tuple(sorted({str(Path(v).resolve(strict=True)) for v in values}))
    if '/' in result:
        raise ValueError('An execution policy cannot expose the whole host filesystem')
    return result


@dataclass(frozen=True)
class ExecutionPolicy:
    """Host capabilities, never candidate arguments. Empty network means offline.

    read_only contains installed runtimes/public inputs. Mutable candidate
    dependencies belong in its bundle. devices contains exact character nodes,
    not CUDA_VISIBLE_DEVICES declarations. sockets are host broker endpoints.
    """
    read_only: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    sockets: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        for field in ('read_only', 'devices', 'sockets'):
            object.__setattr__(self, field, _paths(getattr(self, field)))
        for name in self.devices:
            if not stat.S_ISCHR(Path(name).stat().st_mode):
                raise ValueError('Allocated devices must be exact character device nodes')
        for name in self.sockets:
            if not stat.S_ISSOCK(Path(name).stat().st_mode):
                raise ValueError('Broker endpoints must be Unix sockets')
        environment = dict(self.environment)
        if any(not isinstance(k, str) or not k or '=' in k or not isinstance(v, str)
               or '\0' in k + v for k, v in environment.items()):
            raise ValueError('Invalid execution environment')
        object.__setattr__(self, 'environment', tuple(sorted(environment.items())))
        # Ordinary read mounts must never expose arbitrary device nodes or host
        # process handles; /dev and /proc are synthesized by the executor.
        for name in self.read_only:
            if name in ('/dev', '/proc', '/sys') or name.startswith(('/dev/', '/proc/')):
                raise ValueError('Use explicit device capabilities instead of host /dev or /proc mounts')

    def configuration(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, raw):
        return cls(**raw)


def system_runtime_paths(python=sys.executable):
    """Minimal OS and installed Python runtime mounts, excluding host homes."""
    roots = ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc/ld.so.cache',
             '/etc/ld.so.conf', '/etc/ld.so.conf.d', '/etc/localtime']
    if sys.platform == 'darwin':
        roots += ['/System', '/Library/Apple', '/private/var/db/dyld']
    executable = Path(python).resolve(strict=True)
    # A venv symlinks its executable but keeps packages beside bin/python.
    prefix = Path(python).absolute().parent.parent
    roots += [str(executable.parent.parent)]
    if (prefix / 'pyvenv.cfg').is_file():
        roots += [str(prefix)]
    return tuple(str(Path(p).resolve()) for p in roots if Path(p).exists())


class ProcessIsolation:
    def __init__(self, policy: ExecutionPolicy, *, driver=None):
        self.policy = policy
        self.driver = driver or ('seatbelt' if sys.platform == 'darwin' else 'bubblewrap')
        name = {'seatbelt': 'sandbox-exec', 'bubblewrap': 'bwrap'}.get(self.driver)
        self.executable = shutil.which(name) if name else None
        if not self.executable:
            raise IsolationUnavailable(f'Required isolation driver {self.driver} is unavailable')

    def command(self, workspace, argv, environment, *, writable=True, scratch=None):
        workspace = Path(workspace).resolve(strict=True)
        argv = tuple(argv)
        if argv and Path(argv[0]).is_absolute():
            # Runtime mounts use canonical parents. An absolute path containing
            # ../ or a directory symlink can traverse an unmounted ancestor even
            # when its target runtime is mounted. Preserve the executable leaf:
            # resolving bin/python itself would erase a virtualenv's identity.
            executable = Path(argv[0])
            argv = (str(executable.parent.resolve() / executable.name), *argv[1:])
        if any(workspace == Path(p) or Path(p).is_relative_to(workspace)
               for p in (*self.policy.read_only, *self.policy.devices, *self.policy.sockets)):
            raise ValueError('Host runtime/device/broker paths must be outside the writable workspace')
        env = {**environment, **dict(self.policy.environment), 'HOME': str(workspace),
               'TMPDIR': str(workspace) if writable else '/tmp', 'PYTHONDONTWRITEBYTECODE': '1'}
        if scratch is not None:
            scratch = str(Path(scratch).resolve(strict=True))
            if Path(scratch).is_relative_to(workspace) or workspace.is_relative_to(Path(scratch)):
                raise ValueError('Inference scratch must be separate from its frozen bundle')
            env.update(HOME=scratch, TMPDIR=scratch, HF_HOME=scratch + '/hf',
                       XDG_CACHE_HOME=scratch + '/cache')
        if self.driver == 'bubblewrap':
            # This namespace already owns a private /tmp. Workspace and scratch
            # paths may exceed AF_UNIX's pathname limit (e.g. multiprocessing
            # resource-sharer sockets), so do not use them as temporary roots.
            env['TMPDIR'] = '/tmp'
            command = [self.executable, '--die-with-parent', '--new-session', '--unshare-all',
                       '--cap-drop', 'ALL', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
            for name in self.policy.read_only:
                command += ['--ro-bind', name, name]
            # Preserve Linux merged-/usr loader and executable aliases. Mount
            # sources are canonical, while ELF interpreters often use /lib64.
            for alias in ('/bin', '/sbin', '/lib', '/lib64'):
                target = Path(alias).resolve()
                if Path(alias).is_symlink() and any(target.is_relative_to(Path(p))
                        for p in self.policy.read_only):
                    command += ['--symlink', os.readlink(alias), alias]
            command += ['--bind' if writable else '--ro-bind', str(workspace), str(workspace)]
            if scratch is not None:
                command += ['--bind', scratch, scratch]
            for name in self.policy.devices:
                command += ['--dev-bind', name, name]
            for name in self.policy.sockets:
                command += ['--ro-bind', name, name]
            command += ['--chdir', str(workspace), '--', *argv]
        else:
            q = json.dumps
            reads = (*self.policy.read_only, str(workspace), '/dev/null', '/dev/random', '/dev/urandom',
                     *((scratch,) if scratch is not None else ()))
            rules = ['(version 1)', '(deny default)', '(allow process-exec)', '(allow process-fork)',
                     '(allow signal (target self))',
                     '(allow file-read-metadata)',
                     '(allow file-read* (literal "/"))',
                     '(allow file-read* ' + ' '.join('(subpath ' + q(p) + ')' for p in reads) + ')',
                     '(allow file-write* (literal "/dev/null"))']
            if writable:
                rules += ['(allow file-write* (subpath ' + q(str(workspace)) + '))']
                # Workspace Unix services remain available without granting
                # access to host TCP endpoints or the outside network.
                rules += ['(allow network* (subpath ' + q(str(workspace)) + '))']
            if scratch is not None:
                rules += ['(allow file-write* (subpath ' + q(scratch) + '))']
            for name in self.policy.devices:
                rules += ['(allow file-read* file-write* (literal ' + q(name) + '))']
            for name in self.policy.sockets:
                rules += ['(allow network-outbound (literal ' + q(name) + '))']
            command = [self.executable, '-p', '\n'.join(rules), *argv]
        return tuple(command), env

    def check(self):
        """Fail before research admission when namespaces/profiles are disabled."""
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            from dataclasses import replace
            probe = ProcessIsolation(replace(self.policy, read_only=(*self.policy.read_only, '/usr/bin/true')),
                                     driver=self.driver)
            argv, env = probe.command(temporary, ('/usr/bin/true',), {})
            result = subprocess.run(argv, env=env, capture_output=True, timeout=10)
        if result.returncode:
            raise IsolationUnavailable('Isolation preflight failed: ' + result.stderr.decode(errors='replace')[:1000])
