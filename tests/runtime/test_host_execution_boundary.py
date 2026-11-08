"""Real CPU processes exercise capability escapes, not argv string filters."""
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.adapters.execution import LabExecutionTools
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.access import ActionPermissions, SensitivePolicy, authorized_space
from rsisys.loops.decision_making.harness.tools import execute_operation
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.isolation import ExecutionPolicy, IsolatedProcessBackend, IsolationUnavailable, ProcessIsolation, system_runtime_paths
from rsisys.runtime.resource_account import ResourceAccount


class HostExecutionBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.policy = ExecutionPolicy(read_only=system_runtime_paths())
        try:
            self.backend = IsolatedProcessBackend(self.root / 'executor', policy=self.policy)
        except IsolationUnavailable as exc:
            self.skipTest(str(exc))
        self.lab = LocalProcessLab(self.backend, InMemoryArtifactStore(), timeout_seconds=10)
        self.addCleanup(self.lab.close)

    def execute(self, code):
        result = self.lab.execute(OperationCall('lab_start', {'key': str(len(self.lab._jobs)),
            'argv': [sys.executable, '-c', code]}))
        return self.backend.wait(self.lab._jobs[result.data['job_id']][0])

    def test_allowed_program_cannot_write_or_read_sensitive_host_files(self):
        target = self.root / 'operator-policy.json'
        target.write_text('host-only')
        tools = LabExecutionTools(self.lab)
        policy = SensitivePolicy('host-sensitive', '1', ({'name': 'lab_write_file'},))
        space = authorized_space(ModularDecisionActionSpace(tools.definitions,
            permissions=ActionPermissions(check=policy)))
        blocked = execute_operation(tools, space, OperationCall('lab_write_file', {'path': 'x', 'content': 'x'}))
        self.assertIsInstance(blocked.error, PermissionError)
        code = f'''from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
p=Path({str(target)!r})
try: p.read_text()
except OSError: pass
else: raise AssertionError('host file was readable')
# Linux synthesizes /tmp in the child namespace. Writing the same pathname
# there can create a private shadow; the host assertion below checks that the
# original was not changed. Seatbelt may reject the write altogether.
try: p.write_text('escaped')
except OSError: pass
Path('research.txt').write_text('allowed')
print('confined')
'''
        submitted = execute_operation(tools, space, OperationCall('lab_start', {
            'key': 'allowed', 'argv': [sys.executable, '-c', code]}))
        self.assertIsNone(submitted.error)
        state = self.backend.wait(self.lab._jobs[submitted.outcome.data['job_id']][0])
        self.assertEqual(state.exit_code, 0, state.stderr)
        self.assertEqual(target.read_text(), 'host-only')
        self.assertEqual((self.lab.workspace / 'research.txt').read_text(), 'allowed')

    def test_symlink_to_operator_directory_does_not_expand_capabilities(self):
        target = self.root / 'secrets'
        target.mkdir()
        (target / 'key').write_text('not-a-real-key')
        (self.lab.workspace / 'escape').symlink_to(target)
        state = self.execute("from pathlib import Path; Path('escape/key').read_text()")
        self.assertNotEqual(state.exit_code, 0)

    def test_interpreter_parent_alias_uses_mounted_runtime_and_preserves_venv(self):
        alias = self.root / 'unmounted-interpreter-parent'
        alias.symlink_to(Path(sys.executable).parent.resolve(), target_is_directory=True)
        command, env = self.backend.isolation.command(self.lab.workspace,
            (str(alias / Path(sys.executable).name), '-c', 'import sys; print(sys.prefix)'), {})
        result = subprocess.run(command, env=env, cwd=self.lab.workspace,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(Path(result.stdout.decode().strip()).resolve(), Path(sys.prefix).resolve())

    @unittest.skipUnless(sys.platform == 'linux', 'Linux private tmp namespace')
    def test_multiprocessing_socket_with_long_workspace_and_inference_scratch(self):
        workspace = self.root / ('workspace-' + 'x' * 120)
        scratch = self.root / ('scratch-' + 'x' * 120)
        workspace.mkdir()
        scratch.mkdir()
        code = '''import tempfile
from multiprocessing.connection import Listener
assert tempfile.gettempdir() == '/tmp'
with Listener() as listener:
    assert listener.address.startswith('/tmp/')
print('socket bound')
'''
        for writable in (True, False):
            with self.subTest(writable=writable):
                argv, env = self.backend.isolation.command(workspace,
                    (sys.executable, '-c', code), {}, writable=writable,
                    scratch=None if writable else scratch)
                result = subprocess.run(argv, env=env, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(result.stdout.strip(), b'socket bound')
        self.assertEqual(list(workspace.iterdir()), [])
        self.assertEqual(list(scratch.iterdir()), [])

    def test_raw_network_client_cannot_bypass_host_model_account(self):
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        listener.settimeout(.1)
        state = self.execute(f"import socket; socket.create_connection({listener.getsockname()!r},timeout=1)")
        self.assertNotEqual(state.exit_code, 0)
        with self.assertRaises(TimeoutError):
            listener.accept()

    def test_child_cannot_read_parent_through_proc_or_open_unassigned_device(self):
        code = f'''from pathlib import Path
for name in ['/proc/{__import__('os').getpid()}/environ','/dev/nvidia7']:
    try: Path(name).read_bytes()
    except OSError: pass
    else: raise AssertionError('unassigned host capability accessible')
'''
        state = self.execute(code)
        self.assertEqual(state.exit_code, 0, state.stderr)

    def test_missing_isolation_driver_fails_before_any_research_exec(self):
        with patch('rsisys.runtime.isolation.shutil.which', return_value=None):
            with self.assertRaises(IsolationUnavailable):
                IsolatedProcessBackend(self.root / 'never-created', policy=self.policy)
        self.assertFalse((self.root / 'never-created').exists())

    def test_runtime_mount_cannot_expose_executor_recovery_state(self):
        with self.assertRaisesRegex(ValueError, 'recovery'):
            IsolatedProcessBackend(self.root / 'other', policy=ExecutionPolicy(read_only=(str(self.root),)))

    def test_offline_occupancy_uses_separate_durable_account(self):
        account = ResourceAccount(self.root / 'resources.json', allocation={'device_nodes': []}, max_seconds=20)
        self.backend.resource_account = account
        state = self.execute("print('offline inference fixture')")
        self.assertEqual(state.exit_code, 0)
        jobs = json.loads(account.path.read_text())['jobs']
        self.assertEqual(len(jobs), 1)
        row = next(iter(jobs.values()))
        self.assertEqual(row['state'], 'succeeded')
        self.assertLess(row['elapsed_seconds'], 10)
        self.assertFalse((self.root / 'model_calls.json').exists())
        account.reserve('unknown-effect', 10)
        with self.assertRaises(PermissionError):
            account.reserve('over-budget', 11)


if __name__ == '__main__':
    unittest.main()
