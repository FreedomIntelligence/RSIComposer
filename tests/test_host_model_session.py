from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from rsisys.loops.contracts.model_service import ModelRef, GenerationRequest, GenerationResponse, TokenUsage
from rsisys.runtime.accounting import AccountedModelBackend, HostAdmissionStop
from rsisys.runtime.isolation import ExecutionPolicy, IsolatedProcessBackend, IsolationUnavailable, system_runtime_paths
from rsisys.runtime.model_session import HostModelSession, ModelSessionBroker
from rsisys.runtime.sandbox import SandboxSpec, ExecRequest


MODEL = ModelRef('host', 'synthetic-provider', 'v1')
REQUEST = GenerationRequest('decision_maker', MODEL, 'CPU synthetic prompt')


class Transport:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse('fixture-response', request.model, True, TokenUsage(3, 2))


class HostModelSessionTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.transport = Transport()
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.return_value = [1, 2, 3]

    def backend(self, limit=2, *, resume=False):
        return AccountedModelBackend(self.transport, self.tokenizer, self.root / 'account',
                                     max_calls=limit, resume=resume)

    def test_pluggable_clients_share_reservations_and_cannot_reset_or_create_extra_budget(self):
        first = self.backend()
        second = self.backend(resume=True)
        with ThreadPoolExecutor(2) as pool:
            values = list(pool.map(lambda b: b.generate(REQUEST), (first, second)))
        self.assertEqual(len(values), 2)
        for backend in (first, second):
            with self.assertRaises(HostAdmissionStop):
                backend.generate(REQUEST)
        ledger = json.loads((first.root / 'model_calls.json').read_text())
        self.assertEqual(ledger, {'attempted': 2, 'limit': 2})
        requests = [json.loads(line) for line in (first.root / 'model_requests.jsonl').read_text().splitlines()]
        self.assertEqual({r['call'] for r in requests}, {1, 2})
        with self.assertRaisesRegex(ValueError, 'reset'):
            self.backend()
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.backend(3, resume=True)

    def test_failed_provider_call_keeps_reservation_and_recorded_usage_is_not_invented(self):
        backend = self.backend(1)
        self.transport.generate = Mock(side_effect=TimeoutError('unknown provider outcome'))
        with self.assertRaises(TimeoutError):
            backend.generate(REQUEST)
        resumed = self.backend(1, resume=True)
        with self.assertRaises(HostAdmissionStop):
            resumed.generate(REQUEST)
        self.assertFalse((backend.root / 'model_responses.jsonl').exists())
        self.assertIn('TimeoutError', (backend.root / 'model_errors.jsonl').read_text())

    def test_stop_during_token_counting_does_not_reserve_or_call_provider(self):
        backend = self.backend()
        def count(_):
            backend.stop_requested = True
            return 3
        backend.token_counter = count
        with self.assertRaises(HostAdmissionStop):
            backend.generate(REQUEST)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(self.transport.requests, [])

    def test_session_does_not_allow_a_different_unregistered_model(self):
        backend = self.backend()
        session = HostModelSession({'host': backend}, (MODEL,))
        with self.assertRaises(PermissionError):
            session.generate(replace(REQUEST, model=replace(MODEL, revision='other')))
        self.assertEqual(backend.calls, 0)

    def test_isolated_worker_queries_broker_and_every_attempt_uses_shared_account(self):
        backend = self.backend(1)
        broker = ModelSessionBroker(self.root / 'broker.sock', HostModelSession({'host': backend}, (MODEL,)))
        self.addCleanup(broker.close)
        try:
            worker = IsolatedProcessBackend(self.root / 'worker', policy=ExecutionPolicy(
                read_only=system_runtime_paths(), sockets=(str(broker.path),)))
        except IsolationUnavailable as exc:
            self.skipTest(str(exc))
        self.addCleanup(worker.close_all)
        sandbox = worker.create(SandboxSpec('local', 10))
        code = f'''import socket,json
with socket.socket(socket.AF_UNIX) as s:
    s.connect({str(broker.path)!r})
    s.sendall({(json.dumps(asdict(REQUEST)) + chr(10)).encode()!r})
    print(s.makefile('rb').readline().decode().strip())
'''
        for expected in ('fixture-response', 'HostAdmissionStop'):
            process = worker.exec(sandbox, ExecRequest((sys.executable, '-c', code), 10))
            state = worker.wait(process)
            self.assertEqual(state.exit_code, 0, state.stderr)
            self.assertIn(expected, state.stdout)
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(json.loads((backend.root / 'model_calls.json').read_text())['attempted'], 1)


if __name__ == '__main__':
    unittest.main()
