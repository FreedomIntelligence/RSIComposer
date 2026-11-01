#!/usr/bin/env python3
"""Exercise the real optional DSH runtime against a loopback-only model fixture.

From the RSISys directory, after ``pip install -e '.[dsh]'``::

    python examples/dsh_smoke.py

No model server, network credentials, Node installation, or model download is
needed. DSH really launches and persists sessions; only the model HTTP responses
and the demo Actor/backend are simulated. Every request must advertise no native
DSH tools. Temporary homes and the local HTTP server are removed on exit.

``--sdk-only`` checks session creation/run/restart/resume without assembling RSI.
SDK persistence is checked independently; this is not an RSI snapshot recovery
example. ``--compaction-only`` separately exercises native automatic compaction
at a 4096-token context window using deterministic checkpoint responses and
synthetic token-usage estimates; it does not measure summary quality.

The fixture supports OpenAI ``/v1/chat/completions`` and Messages ``/v1/messages``.
The rc2 deepseek-official adapter actually uses the latter; a base_url override
does not switch its wire protocol to OpenAI chat completions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL = 'rsisys-smoke-fixture'
API_KEY = 'local-smoke-not-a-secret'
API_KEY_ENV = 'RSISYS_DSH_SMOKE_API_KEY'
MARKER = 'rsisys-dsh-persisted-marker'


def require(condition, message):
    """Keep smoke checks active even when Python runs with -O."""
    if not condition:
        raise AssertionError(message)


class LocalModelFixture:
    """A finite deterministic chat-completions/Messages SSE conversation."""

    def __init__(self, responses=(), *, responder=None, max_requests=32):
        self.responses = list(responses)
        self.responder = responder
        self.max_requests = max_requests
        self.requests = []
        self.compaction_requests = []
        self.paths = []
        self.errors = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    require(self.path in ('/v1/chat/completions', '/v1/messages'),
                            f'Unexpected path: {self.path}')
                    require(self.headers.get('Authorization') == f'Bearer {API_KEY}'
                            or self.headers.get('x-api-key') == API_KEY,
                            'Fixture did not receive its dummy credential')
                    payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    require(payload.get('model') == MODEL, 'Unexpected model route')
                    require(payload.get('stream') is True, 'Expected a streaming request')
                    require(not payload.get('tools'),
                            f'Native DSH tools escaped the restricted patch: {payload.get("tools")}')
                    require(not payload.get('functions'), 'Legacy native functions must also be absent')
                    require(isinstance(payload.get('messages'), list), 'Expected chat messages')
                    index = len(fixture.requests)
                    fixture.requests.append(payload)
                    fixture.paths.append(self.path)
                    compacting = self.headers.get('x-deepseek-harness-compact') == '1'
                    fixture.compaction_requests.append(compacting)
                    require(index < fixture.max_requests, 'Fixture request budget exceeded')
                    if fixture.responder is None:
                        require(index < len(fixture.responses), 'Unexpected extra model request')
                        response = fixture.responses[index]
                        input_tokens = 100
                    else:
                        response = fixture.responder(payload, compacting)
                        # Synthetic usage for this deterministic server, not a model tokenizer.
                        input_tokens = max(1, len(json.dumps(payload)) // 4)
                    if isinstance(response, dict):
                        response = json.dumps(response)
                    body = fixture.stream(self.path, response, index, input_tokens=input_tokens)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    self.wfile.flush()
                except Exception as exc:
                    fixture.errors.append(f'{type(exc).__name__}: {exc}')
                    self.send_error(400, 'Local fixture rejected request')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                       name='rsisys-smoke-http')

    @staticmethod
    def stream(path, response, index, *, input_tokens=100):
        if path == '/v1/messages':
            events = [
                {'type': 'message_start', 'message': {'id': f'msg-smoke-{index}',
                 'type': 'message', 'role': 'assistant', 'model': MODEL, 'content': [],
                 'stop_reason': None, 'stop_sequence': None,
                 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}},
                {'type': 'content_block_start', 'index': 0,
                 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0,
                 'delta': {'type': 'text_delta', 'text': response}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                 'usage': {'output_tokens': 30}},
                {'type': 'message_stop'},
            ]
            return ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'
                           for event in events).encode()
        common = {'id': f'chatcmpl-smoke-{index}', 'object': 'chat.completion.chunk',
                  'created': 1, 'model': MODEL}
        chunks = [
            {**common, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': response},
                                   'finish_reason': None}]},
            {**common, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
             'usage': {'prompt_tokens': input_tokens, 'completion_tokens': 30,
                       'total_tokens': input_tokens + 30}},
        ]
        return (''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks)
                + 'data: [DONE]\n\n').encode()

    @property
    def base_url(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def verify(self):
        require(not self.errors, f'HTTP fixture errors: {self.errors}')
        if self.responder is None:
            require(len(self.requests) == len(self.responses),
                    f'Expected {len(self.responses)} model requests, got {len(self.requests)}')


@contextmanager
def fixture_credential():
    previous = os.environ.get(API_KEY_ENV)
    os.environ[API_KEY_ENV] = API_KEY
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(API_KEY_ENV, None)
        else:
            os.environ[API_KEY_ENV] = previous


def sdk_smoke(root):
    from deepseek_harness import DeepSeekHarness
    from rsisys.loops.decision_making.harness import dsh

    patch = Path(dsh.__file__).with_name('dsh.patch.yml')
    require(patch.is_file(), 'The packaged restricted DSH patch is missing')
    workspace = root / 'sdk-workspace'
    workspace.mkdir()
    session_id = 'rsisys-smoke-sdk'
    responses = ['persisted first response', 'restored second response']
    with LocalModelFixture(responses) as fixture:
        options = dict(dsh_home=str(root / 'sdk-home'), cwd=str(workspace),
                       profile='sdk-minimal', patches=(str(patch),), model=MODEL,
                       base_url=fixture.base_url, api_key=API_KEY,
                       env={'DSH_CONTEXT_WINDOW': '32768'}, max_tokens=2048,
                       initialize_timeout_seconds=15, request_timeout_seconds=15,
                       shutdown_timeout_seconds=2)
        with DeepSeekHarness(**options) as runtime:
            session = runtime.start_session(session_id)
            require(session.id == session_id, 'Session id changed during creation')
            first = session.run(f'Remember this exact marker: {MARKER}')
            require(first.finish_reason == 'completed',
                    f'First SDK turn failed: {first.finish_reason}; fixture={fixture.errors}; '
                    f'last_event={first.events[-1:]}')
            require(first.final_response == responses[0], 'First response did not pass through DSH')
        with DeepSeekHarness(**options) as runtime:
            before = len(fixture.requests)
            restored = runtime.resume_session(session_id)
            require(restored.id == session_id, 'Resume changed the session id')
            require(len(fixture.requests) == before, 'Resume unexpectedly requested model output')
            second = restored.run('Continue the saved conversation.')
            require(second.finish_reason == 'completed', f'Resumed SDK turn failed: {second.finish_reason}')
            require(second.final_response == responses[1], 'Resumed response did not pass through DSH')
        fixture.verify()
        history = fixture.requests[1]['messages']
        require(any(message.get('role') == 'user' and MARKER in json.dumps(message.get('content'))
                    for message in history), 'Resumed HTTP request lost the first user turn')
        require(any(message.get('role') == 'assistant'
                    and responses[0] in json.dumps(message.get('content')) for message in history),
                'Resumed HTTP request lost the first assistant turn')
        return {'session_id': session_id, 'runtime_restart_resume': True,
                'restored_user_and_assistant_history': True, 'model_requests': len(fixture.requests),
                'native_tools': [], 'http_paths': sorted(set(fixture.paths))}


def integration_smoke(root, timeout):
    from rsisys.config import DemoConfig, build_demo
    from rsisys.loops.decision_making.harness.dsh import (
        DSHConfig, DeepSeekDecisionHarness, dsh_decision_factory,
    )

    responses = [
        {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': MARKER}},
        {'kind': 'explore', 'strategy': 'Exercise the demo Actor with the prepared input',
         'mode': 'execute', 'arguments': {}},
        {'kind': 'finish', 'reason': 'The deterministic DSH integration smoke is complete'},
    ]
    workspace = root / 'rsi-workspace'
    workspace.mkdir()
    with LocalModelFixture(responses) as fixture, fixture_credential():
        config = DSHConfig(home=root / 'rsi-home', workspace=workspace, model=MODEL,
                           session_id='rsisys-smoke-integration', base_url=fixture.base_url,
                           api_key_env=API_KEY_ENV)
        demo = DemoConfig.load(ROOT / 'configs/demo.json')
        runner, store = build_demo(demo, decision_factory=dsh_decision_factory(config))
        with runner:
            harness = runner.decision_making.harness
            require(isinstance(harness, DeepSeekDecisionHarness), 'Factory did not install DSH')
            deadline = time.monotonic() + timeout
            steps = []
            while time.monotonic() < deadline:
                require(not fixture.errors, f'HTTP fixture errors: {fixture.errors}')
                step = runner.step('Exercise real DSH decisions using only deterministic local fixtures')
                steps.append(step.kind)
                if step.kind == 'finished':
                    break
                # The DSH policy is asynchronous: waiting is a scheduling yield.
                time.sleep(0.01)
            else:
                raise TimeoutError(f'DSH integration did not finish in {timeout}s; last steps: {steps[-10:]}')
            operations = harness.memory.history(kind='operation')
            prepared = [entry for entry in operations
                        if entry.details.get('operation', {}).get('name') == 'prepare_data']
            require(len(prepared) == 1, 'Expected exactly one authorized prepare_data operation')
            require(prepared[0].details['result']['artifacts'], 'prepare_data produced no artifact')
            require(harness.state.issued == 1, 'Expected one parsed and dispatched directive')
            require(harness.state.completed == 1, 'The demo exploration did not complete')
            require(harness.state.last_directive.input_refs, 'Prepared artifact was not handed to the Actor')
            require(harness.state.finished, 'The final JSON finish action was not applied')
            require('waiting' in steps, 'Expected asynchronous DSH scheduling to yield')
            result = {'authorized_operations': len(prepared), 'directives_completed': harness.state.completed,
                      'finished': True, 'async_wait_observed': True, 'experience_records': len(store.records()),
                      'model_requests': len(fixture.requests), 'native_tools': [],
                      'http_paths': sorted(set(fixture.paths)),
                      'actor_and_execution_simulated': True}
        fixture.verify()
        return result


@contextmanager
def runtime_deadline(runtime, timeout):
    """Session.run has no notification deadline in rc2; close it if it stalls."""
    expired = threading.Event()

    def interrupt():
        expired.set()
        runtime.close()

    timer = threading.Timer(timeout, interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join(timeout=6)
        if expired.is_set():
            raise TimeoutError(f'DSH compaction smoke exceeded its {timeout}s runtime deadline')


def compaction_smoke(root, timeout):
    """Fill actual session history until the unmodified shipped patch compacts it."""
    from deepseek_harness import DeepSeekHarness
    from rsisys.loops.decision_making.harness import dsh

    checkpoint_marker = 'rsisys-native-checkpoint-verified'
    sections = ('Primary Request and Intent', 'Key Technical Concepts', 'Files and Code',
                'Errors and Fixes', 'Pending Jobs', 'Current Work', 'Next Step', 'Critical Context')
    summary = '\n\n'.join(f'## {name}\n- '
                           + (checkpoint_marker if name == 'Critical Context' else '(none)')
                           for name in sections)

    def respond(payload, compacting):
        require(0 < payload.get('max_tokens', 0) <= 1024, 'Output cap exceeded 1024 tokens')
        if compacting:
            require('compaction engine' in json.dumps(payload['messages'][-1]),
                    'Expected the native checkpoint instruction on a summary request')
            return summary
        return {'kind': 'wait', 'reason': 'Deterministic history pressure fixture'}

    workspace = root / 'compaction-workspace'
    workspace.mkdir()
    with LocalModelFixture(responder=respond) as fixture:
        options = dict(dsh_home=str(root / 'compaction-home'), cwd=str(workspace),
                       profile='sdk-minimal', patches=(str(Path(dsh.__file__).with_name('dsh.patch.yml')),),
                       model=MODEL, base_url=fixture.base_url, api_key=API_KEY,
                       env={'DSH_CONTEXT_WINDOW': '4096'}, max_tokens=512,
                       initialize_timeout_seconds=15, request_timeout_seconds=15,
                       shutdown_timeout_seconds=2)
        events = []
        with DeepSeekHarness(**options) as runtime, runtime_deadline(runtime, timeout):
            session = runtime.start_session('rsisys-smoke-compaction')
            # Distinct text grows real persisted history; no /compact command or
            # pressure-config override is sent. The fixture alone invents usage.
            for turn in range(12):
                text = f'History turn {turn}:\n' + '\n'.join(
                    f'Observation {turn}:{row}: deterministic local evidence retained for context pressure.'
                    for row in range(8))
                result = session.run(text)
                require(result.finish_reason == 'completed',
                        f'Compaction turn failed: {result.finish_reason}; fixture={fixture.errors}; '
                        f'last_event={result.events[-1:]}')
                events.extend(result.events)
                if any(event['type'] == 'compaction/summary' for event in events):
                    # Automatic compaction precedes the ordinary generation in
                    # this same turn: its final JSON must survive the checkpoint.
                    require(json.loads(result.final_response)['kind'] == 'wait',
                            'Expected structured decision output after compaction')
                    break
            else:
                raise AssertionError('Native automatic compaction did not occur within 12 history turns')
        fixture.verify()
        starts = [event for event in events if event['type'] == 'compaction/start']
        summaries = [event for event in events if event['type'] == 'compaction/summary']
        ends = [event for event in events if event['type'] == 'compaction/end']
        require(starts and summaries and ends, 'Missing native compaction transaction events')
        require(all(not event['data'].get('sourceCommandId') for event in starts),
                'Compaction should be automatic, without a manual source command')
        require(all(not event['data'].get('error') for event in ends),
                f'A native compaction attempt failed: {[event["data"] for event in ends]}')
        require(all(event['data'].get('shadowedSeqs') for event in summaries),
                'Compaction did not replace any persisted conversation events')
        require(all(event['data'].get('llmStreamCall') is True for event in summaries),
                'Compaction did not use the native model-backed summarizer')
        require(any(event['type'] == 'request/context' and event['data']['contextWindow'] == 4096
                    for event in events), 'Runtime did not accept the 4096-token context window')
        require(any(fixture.compaction_requests), 'The fixture never received a marked summary request')
        final_request = json.dumps(fixture.requests[-1]['messages'])
        require(not fixture.compaction_requests[-1], 'Final request was still a summary request')
        require('<compacted-summary>' in final_request and checkpoint_marker in final_request,
                'Follow-up generation did not consume the native checkpoint')
        return {'automatic_compaction': True, 'context_window': 4096, 'generation_max_tokens': 512,
                'summary_max_tokens': 1024, 'compaction_summaries': len(summaries),
                'summary_requests': sum(fixture.compaction_requests),
                'checkpoint_used_by_followup': True, 'continued_structured_output': True,
                'model_requests': len(fixture.requests), 'history_turns': turn + 1,
                'usage_is_synthetic': True, 'native_tools': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--sdk-only', action='store_true', help='Only check the SDK session lifecycle')
    mode.add_argument('--compaction-only', action='store_true', help='Only check native automatic compaction')
    parser.add_argument('--timeout', type=float, default=45, help='Integration/compaction deadline in seconds')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    try:
        sdk_version = version('deepseek-harness-sdk')
    except ModuleNotFoundError:
        parser.exit(2, 'Install the optional SDK first: python -m pip install -e ".[dsh]"\n')
    with tempfile.TemporaryDirectory(prefix='rsisys-dsh-smoke-') as directory:
        root = Path(directory)
        report = {'sdk_version': sdk_version, 'real_dsh_runtime': True, 'real_model_used': False}
        if args.compaction_only:
            report['compaction'] = compaction_smoke(root, args.timeout)
        else:
            report['sdk'] = sdk_smoke(root)
            if not args.sdk_only:
                report['integration'] = integration_smoke(root, args.timeout)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
