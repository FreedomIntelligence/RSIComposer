"""Node-local Unix broker relay outside an isolated candidate's namespace.

Only this trusted wrapper receives a host HTTP capability. The candidate gets
one mounted Unix socket and its original, explicitly selected environment.
"""
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading

from .remote_services import HOST_TOKEN_ENV, HOST_URL_ENV, post_worker
from .workers import worker_environment


class SocketRelay:
    def __init__(self, route, *, environment=None):
        environment = os.environ if environment is None else environment
        if not isinstance(route, str) or not route.startswith('/sockets/') or route.count('/') != 2:
            raise ValueError('Socket relay requires a host-issued socket route')
        self.route, self.base_url, self.token = route, environment[HOST_URL_ENV], environment[HOST_TOKEN_ENV]
        self.temporary = tempfile.TemporaryDirectory(prefix='rsisys-relay-')
        self.path = str(Path(self.temporary.name) / 'broker.sock')
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    self.request.settimeout(300)
                    data = self.rfile.readline(1_000_001)
                    if len(data) > 1_000_000 or not data.endswith(b'\n'):
                        raise ValueError('Invalid socket relay request size')
                    request = json.loads(data)
                    response = post_worker(owner.base_url + owner.route, owner.token,
                                           {'request': request}, timeout=300)['response']
                except Exception as exc:
                    # Credentials and host paths stay outside candidate diagnostics.
                    response = {'error': 'Host inference relay failed: ' + type(exc).__name__}
                try:
                    self.wfile.write(json.dumps(response, allow_nan=False).encode() + b'\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
            block_on_close = False

        try:
            self.server = Server(self.path, Handler)
        except BaseException:
            self.temporary.cleanup()
            raise
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.05), daemon=True)
        self.thread.start()

    def command(self, command, source, environment):
        """Replace only the bind source; the sandbox's destination is unchanged."""
        command, environment = list(command), dict(environment)
        mounted = False
        for index in range(len(command) - 2):
            if command[index] in ('--bind', '--ro-bind') and command[index + 1] == source:
                command[index + 1] = self.path
                mounted = True
        if not mounted:
            # Non-sandboxed trusted transports can pass the socket as a literal
            # argument. This also permits a standard-library CPU relay probe.
            if source not in command:
                raise ValueError('Command does not use the registered Unix socket')
            command = [self.path if value == source else value for value in command]
            environment = {key: self.path if value == source else value for key, value in environment.items()}
        if any(key.startswith(('RSISYS_', 'OPENAI_')) for key in environment):
            raise ValueError('Isolated socket clients must not receive host capabilities')
        return command, environment

    def run(self, command, source, environment):
        command, environment = self.command(command, source, environment)
        return subprocess.call(command, env=environment, stdin=subprocess.DEVNULL)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def socket_relay_command(command, environment, source, route, payload_path):
    """Prepare trusted relay execution; the persisted payload contains no token."""
    payload = {'command': list(command), 'environment': dict(environment), 'source': source, 'route': route}
    if any(key.startswith(('RSISYS_', 'OPENAI_')) for key in environment):
        raise ValueError('Candidate environment cannot contain host capabilities')
    payload_path = Path(payload_path).resolve()
    payload_path.write_text(json.dumps(payload))
    wrapper = (sys.executable, '-m', 'rsicomposer.runtime.execution.socket_relay', str(payload_path))
    return wrapper, worker_environment(**environment)


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    with SocketRelay(payload['route']) as relay:
        return relay.run(payload['command'], payload['source'], payload['environment'])


if __name__ == '__main__':
    raise SystemExit(main())
