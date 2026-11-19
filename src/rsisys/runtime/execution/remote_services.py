"""Host-issued worker capabilities for artifacts and observation over HTTP.

The address is transport metadata. Tokens select a component and a captured host
operation scope; request bodies cannot choose either. Evidence stays host-owned.
"""
import base64
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import threading
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from rsisys.loops.contracts.artifacts import ArtifactRef

HOST_URL_ENV = 'RSISYS_HOST_URL'
HOST_TOKEN_ENV = 'RSISYS_HOST_TOKEN'
LOG_URL_ENV = 'RSISYS_LOG_URL'
LOG_TOKEN_ENV = 'RSISYS_LOG_TOKEN'


def service_addresses(context):
    """The Ray execution path sets addresses; importing core never imports Ray."""
    options = getattr(context, '_network_host', {})
    return options.get('bind_host', '127.0.0.1'), options.get('advertise_host', '127.0.0.1')


class HostWorkerServices:
    def __init__(self, context, *, bind_host='127.0.0.1', advertise_host='127.0.0.1',
                 max_request_bytes=16 * 1024 * 1024):
        if (not isinstance(bind_host, str) or not bind_host or not isinstance(advertise_host, str)
                or not advertise_host or any(c in advertise_host for c in '/:@?#')):
            raise ValueError('Worker service requires bind and advertised host names')
        if type(max_request_bytes) is not int or max_request_bytes < 1:
            raise ValueError('Worker request size must be a positive integer')
        self.context = context
        self.authority = context._artifact_access
        self._tokens, self._lock, self._closed = {}, threading.RLock(), False
        self._sockets = {}
        self._issued_tokens = set()
        self.inbox = (Path(context.data_dir) if context.data_dir is not None
                      else Path(context.budget.root)) / 'log-inbox'
        self.inbox.mkdir(parents=True, exist_ok=True)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, status, payload):
                data = json.dumps(payload, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self.connection.settimeout(30)
                auth = self.headers.get('Authorization', '')
                with owner._lock:
                    capability = owner._tokens.get(auth[7:]) if auth.startswith('Bearer ') else None
                if capability is None:
                    self.reply(401, {'error': 'Invalid worker capability'})
                    return
                try:
                    if self.headers.get('Transfer-Encoding') is not None:
                        raise ValueError('Transfer encoding is not supported')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= max_request_bytes:
                        self.reply(413, {'error': 'Invalid worker request size'})
                        return
                    raw = json.loads(self.rfile.read(size))
                    with owner.authority.use_captured_scope(capability):
                        result = owner._request(self.path, raw, capability[0], capability=capability)
                except PermissionError:
                    self.reply(403, {'error': 'Worker scope or artifact access denied'})
                except (ValueError, KeyError, TypeError):
                    self.reply(400, {'error': 'Invalid worker request'})
                except Exception:
                    self.reply(500, {'error': 'Host worker service failed'})
                else:
                    self.reply(200, result)

        class Server(ThreadingHTTPServer):
            daemon_threads = False
            slots = threading.BoundedSemaphore(32)

            def process_request(self, request, address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, address)
                except BaseException:
                    self.slots.release()
                    raise

            def process_request_thread(self, request, address):
                try:
                    super().process_request_thread(request, address)
                finally:
                    self.slots.release()

        self.server = Server((bind_host, 0), Handler)
        self.base_url = f'http://{advertise_host}:{self.server.server_port}'
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.05), daemon=True)
        self.thread.start()

    @property
    def secret_values(self):
        """Issued credentials stay restricted even after access is revoked."""
        with self._lock:
            return tuple(self._issued_tokens)

    def environment(self, component, supplied=None):
        model_access = getattr(self.context, '_model_access', None)
        delegated = model_access.nested_environment(supplied) if model_access is not None else {}
        if component is None:
            with self._lock:
                if self._closed:
                    raise RuntimeError('Worker services are closed')
            return delegated
        if not isinstance(component, str) or not component:
            raise ValueError('Worker capability requires a component name')
        capability = self.authority.capture_scope(component)
        with self._lock:
            if self._closed:
                raise RuntimeError('Worker services are closed')
            self._tokens = {key: value for key, value in self._tokens.items()
                            if self.authority.captured_scope_active(value)}
            token = next((key for key, value in self._tokens.items() if value == capability), None)
            if token is None:
                token = secrets.token_urlsafe(32)
                self._issued_tokens.add(token)
                self._tokens[token] = capability
        environment = dict(delegated, **{HOST_URL_ENV: self.base_url, HOST_TOKEN_ENV: token,
                       LOG_URL_ENV: self.base_url + '/log', LOG_TOKEN_ENV: token})
        if model_access is not None:
            environment['RSISYS_MODEL_BASE_URL'] = model_access.server.base_url
            environment.update(model_access.server.worker_environment(component, supplied))
        return environment

    def add_socket(self, component, path):
        """Register a host-selected broker endpoint for the current operation."""
        capability = self.authority.capture_scope(component)
        if capability[1] is None or not self.authority.captured_scope_active(capability):
            raise PermissionError('Socket relay requires an active host operation scope')
        path = Path(path).resolve(strict=True)
        identity = path.stat()
        if not stat.S_ISSOCK(identity.st_mode):
            raise ValueError('Socket relay requires a Unix broker socket')
        with self._lock:
            if self._closed:
                raise RuntimeError('Worker services are closed')
            self._sockets = {key: value for key, value in self._sockets.items()
                             if self.authority.captured_scope_active(value['capability'])}
            route = '/sockets/' + secrets.token_hex(16)
            self._sockets[route] = dict(path=str(path), inode=(identity.st_dev, identity.st_ino),
                                       capability=capability)
        return route

    def _socket_request(self, route, raw, component, capability):
        with self._lock:
            target = self._sockets.get(route)
        if (target is None or capability != target['capability'] or component != capability[0]
                or not self.authority.captured_scope_active(target['capability'])):
            raise PermissionError('Socket relay is not available to this host operation')
        if set(raw) != {'request'}:
            raise ValueError('Socket relay accepts a single broker request')
        identity = Path(target['path']).stat()
        if (not stat.S_ISSOCK(identity.st_mode)
                or (identity.st_dev, identity.st_ino) != target['inode']):
            raise PermissionError('The registered host broker socket changed')
        body = json.dumps(raw['request'], allow_nan=False).encode() + b'\n'
        if len(body) > 1_000_000:
            raise ValueError('Socket relay request is too large')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(300)
            client.connect(target['path'])
            client.sendall(body)
            with client.makefile('rb') as stream:
                response = stream.readline(1_000_001)
        if len(response) > 1_000_000 or not response.endswith(b'\n'):
            raise ValueError('Host broker returned an invalid response')
        return {'response': json.loads(response)}

    def _request(self, route, raw, component, *, capability=None):
        if not isinstance(raw, dict):
            raise ValueError('Worker request must be an object')
        if route.startswith('/sockets/'):
            return self._socket_request(route, raw, component, capability)
        artifacts = self.authority.component(component)
        if route == '/artifacts/read' and set(raw) == {'ref'}:
            return {'content': base64.b64encode(artifacts.read(ArtifactRef(**raw['ref']))).decode()}
        if route == '/artifacts/put' and set(raw) == {'content', 'kind'}:
            return asdict(artifacts.put(base64.b64decode(raw['content'], validate=True), kind=raw['kind']))
        if route == '/artifacts/refs' and not raw:
            return [asdict(ref) for ref in artifacts.refs()]
        if route == '/log' and set(raw) == {'values', 'step', 'series'}:
            if self.authority.current(component)['sealed']:
                return {'accepted': True}
            from rsisys.runtime.observation.client import split
            metrics, notes = split(raw['values'])
            step, series = raw['step'], raw['series']
            if step is not None and (type(step) is not int or not 0 <= step <= 2**53):
                raise ValueError('Invalid log step')
            if series is not None and (not isinstance(series, str) or not 0 < len(series) <= 100):
                raise ValueError('Invalid log series')
            record = dict(metrics=metrics, notes=notes, step=step, series=series,
                          time=time.time(), pid='remote-' + component)
            with self._lock, (self.inbox / 'remote.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            return {'accepted': True}
        raise ValueError('Unknown worker route or request fields')

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._tokens.clear()
            self._sockets.clear()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def post_worker(url, token, payload, *, timeout=30):
    """Send a bounded worker message; service errors remain visible to the caller."""
    request = Request(url, data=json.dumps(payload, allow_nan=False).encode(),
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}, method='POST')
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    # These are cluster-local capability routes, not provider requests. Never
    # forward bearer tokens through an inherited proxy or an HTTP redirect.
    with build_opener(ProxyHandler({}), NoRedirect).open(request, timeout=timeout) as response:
        return json.loads(response.read())


class RemoteArtifacts:
    """A worker's scoped artifact client; never dereference a supplied URI directly."""
    def __init__(self, base_url, token):
        self.base_url, self._token = base_url.rstrip('/'), token

    @classmethod
    def from_environment(cls):
        return cls(os.environ[HOST_URL_ENV], os.environ[HOST_TOKEN_ENV])

    def read(self, ref):
        result = post_worker(self.base_url + '/artifacts/read', self._token, {'ref': asdict(ref)})
        return base64.b64decode(result['content'], validate=True)

    def put(self, content, *, kind):
        if not isinstance(content, bytes):
            raise TypeError('artifact content must be bytes')
        result = post_worker(self.base_url + '/artifacts/put', self._token,
            {'content': base64.b64encode(content).decode(), 'kind': kind})
        return ArtifactRef(**result)

    def put_file(self, path, *, kind):
        return self.put(Path(path).read_bytes(), kind=kind)

    def refs(self):
        return tuple(ArtifactRef(**ref) for ref in
            post_worker(self.base_url + '/artifacts/refs', self._token, {}))
