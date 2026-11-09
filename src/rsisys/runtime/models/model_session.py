"""A credential-free Unix broker for isolated optimizers and Lab programs."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import socketserver
import threading

from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.models.accounting import AccountedModelBackend


class HostModelSession:
    def __init__(self, backends, models):
        self.backends, self.models = dict(backends), tuple(models)
        if not self.models or any(not isinstance(b, AccountedModelBackend) for b in self.backends.values()):
            raise TypeError('Model sessions require host-accounted backends and explicit model bindings')
        if any(model.backend_id not in self.backends for model in self.models):
            raise ValueError('Unknown model backend in session')

    def generate(self, request):
        if request.model not in self.models:
            raise PermissionError('Model is not registered in this host session')
        return self.backends[request.model.backend_id].generate(request)

    def request(self, raw):
        raw = dict(raw)
        raw['model'] = ModelRef(**raw['model'])
        return asdict(self.generate(GenerationRequest(**raw)))


class ModelSessionBroker:
    def __init__(self, path, session, *, max_request_bytes=2 * 1024 * 1024, max_connections=4):
        self.path, self.session = Path(path), session
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # A stale address may still have a live owner; never unlink it.
            raise FileExistsError('Model broker endpoint already exists')
        broker = self
        self._closed = False
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(30)
                try:
                    data = self.rfile.readline(max_request_bytes + 1)
                    if len(data) > max_request_bytes or not data.endswith(b'\n'):
                        raise ValueError('Invalid model broker frame')
                    result = {'response': broker.session.request(json.loads(data))}
                except Exception as exc:
                    result = {'error': type(exc).__name__, 'message': str(exc)[:1000]}
                self.wfile.write(json.dumps(result, allow_nan=False).encode() + b'\n')
        class Server(socketserver.ThreadingUnixStreamServer):
            slots = threading.BoundedSemaphore(max_connections)

            def process_request(self, request, client_address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self.slots.release()
                    raise

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self.slots.release()
        self.server = Server(str(self.path), Handler)
        self.server.daemon_threads = False
        os.chmod(self.path, 0o600)
        info = self.path.stat()
        self._identity = (info.st_dev, info.st_ino)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        if self._closed:
            return
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        try:
            info = self.path.stat()
        except FileNotFoundError:
            pass
        else:
            if (info.st_dev, info.st_ino) == self._identity:
                self.path.unlink()
        self._closed = True
