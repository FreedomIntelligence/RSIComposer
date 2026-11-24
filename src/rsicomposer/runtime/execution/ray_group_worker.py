"""Trusted class instances sharing one leased, GPU-masked worker process."""
import fcntl
from multiprocessing.connection import Listener
import os
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ray import cloudpickle
    objects = {}
    listener = Listener(sys.argv[1], family='AF_UNIX', authkey=bytes.fromhex(sys.argv[2]))
    try:
        while True:
            connection = listener.accept()
            try:
                request = cloudpickle.loads(connection.recv_bytes())
                mode, key = request['mode'], request.get('key')
                if mode == 'create':
                    marker = Path(request['marker']).open('a+b')
                    fcntl.flock(marker, fcntl.LOCK_SH)
                    environment = request.get('environment', {})
                    previous = dict(os.environ)
                    for name in tuple(os.environ):
                        if name.startswith(('RSICOMPOSER_', 'OPENAI_')) and name not in environment:
                            os.environ.pop(name)
                    os.environ.update(environment)
                    try:
                        objects[key] = (request['cls'](*request['args'], **request['kwargs']), marker)
                    except BaseException:
                        marker.close()
                        raise
                    finally:
                        os.environ.clear()
                        os.environ.update(previous)
                    result = {'pid': os.getpid(), 'devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
                elif mode == 'call':
                    instance, _ = objects[key]
                    # Calls are serialized per process, so scope credentials cannot
                    # race between colocated objects. CUDA visibility stays fixed.
                    environment = request.get('environment', {})
                    previous = dict(os.environ)
                    for name in tuple(os.environ):
                        if name.startswith(('RSICOMPOSER_', 'OPENAI_')) and name not in environment:
                            os.environ.pop(name)
                    os.environ.update(environment)
                    try:
                        result = getattr(instance, request['method'])(*request['args'], **request['kwargs'])
                    finally:
                        os.environ.clear()
                        os.environ.update(previous)
                elif mode == 'destroy':
                    instance, marker = objects[key]
                    close = getattr(instance, 'close', None)
                    if callable(close):
                        close()
                    del objects[key]
                    Path(marker.name).touch()
                    marker.close()
                    result = None
                elif mode == 'close':
                    result = None
                else:
                    raise ValueError('Unknown worker group operation')
                connection.send_bytes(cloudpickle.dumps({'ok': True, 'result': result}))
                if mode == 'close':
                    break
            except BaseException as exc:
                connection.send_bytes(cloudpickle.dumps({'ok': False, 'error': exc}))
            finally:
                connection.close()
    finally:
        for instance, marker in objects.values():
            close = getattr(instance, 'close', None)
            try:
                if callable(close):
                    close()
            finally:
                Path(marker.name).touch()
                marker.close()
        listener.close()


if __name__ == '__main__':
    main()
