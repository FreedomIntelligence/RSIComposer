"""Worker-side transport, semantic identity and checkpoint replay wiring."""
from copy import deepcopy
import json
import re
import sys

from rsisys.runtime.execution.replay import ReplayLog, request_digest
from rsisys.runtime.execution.research_bridge import emit_event
from rsisys.runtime.storage.files import file_hash


class UpstreamWorker:
    """Shared boundary for dependency-isolated upstream agent processes.

    Runtime installations and interpreter paths are launch inputs. Task contents
    remain part of the research identity even after their source path is removed
    from durable state. Only checkpoint-committed exchanges are replayed.
    """
    _LOCATION_FIELDS = frozenset({'root', 'directory', 'resume', 'upstream_root',
                                  'task_file', 'python', 'worker_python'})

    def send(self, event):
        emit_event(event)

    def receive(self):
        line = sys.stdin.readline()
        if not line:
            raise EOFError('Host closed the upstream callback channel')
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('Upstream callback receipt must be a JSON object')
        return value

    @classmethod
    def semantic_config(cls, config):
        if not isinstance(config, dict):
            raise TypeError('Upstream worker configuration must be a mapping')
        semantic = {key: deepcopy(value) for key, value in config.items() if key not in cls._LOCATION_FIELDS}
        expected = semantic.get('task_sha256')
        if expected is not None and (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('Invalid upstream task content identity')
        if 'task_file' in config:
            digest = file_hash(config['task_file'])
            if expected is not None and expected != digest:
                raise ValueError('Upstream task contents changed after admission')
            semantic['task_sha256'] = digest
        return semantic

    @classmethod
    def identity(cls, config, *, protocol):
        return request_digest({'protocol': protocol, 'config': cls.semantic_config(config)})

    @staticmethod
    def open_replay(directory, base, point=None):
        if point is not None:
            return ReplayLog.resume(directory, base, point)
        return ReplayLog(directory, base)
