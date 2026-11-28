"""Immutable Controller versions stored with the explicit registered JSON codec."""
import json

from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.loops.contracts.controller_version import ControllerVersion
from rsicomposer.runtime.storage.codec import decode, encode
from rsicomposer.runtime.storage.files import json_bytes


def _controller_seed(version, history):
    from rsicomposer.harnesses.memory import HistoryEntry
    if version is not None and not isinstance(version, ControllerVersion):
        raise TypeError('Initial Controller seed must be a ControllerVersion')
    if not isinstance(history, (tuple, list)) or any(not isinstance(row, HistoryEntry) for row in history):
        raise TypeError('Inherited Controller history must contain approved HistoryEntry records')
    if len({row.record_id for row in history}) != len(history):
        raise ValueError('Duplicate inherited Controller history identity')
    return version, tuple(history)


def controller_run_record(config, version, history, *, parent_bound):
    """Keep run bootstrap data separate from code selection and live checkpoints."""
    version, history = _controller_seed(version, history)
    record = {**config.source.to_dict(), 'objective': config.objective}
    if parent_bound:
        # The grant itself (gateway, token, budget) is never portable or written here.
        record['requires_parent'] = True
    elif version is not None or history:
        record['controller_seed'] = {'version': encode(version), 'approved_history': encode(history)}
    return record


def restore_controller_seed(record, config):
    identity = {**config.source.to_dict(), 'objective': config.objective}
    if (not isinstance(record, dict) or not identity.keys() <= record.keys()
            or any(record[key] != value for key, value in identity.items())):
        raise ValueError('Restored run identity does not match configuration')
    if record.get('requires_parent') is True:
        raise RuntimeError('Parent-bound runs cannot be restored independently')
    if record.keys() == identity.keys():
        return None, ()
    if record.keys() != identity.keys() | {'controller_seed'}:
        raise ValueError('Invalid run bootstrap fields')
    seed = record['controller_seed']
    if not isinstance(seed, dict) or set(seed) != {'version', 'approved_history'}:
        raise ValueError('Invalid Controller run seed fields')
    return _controller_seed(decode(seed['version']), decode(seed['approved_history']))


def controller_version_bytes(version):
    """The artifact's exact canonical content, including typed, possibly ordered native state."""
    if not isinstance(version, ControllerVersion):
        raise TypeError('A ControllerVersion is required')
    return json_bytes({'schema': 1, 'version': encode(version)})


def save_controller_version(artifacts, version):
    return artifacts.put(controller_version_bytes(version), kind='controller-version')


def load_controller_version(artifacts, ref):
    if isinstance(ref, dict):
        ref = ArtifactRef(**ref)
    if not isinstance(ref, ArtifactRef) or ref.kind != 'controller-version' or ref.digest is None:
        raise ValueError('A digest-pinned controller-version artifact is required')
    raw = json.loads(artifacts.read(ref))
    if not isinstance(raw, dict) or set(raw) != {'schema', 'version'} or type(raw['schema']) is not int or raw['schema'] != 1:
        raise ValueError('Unsupported Controller version artifact schema')
    version = decode(raw['version'])
    if not isinstance(version, ControllerVersion):
        raise ValueError('Controller version artifact has an unexpected payload')
    return version
