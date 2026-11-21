"""Immutable Controller versions stored with the explicit registered JSON codec."""
import json

from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.loops.contracts.controller_version import ControllerVersion
from rsicomposer.runtime.storage.codec import decode, encode
from rsicomposer.runtime.storage.files import json_bytes


def save_controller_version(artifacts, version):
    if not isinstance(version, ControllerVersion):
        raise TypeError('A ControllerVersion is required')
    return artifacts.put(json_bytes({'schema': 1, 'version': encode(version)}), kind='controller-version')


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
