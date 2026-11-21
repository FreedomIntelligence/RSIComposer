"""Digest-checked weight publications over a shared filesystem.

WeightSync is the transport boundary for future implementations such as NCCL.
FilesystemWeightSync is the only implementation provided here.
"""
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.runtime.storage.artifacts import FileArtifactStore
from rsicomposer.runtime.storage.artifact_access import ComponentArtifacts
from rsicomposer.runtime.storage.files import json_bytes


class WeightSync(Protocol):
    """Publish immutable weights and resolve a verified publication locally."""
    def publish(self, ref: ArtifactRef) -> ArtifactRef: ...

    def fetch(self, ref: ArtifactRef) -> Path: ...


class FilesystemWeightSync:
    """Stream managed artifacts into the shared store; never follow arbitrary URIs."""
    def __init__(self, root, *, artifacts=None, namespace='default'):
        self.artifacts = artifacts
        self.store = FileArtifactStore(root, uri_scheme='file')
        self.namespace = namespace
        self._owner = 'weight-sync:' + sha256(namespace.encode()).hexdigest()

    def _publish(self, ref, artifacts):
        if not isinstance(ref, ArtifactRef) or ref.digest is None:
            raise ValueError('Weight publication requires a digest-pinned ArtifactRef')
        resolve = getattr(artifacts, 'resolve', None)
        if not callable(resolve):
            raise TypeError('Filesystem weight publication requires managed file artifacts')
        source = resolve(ref)
        source_store = artifacts._authority.store if isinstance(artifacts, ComponentArtifacts) else artifacts
        if source_store.access(ref)['sealed']:
            raise PermissionError('Sealed artifacts cannot be published as weights')
        try:
            shared = self.store.resolve(ref)
        except KeyError:
            shared = None
        weights = ref if shared is not None and shared == source else self.store.put_file(source, kind=ref.kind)
        if weights.digest != ref.digest:
            raise ValueError('Weight source digest changed during publication')
        return self.store.put(json_bytes({'schema': 1, 'namespace': self.namespace, 'weights': asdict(weights)}),
            kind='weight-sync-publication', access={'owner': self._owner, 'split': None, 'sealed': False})


    def publish(self, ref):
        return self._publish(ref, self.artifacts)

    def fetch(self, ref):
        if (not isinstance(ref, ArtifactRef) or ref.kind != 'weight-sync-publication'
                or not ref.uri.startswith('file://')):
            raise ValueError('Filesystem weight fetch requires a published file ArtifactRef')
        access = self.store.access(ref)
        if access['owner'] != self._owner or access['sealed']:
            raise PermissionError('Weights were not published by the host on this edge')
        publication = json.loads(self.store.read(ref))
        if (not isinstance(publication, dict) or set(publication) != {'schema', 'namespace', 'weights'}
                or type(publication['schema']) is not int or publication['schema'] != 1):
            raise ValueError('Unsupported weight publication metadata')
        if publication['namespace'] != self.namespace:
            raise PermissionError('Weights were not published on this edge')
        weights = ArtifactRef(**publication['weights'])
        if self.store.access(weights)['sealed']:
            raise PermissionError('Published weights are restricted')
        return self.store.resolve(weights)


class _DirectionalWeightSync:
    def __init__(self, service, artifacts, *, publish, fetch):
        self._service, self._artifacts = service, artifacts
        self._publish_allowed, self._fetch_allowed = publish, fetch

    def publish(self, ref):
        if not self._publish_allowed:
            raise PermissionError('Only the source role may publish on this weight-sync edge')
        return self._service._publish(ref, self._artifacts)

    def fetch(self, ref):
        if not self._fetch_allowed:
            raise PermissionError('Only the target role may fetch on this weight-sync edge')
        return self._service.fetch(ref)


def _edge(name):
    if not isinstance(name, str) or len(parts := name.split(' -> ')) != 2 or any(
            not role or '.' not in role or any(character.isspace() for character in role) for role in parts):
        raise ValueError('Weight-sync edges must name source.role -> target.role')
    return tuple(parts)


def build_weight_sync(config, root, artifacts):
    if not isinstance(config, Mapping):
        raise TypeError('weight_sync must map directed role edges to transports')
    services = {}
    for name, transport in config.items():
        _edge(name)
        if transport != 'filesystem':
            raise NotImplementedError('Only filesystem WeightSync is implemented')
        services[name] = FilesystemWeightSync(root, artifacts=artifacts, namespace=name)
    return MappingProxyType(services)


def component_weight_sync(services, component, artifacts):
    name = component.removeprefix('lab:')
    ports = {}
    for edge, service in services.items():
        source, target = _edge(edge)
        publish = source.rsplit('.', 1)[0] == name
        fetch = target.rsplit('.', 1)[0] == name
        if publish or fetch:
            ports[edge] = _DirectionalWeightSync(service, artifacts, publish=publish, fetch=fetch)
    return MappingProxyType(ports)


def validate_weight_sync_roles(services, resources):
    for edge in services:
        for role in _edge(edge):
            resources.request(role)
