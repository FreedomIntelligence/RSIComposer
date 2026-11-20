"""Trusted host adapter from a method artifact to a Harbor agent.

One adapter describes an agent runtime, independent of the RSI search method.
Candidate data cannot select a different host import or change task budgets.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path


_BINDINGS = {'artifact', 'artifact_dir', 'endpoint', 'model', 'max_steps',
             'max_output_tokens', 'total_output_tokens', 'seed', 'thinking'}


@dataclass(frozen=True)
class HarborCodingAgent:
    """Construct in a trusted method/Lab factory, not from candidate JSON.

    artifact_python_paths are relative directories inside an immutable source
    artifact. Harbor imports that Python on the worker host, outside the task
    sandbox. Set this only for source artifacts trusted to execute there.
    """
    import_path: str = 'rsisys.labs.scenes.coding.agent:CodingAgent'
    version: str = '1'
    kwargs: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=lambda: {
        'candidate_path': 'artifact', 'endpoint': 'endpoint',
        'max_steps': 'max_steps', 'max_output_tokens': 'max_output_tokens',
        'total_output_tokens': 'total_output_tokens', 'seed': 'seed', 'thinking': 'thinking',
    })
    artifact_python_paths: tuple[str, ...] = ()
    skip_model_parsing: bool = True

    def to_json(self):
        if not isinstance(self.import_path, str) or ':' not in self.import_path:
            raise ValueError('Harbor agent needs a module:Class import path')
        if not isinstance(self.version, str) or not self.version:
            raise ValueError('Harbor agent adapter needs a version')
        if type(self.skip_model_parsing) is not bool:
            raise ValueError('skip_model_parsing must be boolean')
        if not isinstance(self.kwargs, dict) or not isinstance(self.bindings, dict):
            raise ValueError('Harbor agent kwargs and bindings must be objects')
        if any(not isinstance(k, str) or not isinstance(v, str) or v not in _BINDINGS for k, v in self.bindings.items()):
            raise ValueError('Unknown Coding Lab agent binding')
        for p in self.artifact_python_paths:
            if not isinstance(p, str) or Path(p).is_absolute() or '..' in Path(p).parts:
                raise ValueError('Agent Python paths must stay within the artifact')
        return json.loads(json.dumps({
            'import_path': self.import_path, 'version': self.version,
            'kwargs': self.kwargs, 'bindings': self.bindings,
            'artifact_python_paths': list(self.artifact_python_paths),
            'skip_model_parsing': self.skip_model_parsing,
        }, allow_nan=False))


def agent_configuration(spec, values):
    """Resolve only the adapter's explicit kwargs, before importing Harbor."""
    artifact = Path(values['artifact']).resolve(strict=True)
    paths = []
    for relative in spec['artifact_python_paths']:
        path = (artifact / relative).resolve(strict=True)
        if not artifact.is_dir() or not path.is_relative_to(artifact) or not path.is_dir():
            raise ValueError('Agent source path must be a directory inside the artifact')
        paths.append(str(path))
    kwargs = deepcopy(spec['kwargs'])
    kwargs.update({key: values[name] for key, name in spec['bindings'].items()})
    return {
        'name': None, 'import_path': spec['import_path'],
        'model_name': values['model'], 'skip_model_parsing': spec['skip_model_parsing'],
        'override_timeout_sec': values['agent_timeout'], 'kwargs': kwargs,
    }, paths
