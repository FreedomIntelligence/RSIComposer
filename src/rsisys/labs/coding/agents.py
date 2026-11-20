"""Harbor agent runtimes: how a candidate runs on a coding benchmark's tasks.

A runtime is trusted host configuration, named in a benchmark entry (never in
candidate data): which agent class Harbor imports, which of the benchmark's
budgets are bound into it, and what a candidate may set. Candidate content cannot
select a different import or change a budget.

    [labs.coding.benchmarks.swebench-verified.runtime]
    name = "harbor-instruction"          # the default: the bundled CodingAgent
    [labs.coding.benchmarks.my-bench.runtime]
    name = "harbor-agent-tree"
    options = { import_path = "my_pkg.agent:MyAgent", python_paths = ["src"] }

harbor-instruction accepts a small JSON candidate (an instruction and a bounded
harness setting) and runs it as data. harbor-agent-tree accepts a source tree and
imports it on the worker host, outside the task sandbox, so a Lab runs it only when
its config says trust = ["host-code"].
"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

from rsisys.loops.contracts.domain_agent import DomainAgentSpec

_BINDINGS = {'artifact', 'artifact_dir', 'endpoint', 'model', 'max_steps',
             'max_output_tokens', 'total_output_tokens', 'seed', 'thinking'}
DEFAULT_BINDINGS = {
    'candidate_path': 'artifact', 'endpoint': 'endpoint',
    'max_steps': 'max_steps', 'max_output_tokens': 'max_output_tokens',
    'total_output_tokens': 'total_output_tokens', 'seed': 'seed', 'thinking': 'thinking',
}
INSTRUCTION = DomainAgentSpec('instruction', {
    'instruction': {'type': 'string', 'max_length': 12000, 'required': True},
    'hypothesis': {'type': 'string'}, 'id': {'type': 'string'}, 'sha256': {'type': 'string'},
    'harness': {'type': 'object'},
}, 'data')


@dataclass(frozen=True)
class HarborAgentRuntime:
    """The runtime one coding benchmark runs its candidates with."""
    name: str = 'harbor-instruction'
    version: str = '1'
    model_access: str = 'endpoint-in-worker'
    import_path: str = 'rsisys.labs.coding.agent:CodingAgent'
    kwargs: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=lambda: dict(DEFAULT_BINDINGS))
    python_paths: tuple = ()
    skip_model_parsing: bool = True
    accepts: DomainAgentSpec = INSTRUCTION

    def run(self, candidate, task, budget):
        raise NotImplementedError('Harbor trials run in a worker process; see HarborTrialRunner')

    def validate(self):
        if not isinstance(self.import_path, str) or ':' not in self.import_path:
            raise ValueError('Harbor agent needs a module:Class import path')
        if type(self.skip_model_parsing) is not bool:
            raise ValueError('skip_model_parsing must be boolean')
        if not isinstance(self.kwargs, dict) or not isinstance(self.bindings, dict):
            raise ValueError('Harbor agent kwargs and bindings must be objects')
        if any(not isinstance(k, str) or not isinstance(v, str) or v not in _BINDINGS for k, v in self.bindings.items()):
            raise ValueError('Unknown Coding Lab agent binding')
        for p in self.python_paths:
            if not isinstance(p, str) or Path(p).is_absolute() or '..' in Path(p).parts:
                raise ValueError('Agent Python paths must stay within the candidate tree')

    def to_json(self):
        """What the worker needs to rebuild the agent: trusted configuration only."""
        self.validate()
        return json.loads(json.dumps({
            'import_path': self.import_path, 'version': self.version, 'kwargs': self.kwargs,
            'bindings': self.bindings, 'artifact_python_paths': list(self.python_paths),
            'skip_model_parsing': self.skip_model_parsing}, allow_nan=False))


def instruction_runtime(options):
    if options:
        raise ValueError('harbor-instruction takes no options')
    return HarborAgentRuntime()


def agent_tree_runtime(options):
    allowed = {'import_path', 'python_paths', 'kwargs', 'version', 'bindings'}
    if set(options) - allowed or 'import_path' not in options:
        raise ValueError('harbor-agent-tree requires import_path and accepts only ' + ', '.join(sorted(allowed)))
    return HarborAgentRuntime(
        name='harbor-agent-tree', version=str(options.get('version', '1')), import_path=options['import_path'],
        kwargs=dict(options.get('kwargs', {})), bindings={**DEFAULT_BINDINGS, **options.get('bindings', {})},
        python_paths=tuple(options.get('python_paths', ())),
        accepts=DomainAgentSpec('agent-tree', {}, 'host-code'))


RUNTIMES = {'harbor-instruction': instruction_runtime, 'harbor-agent-tree': agent_tree_runtime}
DEFAULT_RUNTIME = {'name': 'harbor-instruction'}


def agent_configuration(spec, values):
    """Resolve only the runtime's explicit kwargs, before importing Harbor."""
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
