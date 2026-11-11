"""Reuse pinned FunSearch code and give its RNG and reset clock deterministic control."""

import builtins
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace

import numpy as np

REVISION = 'cc53f274237d7ab05c19df939edbc1f9616a7c19'
HASHES = {
    'implementation/config.py': 'bd118e9579e7230d8b1c969a0fc4e10fadcbfa116993c7aadfb988ac4fb500a6',
    'implementation/code_manipulation.py': '8538521e64b0f92a95ee88d0617dd499129a1de19d9502aacba3f109b77f7b3b',
    'implementation/programs_database.py': 'b2bef312d04861f467f304f65788efe959f0493818976b052c46b4703151affc',
}
FILES = ('config', 'code_manipulation', 'programs_database')


def _softmax(values, axis=-1):
    shifted = np.exp(values - np.max(values, axis=axis, keepdims=True))
    return shifted / shifted.sum(axis=axis, keepdims=True)


@lru_cache(maxsize=4)
def modules(source_dir):
    """Load byte-identical author modules with local-only dependency substitutions."""
    source_dir = Path(source_dir).resolve(strict=True)
    implementations = SimpleNamespace()

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'funsearch.implementation' and fromlist:
            return implementations
        if name == 'absl' and fromlist == ('logging',):
            return SimpleNamespace(logging=logging)
        if name == 'scipy':
            return SimpleNamespace(special=SimpleNamespace(softmax=_softmax))
        return builtins.__import__(name, globals, locals, fromlist, level)

    for name in FILES:
        path = source_dir / 'implementation' / f'{name}.py'
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != HASHES[f'implementation/{name}.py']:
            raise RuntimeError(f'Pinned FunSearch source differs: {name}')
        module = ModuleType(f'{__name__}._author_{name}')
        module.__file__ = str(path)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
        sys.modules[module.__name__] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
        setattr(implementations, name, module)
    return implementations


class Population:
    """Controller-owned author ProgramsDatabase and traceable parent identities."""

    def __init__(self, *, seed, islands, reset_every, source_dir):
        self.author = modules(source_dir)
        self.rng = np.random.RandomState(seed)
        self.reset_every = reset_every
        self.versions = {}
        template = self.author.code_manipulation.text_to_program(
            '# Online bin packing. CAPACITY = 100.\n'
            'def priority(item, bins):\n'
            '  return -(bins - item)\n')
        config = self.author.config.ProgramsDatabaseConfig(
            num_islands=islands, functions_per_prompt=2, reset_period=float('inf'))
        self.database = self.author.programs_database.ProgramsDatabase(config, template, 'priority')

    def register(self, source, version, receipt_id, decision_id, scores, island):
        program = self.author.code_manipulation.text_to_function(source)
        # The published DB uses the last score to rank and all scores to cluster.
        # Append the aggregate so ranking uses every development instance.
        score_map = {idx: -bins for idx, bins in enumerate(scores)}
        score_map[len(scores)] = -sum(scores) / len(scores)
        self.database.register_program(program, island, score_map)
        self.versions[source] = {'version': version, 'score_receipt_id': receipt_id,
                                 'score_decision_id': decision_id}

    def _sample(self, action):
        # The upstream module calls np.random globally. Supply only this
        # module's RNG so experiment seeds never perturb the host RNG.
        with _RNG_LOCK:
            upstream = self.author.programs_database
            before = upstream.np
            upstream.np = _NumpyRNG(self.rng)
            try:
                return action()
            finally:
                upstream.np = before

    def prompt(self):
        result = self._sample(self.database.get_prompt)
        parents = []
        for function in self.author.code_manipulation.text_to_program(result.code).functions[:-1]:
            key = 'def priority(item, bins):\n' + function.body
            parents.append(self.versions[key])
        # Upstream returns a next-version field one greater than the actual
        # header suffix. The chat adapter accepts either name and normalizes it.
        return result.code, int(result.island_id), parents

    def reset(self, completed):
        if completed and completed % self.reset_every == 0:
            self._sample(self.database.reset_islands)
            return True
        return False


_RNG_LOCK = RLock()


class _NumpyRNG:
    def __init__(self, rng):
        self.random = rng

    def __getattr__(self, name):
        return getattr(np, name)
