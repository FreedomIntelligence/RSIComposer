"""Optional ReMe retrieval for explicit, source-backed agent memory.

ReMe owns tokenization and BM25 scoring. RSIComposer owns committed notes, evidence,
revision cursors and tasks. The index is a disposable in-memory projection of
current notes, never a second write target or an independently acting agent.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from math import isfinite
from threading import RLock

from rsicomposer.runtime.storage.json_tools import result_id

REME_VERSION = '0.4.2.3'
REME_COMMIT = 'e3fade0a781d080a4be7fb83648a3b2e6d477522'


def _run(coroutine):
    """Support both synchronous harnesses and callers with an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='rsisys-reme') as worker:
        return worker.submit(asyncio.run, coroutine).result()


class ReMeBM25Retrieval:
    """Reuse upstream components without starting ReMe's application/services.

    No file watchers, pickle loading, embedding client, extraction or dream jobs
    are instantiated. Persistent truth is AgentMemory's revisioned JSON. Reopen,
    continuation and restore therefore rebuild exactly the current projection.
    """
    def __init__(self):
        self._components = None
        self._index = None
        self._digest = None
        self._lock = RLock()

    def configuration(self):
        return {'provider': 'reme', 'method': 'bm25', 'package': 'reme-ai',
            'version': REME_VERSION, 'dependency_commit': REME_COMMIT,
            'tokenizer': 'regex', 'filter_stopwords': False, 'k1': 1.5, 'b': 0.75,
            'fields': ['memory_id', 'content', 'applicability'],
            'index': 'rebuildable_in_memory', 'scope': 'current_agent_memory',
            'semantic_search': False, 'model_calls': 0, 'embedding_calls': 0,
            'auto_memory': False, 'auto_dream': False}

    def prepare(self):
        """Fail before binding/claiming a run if the optional runtime is absent."""
        with self._lock:
            if self._components is not None:
                return
            try:
                version = metadata.version('reme-ai')
                if version != REME_VERSION:
                    raise RuntimeError(f'ReMe requires reme-ai=={REME_VERSION}; found {version}')
                from reme.components.keyword_index import BM25Index
                from reme.components.tokenizer import RegexTokenizer
            except ImportError as exc:
                raise RuntimeError('ReMe memory requires the pinned optional runtime; '
                                   'install requirements/reme.txt (see src/rsicomposer/README.md#memory-ownership)') from exc
            self._components = BM25Index, RegexTokenizer

    def rank(self, query, notes):
        """Return positive scores and the exact searchable-corpus fingerprint."""
        documents = {row['memory_id']: '\n'.join(row[key] for key in
                     ('memory_id', 'content', 'applicability'))
                     for row in sorted(notes, key=lambda row: row['memory_id'])}
        digest = result_id(documents)
        with self._lock:
            self.prepare()
            async def retrieve():
                if self._digest != digest:
                    index_type, tokenizer_type = self._components
                    candidate = index_type(tokenizer='', k1=1.5, b=0.75)
                    # These upstream public components support add_docs/retrieve
                    # without application startup. Explicitly disable stopwords:
                    # no tokenizer.start() or index.start() means no file I/O.
                    candidate.tokenizer = tokenizer_type(filter_stopwords=False)
                    await candidate.add_docs(documents)
                    self._index, self._digest = candidate, digest
                return await self._index.retrieve(query, limit=len(documents))
            scores = _run(retrieve())
            if (not isinstance(scores, dict) or any(key not in documents
                    or type(score) not in (int, float) or not isfinite(score) or score <= 0
                    for key, score in scores.items())):
                raise RuntimeError('ReMe returned invalid memory scores')
            return scores, digest
