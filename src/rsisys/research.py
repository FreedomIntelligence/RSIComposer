"""One composition entry for research feedback products.

Application/Demo remain low-level loop builders. Research launches supply a
ResearchAssembly and pass their finished decision loop through assemble_research.
No model calls or execution capabilities are introduced by assembly itself.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

from rsisys.loops.decision_making.feedback.components import (
    ContextMemoryOrganizer, FeedbackProcessor, TrajectoryArchive,
    JournalContextMemory, JournalTrajectoryArchive,
)
from rsisys.loops.decision_making.feedback.pipeline import FeedbackPipeline
from rsisys.runtime.files import atomic_write, json_bytes


def _component(component):
    if component is None:
        return None
    configuration = getattr(component, 'configuration', None)
    result = {'type': f'{type(component).__module__}.{type(component).__qualname__}'}
    if callable(configuration):
        settings = configuration()
        if not isinstance(settings, dict):
            raise TypeError('Research component configuration must be a JSON object')
        result['configuration'] = deepcopy(settings)
    json_bytes(result)
    return result


@dataclass(frozen=True)
class ResearchAssembly:
    """Explicit products: required memory, default traces, optional proposals.

    Journal memory supplies factual views without model inference. Use langmem()
    for the existing upstream manager, or supply an external provider/processor.
    Component state belongs to this run; reconstruction is an explicit recovery
    responsibility, not serialization of arbitrary Python objects.
    """
    context_memory: ContextMemoryOrganizer = field(default_factory=JournalContextMemory)
    memory_processor: FeedbackProcessor | None = None
    trajectory_archive: TrajectoryArchive | None = field(default_factory=JournalTrajectoryArchive)
    harness_proposals: FeedbackProcessor | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        for method in ('record', 'working_memory', 'memory'):
            if not callable(getattr(self.context_memory, method, None)):
                raise TypeError('Research requires an active context/memory organizer: ' + method)
        if self.trajectory_archive is not None and not callable(getattr(self.trajectory_archive, 'records', None)):
            raise TypeError('Research trajectory archive requires records, or None to disable it')
        FeedbackPipeline(memory_processor=self.memory_processor, harness_proposals=self.harness_proposals)
        self.configuration()

    @property
    def requires_model(self):
        return self.memory_processor is not None or self.harness_proposals is not None

    @property
    def memory_provider(self):
        config = _component(self.context_memory).get('configuration', {})
        provider = config.get('provider')
        if isinstance(provider, str) and provider:
            return provider
        return 'journal' if type(self.context_memory) is JournalContextMemory else 'custom'

    def configuration(self):
        return {'schema_version': 1, 'entrypoint': 'rsisys.research.assemble_research',
            'context_memory': _component(self.context_memory),
            'memory_provider': self.memory_provider,
            'memory_processor': _component(self.memory_processor),
            'trajectory_archive': _component(self.trajectory_archive),
            'harness_proposals': _component(self.harness_proposals),
            'trajectory_mode': 'extraction_only',
            'construction_owner': 'main_meta_agent_decision', 'budget_owner': 'host'}

    @classmethod
    def langmem(cls, *, memory_path, namespace, memory_options=None, **components):
        """Load the selected optional integration; never silently fall back."""
        if {'context_memory', 'memory_processor'}.intersection(components):
            raise ValueError('langmem() owns context_memory and memory_processor')
        try:
            from rsisys.loops.decision_making.feedback.langmem import LangMemContextMemory
        except ImportError as exc:
            raise RuntimeError('LangMem research memory requires the installed rsisys[langmem] dependencies') from exc
        memory = LangMemContextMemory(memory_path, namespace=namespace, **(memory_options or {}))
        return cls(context_memory=memory, memory_processor=memory, **components)


def assemble_research(loop, *, assembly: ResearchAssembly, path=None, backend=None,
                      model=None, adapter=None, owner=None, **learning_options):
    """Install the same validated pipeline for generic and domain launchers.

    A journal records the exact assembly alongside processor/host limits. Old
    journals are not silently migrated. A caller may register the service with
    its AssemblyContext via owner; the default adapter also hooks loop cleanup.
    """
    if not isinstance(assembly, ResearchAssembly):
        raise TypeError('Research launches require a ResearchAssembly')
    assembly.validate()
    reserved = {'context_memory', 'processor', 'trajectory_archive', 'assembly_configuration'}
    if reserved.intersection(learning_options):
        raise ValueError('Configure research products through ResearchAssembly')
    if getattr(loop, 'feedback_service', None) is not None:
        raise ValueError('Research feedback is already assembled')
    if owner is not None and (not callable(getattr(owner, 'add_cleanup', None))
                              or getattr(owner, '_closed', False)):
        raise ValueError('Research owner must accept cleanup callbacks')
    if getattr(assembly.memory_processor, 'requires_tool_calling', False) is True:
        selected = backend if backend is not None else getattr(getattr(loop.harness, 'policy', None), 'backend', None)
        if getattr(selected, 'supports_tool_calling', False) is not True:
            raise ValueError('LangMem research memory requires a backend with native tool calling')
    from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning
    configuration = assembly.configuration()
    if path is not None:
        manifest = Path(path).parent / 'research_assembly.json'
        if manifest.exists() and json.loads(manifest.read_text()) != configuration:
            raise ValueError('Research assembly configuration differs from the workspace manifest')
    enable_feedback_learning(loop, path=path, backend=backend, model=model, adapter=adapter,
        context_memory=assembly.context_memory, trajectory_archive=assembly.trajectory_archive,
        processor=FeedbackPipeline(memory_processor=assembly.memory_processor,
                                   harness_proposals=assembly.harness_proposals),
        assembly_configuration=configuration, **learning_options)
    service = loop.feedback_service
    try:
        # Persist configuration even when a run finishes before its first receipt.
        service.learning._commit(deepcopy(service.learning.state))
        if path is not None:
            atomic_write(Path(path).parent / 'research_assembly.json', json_bytes(configuration))
        if owner is not None:
            owner.add_cleanup(service.close)
    except BaseException as exc:
        try:
            service.close()
        except BaseException as cleanup_error:
            exc.add_note(f'Research service cleanup also failed: {cleanup_error}')
        raise
    loop.research_assembly = deepcopy(configuration)
    return service
