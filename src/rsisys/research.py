"""One composition entry for research feedback products.

Application/Demo remain low-level loop builders. Research launches supply a
ResearchAssembly and pass their finished decision loop through assemble_research.
No model calls or execution capabilities are introduced by assembly itself.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from rsisys.loops.decision_making.feedback.components import (
    ContextMemoryOrganizer, FeedbackProcessor, TrajectoryArchive,
    JournalContextMemory, JournalTrajectoryArchive,
)
from rsisys.loops.decision_making.feedback.pipeline import FeedbackPipeline
from rsisys.loops.decision_making.feedback.context_memory import validate_context_memory
from rsisys.loops.decision_making.feedback.archive import validate_archive
from rsisys.loops.decision_making.feedback.scheduling import validate_scheduler
from rsisys.loops.contracts.feedback_modules import FeedbackScheduler, FeedbackFailurePolicy, FeedbackProposer, FeedbackAdmissionPolicy
from rsisys.runtime.storage.files import atomic_write, json_bytes
from rsisys.loops.decision_making.harness.agent_memory import AgentMemory, install_agent_memory


def _component(component):
    if component is None:
        return None
    configuration = getattr(component, 'configuration', None)
    result = {'type': f'{type(component).__module__}.{type(component).__qualname__}'}
    if callable(component) and not callable(getattr(component, 'context', None)):
        result['callable'] = {'module': getattr(component, '__module__', type(component).__module__),
                              'name': getattr(component, '__qualname__', type(component).__qualname__)}
    if callable(configuration):
        settings = configuration()
        if not isinstance(settings, dict):
            raise TypeError('Research component configuration must be a JSON object')
        result['configuration'] = deepcopy(settings)
    json_bytes(result)
    return result


@dataclass(frozen=True)
class ResearchAssembly:
    """One acting meta-agent with explicit memory, plus optional integrations.

    The default memory stores and retrieves notes written by the acting meta-agent.
    Use langmem() to explicitly install the legacy independent extraction workflow.
    Component state belongs to this run; reconstruction uses explicit component
    state adapters or a recovery contract, not arbitrary Python serialization.
    """
    context_memory: ContextMemoryOrganizer = field(default_factory=AgentMemory)
    memory_processor: FeedbackProcessor | None = None
    trajectory_archive: TrajectoryArchive | None = field(default_factory=JournalTrajectoryArchive)
    harness_proposals: FeedbackProcessor | None = None
    improvement_proposals: FeedbackProposer | FeedbackProcessor | None = None
    scheduler: FeedbackScheduler | None = None
    proposal_requires_memory: bool = False
    failure_policy: FeedbackFailurePolicy = field(default_factory=FeedbackFailurePolicy)
    admission: FeedbackAdmissionPolicy | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        validate_context_memory(self.context_memory)
        validate_archive(self.trajectory_archive)
        if self.scheduler is not None:
            validate_scheduler(self.scheduler)
        if self.admission is not None and not callable(getattr(self.admission, 'admit', None)):
            raise TypeError('Feedback admission requires admit(event, pending)')
        if not isinstance(self.failure_policy, FeedbackFailurePolicy):
            raise TypeError('failure_policy must be a FeedbackFailurePolicy')
        FeedbackPipeline(memory_processor=self.memory_processor, harness_proposals=self.harness_proposals,
                         improvement_proposals=self.improvement_proposals,
                         proposal_requires_memory=self.proposal_requires_memory)
        self.configuration()

    def prepare(self):
        """Check optional local runtimes before claiming a new research run."""
        self.validate()
        if isinstance(self.context_memory, AgentMemory):
            self.context_memory.prepare()

    @property
    def requires_model(self):
        return FeedbackPipeline(memory_processor=self.memory_processor, harness_proposals=self.harness_proposals,
            improvement_proposals=self.improvement_proposals,
            proposal_requires_memory=self.proposal_requires_memory).max_calls_per_job > 0

    @property
    def memory_provider(self):
        config = _component(self.context_memory).get('configuration', {})
        provider = config.get('provider')
        if isinstance(provider, str) and provider:
            return provider
        return 'journal' if type(self.context_memory) is JournalContextMemory else 'custom'

    def configuration(self):
        return {**({'admission': _component(self.admission)} if self.admission is not None else {}),
            'schema_version': 5, 'entrypoint': 'rsisys.research.assemble_research',
            'context_protocol': 'text-v1', 'context_delivery': 'replace_current_view',
            'archive_protocol': 'raw-event-v1', 'processor_protocol': 'product-v1',
            'proposal_protocol': 'product-v1', 'scheduler_protocol': 'selection-v1',
            'context_memory': _component(self.context_memory),
            'memory_provider': self.memory_provider,
            'memory_processor': _component(self.memory_processor),
            'trajectory_archive': _component(self.trajectory_archive),
            'harness_proposals': _component(self.harness_proposals),
            'improvement_proposals': _component(self.improvement_proposals),
            'scheduler': _component(self.scheduler),
            'proposal_requires_memory': self.proposal_requires_memory,
            'failure_policy': asdict(self.failure_policy),
            'trajectory_mode': 'archive_only', 'dataset_construction_owner': 'lab',
            'receipt_delivery': 'direct_before_processing',
            'construction_owner': 'main_meta_agent_decision', 'budget_owner': 'host',
            'agent_topology': ('single_acting_meta_agent' if isinstance(self.context_memory, AgentMemory)
                               and not self.requires_model else 'explicit_custom_workflow')}

    @classmethod
    def single_meta(cls, *, memory_path=None, require_memory_review=False,
                    memory_backend='literal', **components):
        """The meta-agent writes its own memories; storage never calls a model."""
        if {'context_memory', 'memory_processor', 'harness_proposals', 'improvement_proposals'}.intersection(components):
            raise ValueError('single_meta() owns memory and has no independent model processing stages')
        return cls(context_memory=AgentMemory(memory_path, require_review=require_memory_review,
                                              retrieval_backend=memory_backend), **components)

    @classmethod
    def reme(cls, *, memory_path=None, require_memory_review=False, **components):
        """Agent-owned ReMe retrieval; optional proposal stages remain explicit."""
        if {'context_memory', 'memory_processor'}.intersection(components):
            raise ValueError('reme() owns memory and has no independent memory organizer')
        return cls(context_memory=AgentMemory(memory_path, require_review=require_memory_review,
                                              retrieval_backend='reme'), **components)

    @classmethod
    def langmem(cls, *, memory_path, namespace, memory_options=None, **components):
        """Explicit legacy extraction workflow; not the single-meta default."""
        if {'context_memory', 'memory_processor'}.intersection(components):
            raise ValueError('langmem() owns context_memory and memory_processor')
        try:
            from rsisys.loops.decision_making.feedback.langmem import LangMemContextMemory
        except ImportError as exc:
            raise RuntimeError('LangMem research memory requires the installed rsisys[langmem] dependencies') from exc
        memory = LangMemContextMemory(memory_path, namespace=namespace, **(memory_options or {}))
        return cls(context_memory=memory, memory_processor=memory, **components)


def assemble_research(loop, *, assembly: ResearchAssembly, path=None, backend=None,
                      model=None, model_provider=None, adapter=None, owner=None, **learning_options):
    """Install the same validated pipeline for generic and domain launchers.

    A journal records the exact assembly alongside processor/host limits. Old
    journals are not silently migrated. A caller may register the service with
    its AssemblyContext via owner; the default adapter also hooks loop cleanup.
    """
    if not isinstance(assembly, ResearchAssembly):
        raise TypeError('Research launches require a ResearchAssembly')
    assembly.prepare()
    reserved = {'context_memory', 'processor', 'trajectory_archive', 'assembly_configuration',
                'memory_processor', 'harness_proposals', 'improvement_proposals',
                'scheduler', 'proposal_requires_memory', 'failure_policy', 'admission'}
    if reserved.intersection(learning_options):
        raise ValueError('Configure research products through ResearchAssembly')
    if getattr(loop, 'feedback_service', None) is not None:
        raise ValueError('Research feedback is already assembled')
    if isinstance(assembly.context_memory, AgentMemory):
        from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
        if not isinstance(loop.harness, DefaultDecisionHarness):
            raise TypeError('Custom harness must supply its own explicit agent-memory tool/context integration')
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
    context_memory = assembly.context_memory
    if isinstance(context_memory, AgentMemory):
        # Assembly is reusable configuration. Mutable run state belongs to the
        # new harness, including when the same ModuleAssembly resumes a run.
        context_memory = context_memory.clone()
        context_memory.bind(loop.harness, path)
    enable_feedback_learning(loop, path=path, backend=backend, model=model, model_provider=model_provider,
        adapter=adapter,
        context_memory=context_memory, trajectory_archive=assembly.trajectory_archive,
        scheduler=assembly.scheduler, failure_policy=assembly.failure_policy, admission=assembly.admission,
        processor=FeedbackPipeline(memory_processor=assembly.memory_processor,
                                   harness_proposals=assembly.harness_proposals,
                                   improvement_proposals=assembly.improvement_proposals,
                                   proposal_requires_memory=assembly.proposal_requires_memory),
        assembly_configuration=configuration, **learning_options)
    service = loop.feedback_service
    try:
        # Persist configuration even when a run finishes before its first receipt.
        service.learning._commit(deepcopy(service.learning.state))
        if path is not None:
            atomic_write(Path(path).parent / 'research_assembly.json', json_bytes(configuration))
        if isinstance(context_memory, AgentMemory):
            install_agent_memory(loop, context_memory, path=path)
        if owner is not None:
            register = getattr(owner, 'add_owned_cleanup', owner.add_cleanup)
            register(service.close)
    except BaseException as exc:
        try:
            service.close()
        except BaseException as cleanup_error:
            exc.add_note(f'Research service cleanup also failed: {cleanup_error}')
        raise
    loop.research_assembly = deepcopy(configuration)
    return service
