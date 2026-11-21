"""One composition entry for research feedback products.

ModuleAssembly.build constructs the loops and installs an optional ResearchAssembly
through assemble_research. Memory providers supply their own optional lifecycle;
custom harnesses supply explicit feedback adapters and any required memory host.
No model calls or execution capabilities are introduced by assembly itself.
"""
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rsicomposer.harnesses.decision.agent_memory import AgentMemory
from rsicomposer.harnesses.decision.feedback.archive import validate_archive
from rsicomposer.harnesses.decision.feedback.components import (
    JournalTrajectoryArchive,
)
from rsicomposer.harnesses.decision.feedback.context_memory import validate_context_memory
from rsicomposer.harnesses.decision.feedback.pipeline import FeedbackPipeline
from rsicomposer.harnesses.decision.feedback.scheduling import validate_scheduler
from rsicomposer.loops.contracts.feedback_modules import (
    FeedbackAdmissionPolicy,
    FeedbackFailurePolicy,
    FeedbackProcessor,
    FeedbackScheduler,
    TrajectoryArchive,
)
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
from rsicomposer.loops.contracts.feedback_context import ControllerMemory
from rsicomposer.runtime.memory import memory_lifecycle


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
    """One acting meta-agent with explicit memory, plus optional integrations.

    The default memory stores and retrieves notes written by the acting meta-agent.
    Component state belongs to this run; reconstruction uses explicit component
    state adapters or a recovery contract, not arbitrary Python serialization.
    """
    context_memory: ControllerMemory = field(default_factory=AgentMemory)
    memory_processor: FeedbackProcessor | None = None
    trajectory_archive: TrajectoryArchive | None = field(default_factory=JournalTrajectoryArchive)
    harness_proposals: FeedbackProcessor | None = None
    improvement_proposals: FeedbackProcessor | None = None
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
        for processor in (self.memory_processor, self.harness_proposals, self.improvement_proposals):
            if (getattr(processor, 'requires_context_memory_binding', False) is True
                    and (processor is not self.context_memory or processor is not self.memory_processor)):
                raise ValueError('The memory processor must be the same object as context_memory')
        FeedbackPipeline(memory_processor=self.memory_processor, harness_proposals=self.harness_proposals,
                         improvement_proposals=self.improvement_proposals,
                         proposal_requires_memory=self.proposal_requires_memory)
        self.configuration()

    def prepare(self):
        """Check optional local runtimes before claiming a new research run."""
        self.validate()
        lifecycle = memory_lifecycle(self.context_memory)
        if lifecycle is not None:
            lifecycle.prepare()

    @property
    def requires_model(self):
        return FeedbackPipeline(memory_processor=self.memory_processor, harness_proposals=self.harness_proposals,
            improvement_proposals=self.improvement_proposals,
            proposal_requires_memory=self.proposal_requires_memory).max_calls > 0

    @property
    def memory_provider(self):
        config = _component(self.context_memory).get('configuration', {})
        provider = config.get('provider')
        if isinstance(provider, str) and provider:
            return provider
        return 'custom'

    def configuration(self):
        memory = _component(self.context_memory)
        return {**({'admission': _component(self.admission)} if self.admission is not None else {}),
            'schema_version': 9, 'entrypoint': 'rsicomposer.research.assemble_research',
            'context_protocol': 'text-v1', 'context_delivery': 'replace_current_view',
            'archive_protocol': 'raw-event-v1', 'processor_protocol': 'product-v1',
            'proposal_protocol': 'product-v1', 'scheduler_protocol': 'selection-v1',
            'context_memory': memory,
            'memory_provider': self.memory_provider,
            'memory_processor': _component(self.memory_processor),
            'memory_processor_binding': ('context_memory' if self.memory_processor is self.context_memory
                                         else 'independent'),
            'trajectory_archive': _component(self.trajectory_archive),
            'harness_proposals': _component(self.harness_proposals),
            'improvement_proposals': _component(self.improvement_proposals),
            'scheduler': _component(self.scheduler),
            'proposal_requires_memory': self.proposal_requires_memory,
            'failure_policy': asdict(self.failure_policy),
            'trajectory_mode': 'archive_only', 'dataset_construction_owner': 'lab',
            'receipt_delivery': 'direct_before_processing',
            'construction_owner': 'main_meta_agent_decision', 'budget_owner': 'host',
            'agent_topology': ('single_acting_meta_agent' if memory.get('configuration', {}).get('organization_owner') == 'acting_meta_agent'
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


@dataclass(frozen=True)
class MemoryAssembly:
    """A Controller's Memory integration: memory and feedback products for its decision loop.

    research describes the products; backend optionally overrides the model used by
    feedback stages; adapter_factory(harness) supplies an explicit feedback adapter
    for a custom harness; options are passed to the feedback learning service.
    The host installs it once per run, after the decision loop is constructed.
    """
    research: ResearchAssembly
    backend: object = None
    adapter_factory: object = None
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.research, ResearchAssembly):
            raise TypeError('MemoryAssembly.research must be a ResearchAssembly')
        if self.adapter_factory is not None and not callable(self.adapter_factory):
            raise TypeError('adapter_factory must be callable')
        if not isinstance(self.options, dict):
            raise TypeError('options must be a dict')
        reserved = {'loop', 'assembly', 'owner', 'adapter', 'backend'}
        if reserved.intersection(self.options):
            raise ValueError('Harness bindings cannot be overridden by options')
        self.research.validate()

    def install(self, loop, *, owner, path):
        options = dict(self.options)
        options.setdefault('path', path)
        adapter = self.adapter_factory(loop.harness) if self.adapter_factory is not None else None
        return assemble_research(loop, assembly=self.research, backend=self.backend,
                                 owner=owner, adapter=adapter, **options)


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
    if loop.integration is not None:
        raise ValueError('Research feedback is already assembled')
    lifecycle = memory_lifecycle(assembly.context_memory)
    if lifecycle is not None:
        lifecycle.validate_host(loop)
    if owner is not None and (not callable(getattr(owner, 'add_cleanup', None))
                              or getattr(owner, '_closed', False)):
        raise ValueError('Research owner must accept cleanup callbacks')
    if getattr(assembly.memory_processor, 'requires_tool_calling', False) is True:
        selected = backend if backend is not None else getattr(getattr(loop, 'policy', None), 'backend', None)
        if getattr(selected, 'supports_tool_calling', False) is not True:
            raise ValueError('This memory processor requires a backend with native tool calling')
    from rsicomposer.harnesses.decision.learning_harness import enable_feedback_learning
    configuration = assembly.configuration()
    if path is not None:
        manifest = Path(path).parent / 'research_assembly.json'
        if manifest.exists() and json.loads(manifest.read_text()) != configuration:
            raise ValueError('Research assembly configuration differs from the workspace manifest')
    context_memory = assembly.context_memory
    if lifecycle is not None:
        context_memory = lifecycle.for_run(loop, path)
        validate_context_memory(context_memory)
    processor = (context_memory if assembly.memory_processor is assembly.context_memory
                 else assembly.memory_processor)
    enable_feedback_learning(loop, path=path, backend=backend, model=model, model_provider=model_provider,
        adapter=adapter,
        context_memory=context_memory, trajectory_archive=assembly.trajectory_archive,
        scheduler=assembly.scheduler, failure_policy=assembly.failure_policy, admission=assembly.admission,
        processor=FeedbackPipeline(memory_processor=processor,
                                   harness_proposals=assembly.harness_proposals,
                                   improvement_proposals=assembly.improvement_proposals,
                                   proposal_requires_memory=assembly.proposal_requires_memory),
        assembly_configuration=configuration, **learning_options)
    service = loop.integration
    from rsicomposer.method_state import ResearchMemoryState
    experience = owner.experience if owner is not None else loop.host_execution.experience
    service.memory_state = ResearchMemoryState(context_memory, experience=experience,
        controller_version=getattr(loop, 'controller_version', None),
        resuming=getattr(owner, 'resuming', False) is True)
    observer = (service.memory_state.observe if service.memory_state.rebuild is not None
                else service.learning.context_binding.observer)
    service.learning.context_observer = observer
    if owner is not None:
        owner._memory_observer = observer
    try:
        # Persist configuration even when a run finishes before its first receipt.
        service.learning._commit(deepcopy(service.learning.state))
        if path is not None:
            atomic_write(Path(path).parent / 'research_assembly.json', json_bytes(configuration))
        if lifecycle is not None:
            lifecycle.install(loop, context_memory, path=path)
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
