"""Research-specific input window, memory and local tools."""
from copy import deepcopy
from dataclasses import replace
from rsicomposer.harnesses.decision.attempts import AttemptContract
from rsicomposer.harnesses.decision.context import BasicContextCompactor
from rsicomposer.harnesses.decision.memory import DecisionMemory
from rsicomposer.harnesses.decision.memory.feedback import RecentFeedbackSelector
from rsicomposer.harnesses.decision.research import ResearchAgenda
from rsicomposer.harnesses.decision.state import DecisionContext, ControllerHarnessState
from rsicomposer.harnesses.decision.public_policy import DecisionInputAdapter
from rsicomposer.loops.contracts.decision import OperationSpec
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.evidence.records import is_host_final_event
from rsicomposer.runtime.evidence.retrieval import controller_experience

class ResearchControllerHarness:
    def __init__(self, *, local_tools, experience, experience_limit=32, memory=None,
                 memory_limit=32, feedback_selector=None, context_compactor=None,
                 research_agenda=None, attempt_contract=None):
        if type(experience_limit) is not int or experience_limit < 1 or type(memory_limit) is not int or memory_limit < 1:
            raise ValueError('Experience and memory limits must be positive integers')
        self.local_tools, self.experience = local_tools, controller_experience(experience)
        self.experience_limit, self.memory_limit = experience_limit, memory_limit
        self.memory = memory if memory is not None else DecisionMemory()
        self.research_agenda = research_agenda if research_agenda is not None else ResearchAgenda()
        self.context_compactor = context_compactor if context_compactor is not None else BasicContextCompactor()
        self.feedback_selector = feedback_selector if feedback_selector is not None else RecentFeedbackSelector()
        self.attempt_contract = attempt_contract if attempt_contract is not None else AttemptContract()
        self.feedback_proposal_resolver = None
        self.evidence_observer = None
        self._context_processors, self._cleanup_callbacks = [], []
        self._state = ControllerHarnessState()
        self._execution = None
        self.prepared_context = None
        self.feedback_learning = self.feedback_service = self.research_memory = None
        self.last_working_view = None

    def state(self):
        """Capture this Harness version through its explicit recovery port."""
        return {'observations': deepcopy(self._state),
                'research_agenda': self.research_agenda.export(),
                'memory': deepcopy(self.memory.export())}

    def restore(self, state):
        import json
        if not isinstance(state, dict) or set(state) != {'observations', 'research_agenda', 'memory'}:
            raise ValueError('Invalid research Harness state')
        observations = deepcopy(state['observations'])
        if not isinstance(observations, ControllerHarnessState):
            raise TypeError('Invalid research Harness observation state')
        observations.__post_init__()
        saved = json.loads(json.dumps(state['memory'], allow_nan=False))
        current = json.loads(json.dumps(self.memory.export(), allow_nan=False))
        prefix = current['history'][:len(saved['history'])]
        if current['history'] and prefix != saved['history']:
            raise ValueError('Controller history differs from the checkpoint prefix')
        self.research_agenda.restore(state['research_agenda'])
        if len(current['history']) < len(saved['history']) or not current['history']:
            self.memory._load(saved)
            self.memory._commit(self.memory._history, self.memory._memories)
        self._state = observations

    def history(self):
        return self.memory.history()

    def seed_history(self, entries):
        self.memory.seed(tuple(entries))

    def bind_execution(self, read):
        if not callable(read):
            raise TypeError('Research Harness execution view must be callable')
        self._execution = read
        research_tools = getattr(self.local_tools, 'research_tools', None)
        if research_tools is not None:
            research_tools.decision_graph = lambda: read()['decision_graph']

    def freeze_decision_metadata(self, policy, decision_id):
        from dataclasses import asdict
        from rsicomposer.host.ledger import DecisionResearchMetadata
        read = optional_method(policy, 'decision_metadata')
        metadata = read(decision_id) if read is not None else DecisionResearchMetadata()
        if not isinstance(metadata, DecisionResearchMetadata):
            raise TypeError('Native research metadata must be DecisionResearchMetadata')
        return asdict(metadata)

    def admit_decision_metadata(self, decision, requested_snapshot, records):
        from rsicomposer.host.ledger import DecisionResearchMetadata, admit_decision_metadata
        snapshot = admit_decision_metadata(decision, DecisionResearchMetadata.from_record(requested_snapshot), records,
            attempt_contract=self.attempt_contract, proposal_resolver=self.feedback_proposal_resolver)
        return {'record': snapshot, 'context': {key: deepcopy(snapshot[key]) for key in
            ('research_attempt_id', 'research_attempt_title', 'feedback_proposal_ids')}}

    def validate_execution_records(self, records):
        from rsicomposer.host.ledger import validate_decision_metadata
        validate_decision_metadata(records)

    def tools(self):
        return tuple(definition.spec for definition in self.local_tools.definitions)

    def execute_tool(self, call):
        return self.local_tools.execute(call)

    def window(self, objective, capabilities, feedback):
        read = self._execution
        if read is None:
            raise RuntimeError('Research Controller Harness requires a bound execution read view')
        execution = read()
        context = DecisionContext(objective,
            deepcopy(tuple(self.feedback_selector.select(self.experience, objective, limit=self.experience_limit))),
            tuple(OperationSpec(**{key: value for key, value in row.items() if key != 'kind'}) for row in capabilities),
            deepcopy(tuple(execution['operation_results'])),
            self.memory.recall(limit=self.memory_limit), history=self.memory.history(),
            decision_model=execution['decision_model'], pending=tuple(execution['pending']),
            feedbacks=deepcopy(feedback),
            progresses=deepcopy(tuple(execution['progresses'])),
            run_control=deepcopy(execution['run_control']), research_agenda=self.research_agenda.context(),
            decision_graph=execution['decision_graph'])
        self.prepared_context = self._project_context(context)
        return DecisionInputAdapter()(self.prepared_context)

    def receipt_context_delivered(self, event_id):
        """Report a canonical receipt already accepted by the external Memory."""
        return self.evidence_observer is not None and event_id in self._state.observed_event_ids

    def observe(self, event):
        if is_host_final_event(event):
            return
        if event.kind == 'history':
            from rsicomposer.harnesses.memory import HistoryEntry
            raw = deepcopy(event.output['history_entry'])
            raw['source_refs'] = tuple(raw.get('source_refs', ()))
            self.memory.append(HistoryEntry(**raw))
            return
        if event.event_id in self._state.observed_event_ids:
            return
        if self.evidence_observer is not None:
            self.evidence_observer(deepcopy(event))
        if event.event_id not in self._state.observed_event_ids:
            self._state.observed_event_ids += (event.event_id,)


    def bind_feedback(self, service):
        """Own one rich projection before rendering the public or native Policy window."""
        from .learning_harness import ResearchFeedbackProjection
        projection = ResearchFeedbackProjection(self, service)
        project_memory = optional_method(service.memory, 'project_context')
        processors = (projection, project_memory) if project_memory is not None else (projection,)
        self.feedback_service, self.feedback_learning = service, service.learning
        self.research_memory = service.memory
        self.feedback_proposal_resolver = service.resolve_proposals
        self.evidence_observer = lambda event: (service.learning.context_observer or
            service.learning.context_binding.observer)(event) if (
                service.learning.context_observer or service.learning.context_binding.observer) is not None else None
        self._context_processors.extend(processors)

        def detach():
            for processor in processors:
                self._context_processors.remove(processor)
            self.feedback_service = self.feedback_learning = self.research_memory = None
            self.feedback_proposal_resolver = self.evidence_observer = None
        return detach

    def attempt_interpretations(self):
        return self.feedback_service.attempts() if self.feedback_service is not None else {'items': []}

    def add_context_processor(self, processor):
        self._context_processors.append(processor)

    def add_cleanup(self, callback):
        self._cleanup_callbacks.append(callback)

    def relinquish_cleanups(self, matches):
        """Transfer closes claimed by the enclosing owner; keep Harness-local cleanup."""
        self._cleanup_callbacks[:] = [close for close in self._cleanup_callbacks if not matches(close)]

    def _project_context(self, context):
        from rsicomposer.harnesses.decision.research_state import (
            bounded_evidence_view,
            previous_feedback_view,
        )
        previous = previous_feedback_view(context)
        agenda = {key: value for key, value in context.research_agenda.items() if key != 'previous_feedback'}
        if previous is not None:
            agenda['previous_feedback'] = previous
        context = replace(context, research_agenda=agenda)
        context = bounded_evidence_view(context)
        for processor in self._context_processors:
            context = processor(context)
        read = optional_method(self.local_tools, 'runtime_context')
        if read is not None:
            value = read()
            if not isinstance(value, str):
                raise TypeError('runtime_context() must return text')
            agenda = {key: item for key, item in context.research_agenda.items() if key != 'runtime_context'}
            if value:
                agenda['runtime_context'] = value
            context = replace(context, research_agenda=agenda)
        return context

    def close(self):
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        failed, errors = [], []
        for callback in reversed(callbacks):
            try:
                callback()
            except BaseException as error:
                failed.append(callback)
                errors.append(error)
        # A timed-out feedback worker may need another close. Retry failed
        # callbacks only; successful resource releases must not run twice.
        self._cleanup_callbacks.extend(reversed(failed))
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f'Additional harness cleanup failure: {type(error).__name__}: {error}')
            raise errors[0]
