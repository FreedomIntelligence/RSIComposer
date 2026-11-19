import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from inspect import getattr_static

from rsisys.harnesses.decision.attempts import (
    AttemptContract,
    known_attempts,
    requested_attempt,
)
from rsisys.harnesses.decision.context import BasicContextCompactor
from rsisys.host.ledger import DecisionResearchMetadata
from rsisys.harnesses.decision.policy.prompts import DecisionPrompts
from rsisys.harnesses.decision.state import DecisionContext
from rsisys.loops.contracts.decision import Decision, OperationCall
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.model_service import (
    GenerationRequest,
    ModelBackend,
    ModelRef,
)
from rsisys.runtime.storage.json_tools import load_object as _load_object




class ModelDecisionPolicy:
    """Example JSON parser. Provider response parsing stays in policy."""

    def __init__(self, backend: ModelBackend, model: ModelRef, *,
                 max_validation_attempts: int = 1,
                 structured_output: bool | None = None, context_compactor=None, project_context=None, attempt_contract=None,
                 prompts: DecisionPrompts | None = None):
        if type(max_validation_attempts) is not int or not 1 <= max_validation_attempts <= 8:
            raise ValueError('max_validation_attempts must be between 1 and 8')
        if structured_output is not None and type(structured_output) is not bool:
            raise TypeError('structured_output must be a bool or None')
        supports_schema = getattr(backend, 'supports_response_schema', False) is True
        if structured_output is True and not supports_schema:
            raise ValueError('structured_output=True requires backend.supports_response_schema=True')
        self.structured_output = supports_schema if structured_output is None else structured_output
        self.attempt_contract = attempt_contract if attempt_contract is not None else AttemptContract()
        if not isinstance(self.attempt_contract, AttemptContract):
            raise TypeError('attempt_contract must be an AttemptContract')
        self.max_validation_attempts = max_validation_attempts
        self.prompts = prompts if prompts is not None else DecisionPrompts()
        if not isinstance(self.prompts, DecisionPrompts):
            raise TypeError('prompts must be a DecisionPrompts bundle')
        self.backend, self.model = backend, model
        self._context_compactor = context_compactor if context_compactor is not None else BasicContextCompactor()
        self._explicit_compactor = context_compactor is not None
        self._context_runtime = None
        self.context_token_count_mode = "unmeasured"
        self._usage_recorder = None
        self._decision_metadata = None
        if project_context is not None and not isinstance(project_context, dict):
            raise TypeError('project_context must be a JSON object or None')
        project = json.dumps(project_context or {}, sort_keys=True, allow_nan=False)
        if len(project.encode('utf-8')) > 32000:
            raise ValueError('project_context exceeds 32000 bytes; link detailed evidence instead')
        self._project_context_json = project

    @property
    def project_context(self):
        """Host-owned configuration; callers cannot mutate a live run through a reference."""
        return json.loads(self._project_context_json)

    @property
    def context_compactor(self):
        return self._context_compactor

    @context_compactor.setter
    def context_compactor(self, value):
        if not callable(getattr(value, 'compact', None)):
            raise TypeError('context_compactor requires compact(context, recent_limit=...)')
        self._context_compactor, self._explicit_compactor = value, True

    @contextmanager
    def context_scope(self, compactor, *, recent_limit):
        """Own final-request projection without overwriting configured objects."""
        previous = self._context_runtime
        selected = self.context_compactor if self._explicit_compactor else compactor
        self._context_runtime = selected, recent_limit
        try:
            yield
        finally:
            self._context_runtime = previous

    def share_backend(self, wrap):
        """Explicitly share a stateless dependency; wrappers may forward this."""
        self.backend = wrap(self.backend)
        return self.backend

    def _research_instructions(self, context=None):
        text = self.prompts.render('research')
        if context is not None and context.research_agenda.get('agent_memory', {}).get('organization_owner') == 'acting_meta_agent':
            text += '\n' + self.prompts.render('agent_memory')
        return text + ' '

    def _context_inputs(self, context):
        inputs = asdict(context)
        if self.project_context:
            inputs = {'objective': inputs.pop('objective'), 'project_context': self.project_context, **inputs}
        return inputs

    @staticmethod
    def _envelope(properties, required):
        if 'kind' in properties:
            properties = {**properties,
                'research_attempt_id': {'type': ['string', 'null'], 'minLength': 1, 'maxLength': 200,
                    'description': 'Continue a known attempt from decision_graph; never invent an ID.'},
                'research_attempt_title': {'type': 'string', 'maxLength': 500,
                    'description': 'Title alone starts a new attempt; with an ID it cannot rename the attempt.'}}
            properties = {**properties, **{name: {'type': 'array', 'items': {'type': 'string', 'minLength': 1},
                                                 'description': 'Distinct decision IDs; checked before execution.'}
                for name in ('depends_on_decision_ids', 'source_decision_ids')}}
            properties['feedback_proposal_ids'] = {'type': 'array', 'maxItems': 16, 'uniqueItems': True,
                'items': {'type': 'string', 'minLength': 1, 'maxLength': 512},
                'description': 'Optional harness or policy-training proposal IDs selected from feedback; not proof of execution or adoption.'}
        return {'type': 'object', 'properties': {**properties, 'reason': {'type': 'string'}},
                'required': list(required), 'additionalProperties': False}

    @staticmethod
    def _input_schema(schema, action):
        """Copy custom schemas; local references need rebasing before embedding.

        Walk schema positions only: e.g. a literal $ref in const/examples is data.
        No references are rewritten or silently resolved against the envelope.
        """
        def check(node):
            if not isinstance(node, dict):
                return
            for keyword in ('$ref', '$dynamicRef', '$recursiveRef'):
                ref = node.get(keyword)
                if isinstance(ref, str) and (not ref or ref.startswith('#')):
                    raise ValueError(f'Structured output cannot embed local {keyword} {ref!r} '
                                     f'in {action}; use a schema without local references '
                                     'or structured_output=False')
            for keyword in ('properties', 'patternProperties', '$defs', 'definitions',
                            'dependentSchemas', 'dependencies'):
                children = node.get(keyword)
                if isinstance(children, dict):
                    for child in children.values():
                        check(child)
            for keyword in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
                children = node.get(keyword)
                if isinstance(children, list):
                    for child in children:
                        check(child)
            for keyword in ('items', 'additionalItems', 'contains', 'additionalProperties',
                            'unevaluatedItems', 'unevaluatedProperties', 'propertyNames',
                            'not', 'if', 'then', 'else', 'contentSchema'):
                child = node.get(keyword)
                if isinstance(child, list):
                    for item in child:
                        check(item)
                else:
                    check(child)

        copied = deepcopy(schema)
        check(copied)
        return copied

    def _operation_branches(self, context):
        return [self._envelope({
            'kind': {'const': 'invoke'}, 'operation': {'const': spec.name},
            'arguments': self._input_schema(spec.input_schema, f'operation {spec.name!r}'),
        }, ('kind', 'operation', 'arguments')) for spec in context.operations]

    def decision_schema(self, context: DecisionContext) -> dict:
        """Describe this turn's envelopes without choosing a research workflow."""
        branches = self._operation_branches(context)
        branches.extend((
            self._envelope({'kind': {'const': 'wait'}, 'job': {
                'type': ['object', 'null'],
                'properties': {'job_id': {'type': 'string'}, 'backend': {'type': 'string'}},
                'required': ['job_id', 'backend'], 'additionalProperties': False,
            }}, ('kind',)),
            self._envelope({'kind': {'const': 'finish'}}, ('kind',)),
        ))
        return {'oneOf': self._attempt_branches(branches, context)}


    def _attempt_branches(self, branches, context):
        if not self.attempt_contract.required:
            return branches
        result = []
        known = known_attempts(context.decision_graph)
        for branch in branches:
            fields = branch['properties']
            if not self.attempt_contract.applies(fields.get('kind', {}).get('const'),
                                                  fields.get('operation', {}).get('const')):
                result.append(branch)
                continue
            # Separate complete envelope branches survive portable schema conversion;
            # an object-local anyOf/required refinement would be removed on the wire.
            new = deepcopy(branch)
            new['properties']['research_attempt_id'] = {'type': 'null'}
            new['properties']['research_attempt_title'].update(minLength=1, pattern=r'\S')
            new['required'].append('research_attempt_title')
            result.append(new)
            if known:
                continuation = deepcopy(branch)
                continuation['properties']['research_attempt_id'] = {'type': 'string', 'enum': list(known)}
                continuation['required'].append('research_attempt_id')
                result.append(continuation)
        return result

    @contextmanager
    def track_usage(self, record):
        """Report each received response, including rejected parses and guarded retries."""
        previous = self._usage_recorder
        self._usage_recorder = record
        try:
            yield
        finally:
            self._usage_recorder = previous

    def _received(self, response):
        if self._usage_recorder is not None:
            self._usage_recorder(response.usage)

    def decide(self, context: DecisionContext) -> Decision:
        attempt_context = context
        for attempt in range(self.max_validation_attempts):
            # Generation/transport/identity errors are outside the retry boundary.
            response = self._generate_decision(attempt_context)
            try:
                return self._parse_decision(response, attempt_context)
            except (ValueError, TypeError, KeyError) as exc:
                if attempt + 1 == self.max_validation_attempts:
                    raise
                attempt_context = replace(context, objective=context.objective + '\n' +
                    self.prompts.render('validation_correction', error=json.dumps(str(exc)[:600])))
        raise AssertionError('Unreachable validation loop')

    def _control_instructions(self, context):
        return self.prompts.render('control') + ' '

    def _scheduling_instructions(self, context):
        return self.prompts.render('scheduling') + ' '

    def _decision_request(self, context):
        inputs = self._context_inputs(context)
        return GenerationRequest(
            "decision_maker", self.model, json.dumps(inputs, separators=(',', ':')),
            system=self.prompts.render('decision', scheduling=self._scheduling_instructions(context),
                control=self._control_instructions(context), attempt=self._attempt_instructions(),
                research=self._research_instructions(context)),
            **({'response_schema': self.decision_schema(context)} if self.structured_output else {}),
        )

    def _count_request(self, request):
        # Optional capability must be declared, not synthesized by a dynamic proxy.
        counter = (getattr(self.backend, 'count_tokens')
                   if getattr_static(self.backend, 'count_tokens', None) is not None else None)
        if callable(counter):
            self.context_token_count_mode = getattr(self.backend, 'token_count_mode', 'provider_tokenizer')
            return counter(request)
        self.context_token_count_mode = 'utf8_conservative_estimate'
        return len(json.dumps({'messages': [{'role': 'system', 'content': request.system},
                                            {'role': 'user', 'content': request.prompt}]}).encode('utf-8'))

    @staticmethod
    def _summary_current_state(context):
        return {
            'decision_graph_archive': deepcopy(context.research_agenda.get('decision_graph_archive', {})),
            'state_scope': 'Working selection; older operation receipts and adoptions may be archived. '
                           'Missing records never establish a negative claim about the full history.',
            'confirmed_adoptions': [
                {'decision_id': row['decision_id'], 'candidate_ids': row['adopted_candidate_ids']}
                for row in context.decision_graph if row.get('adopted_candidate_ids')],
            'run_control': deepcopy(context.run_control),
        }

    def _summarize_context(self, records, previous, max_tokens, *, current_state=None):
        response = self.backend.generate(GenerationRequest(
            'decision_maker', self.model,
            json.dumps({'previous_summary': previous, 'archived_records': records,
                        'current_state': current_state}),
            system=self.prompts.render('summary'),
            # Summaries are JSON too. Let the serving/backend boundary reject
            # truncated or malformed replies before parsing or committing state.
            **({'response_schema': {'type': 'object',
                'properties': {'summary': {'type': 'string', 'minLength': 1}},
                'required': ['summary'], 'additionalProperties': False}}
               if self.structured_output else {}),
            max_output_tokens=max_tokens))
        self._received(response)
        if response.model != self.model:
            raise ValueError('Summary model identity mismatch')
        payload = _load_object(response.text)
        if set(payload) != {'summary'} or not isinstance(payload['summary'], str) or not payload['summary'].strip():
            raise ValueError('Invalid rolling context summary')
        return payload['summary']

    def _generate_context_request(self, context, build):
        from rsisys.harnesses.decision.context import with_working_decision_graph
        # Prompt selection must never shrink the context used by the parser.
        context = with_working_decision_graph(context)
        compactor, recent_limit = (self._context_runtime if self._context_runtime is not None
                                  else (self.context_compactor, 32))
        if isinstance(compactor, BasicContextCompactor):
            single_meta = context.research_agenda.get('agent_memory', {}).get('organization_owner') == 'acting_meta_agent'
            projected = compactor.compact(context, recent_limit=recent_limit,
                measure=lambda view: self._count_request(build(view)),
                summarize=(None if single_meta else lambda records, previous, limit: self._summarize_context(
                    records, previous, limit, current_state=self._summary_current_state(context))))
            compactor.last_measurement['count_method'] = self.context_token_count_mode
        else:
            projected = compactor.compact(context, recent_limit=recent_limit)
        return self.backend.generate(build(projected))

    def _generate_decision(self, context):
        response = self._generate_context_request(context, self._decision_request)
        self._received(response)
        if response.model != self.model:
            raise ValueError('Model response identity does not match the requested version')
        return response

    @staticmethod
    def _parse_reason(payload):
        reason = payload.get('reason', '')
        if not isinstance(reason, str):
            raise ValueError('reason must be a string')
        return reason

    def _attempt_instructions(self):
        if not self.attempt_contract.required:
            return self.prompts.render('attempt_optional') + ' '
        return self.prompts.render('attempt_required', operations=', '.join(self.attempt_contract.operations)) + ' '

    def _parse_attempt(self, payload, context):
        ident, title = payload.get('research_attempt_id'), payload.get('research_attempt_title', '')
        requested_attempt(ident, title, context.decision_graph)
        self.attempt_contract.validate(payload.get('kind'), payload.get('operation'), ident, title)
        return {'research_attempt_id': ident, 'research_attempt_title': title}

    @staticmethod
    def _parse_lineage(payload, context):
        if any(key in payload for key in ('decision_id', 'adopted_candidate_ids', 'adopted_decision_ids', 'confirmed_edges')):
            raise ValueError('Decision identity and confirmed adoption are system/executor fields')
        known = {row['decision_id']: row for row in context.decision_graph}
        result = {}
        for name in ('depends_on_decision_ids', 'source_decision_ids'):
            refs = payload.get(name, [])
            if (not isinstance(refs, list) or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                    or len(set(refs)) != len(refs)):
                raise ValueError(f'{name} must contain distinct nonempty decision IDs')
            for ref in refs:
                if ref not in known:
                    raise ValueError(f'Unknown or forward decision reference: {ref}')
                if name == 'depends_on_decision_ids' and known[ref]['status'] != 'completed':
                    raise ValueError(f'Dependency must be completed: {ref}')
            result[name] = tuple(refs)
        return result

    def _parse_decision(self, response, context):
        payload = _load_object(response.text)
        kind = payload["kind"]
        if context.research_agenda.get('memory_consolidation') and (
                kind != 'invoke' or payload.get('operation') not in {spec.name for spec in context.operations}):
            raise ValueError('Consider pending Memory evidence before starting another experiment; '
                             'write a source-cited note or record a reasoned memory_review disposition')
        reason = self._parse_reason(payload)
        lineage = self._parse_lineage(payload, context)
        attempt = self._parse_attempt(payload, context)
        refs = payload.get('feedback_proposal_ids', [])
        if not isinstance(refs, list):
            raise ValueError('feedback_proposal_ids must be a list')
        metadata = DecisionResearchMetadata(**attempt, feedback_proposal_ids=tuple(refs))
        common = {'kind', 'reason', 'depends_on_decision_ids', 'source_decision_ids',
                  'research_attempt_id', 'research_attempt_title', 'feedback_proposal_ids'}
        if kind in ('invoke', 'wait', 'finish'):
            allowed = common | {'invoke': {'operation', 'arguments'}, 'wait': {'job'}, 'finish': set()}[kind]
            if set(payload) - allowed:
                raise ValueError('Unexpected decision fields')
        if kind == "invoke":
            decision = Decision.invoke(OperationCall(payload["operation"], payload["arguments"]), reason=reason, simulated=response.simulated, usage=response.usage, **lineage)
        elif kind == "wait":
            decision = Decision("wait", job=JobRef(**payload["job"]) if payload.get("job") is not None else None,
                            reason=reason, simulated=response.simulated, usage=response.usage, **lineage)
        elif kind == "finish":
            decision = Decision("finish", reason=reason, simulated=response.simulated, usage=response.usage, **lineage)
        else:
            raise ValueError(f"Unsupported decision kind: {kind}")
        self._decision_metadata = (decision.decision_id, metadata)
        return decision

    def decision_metadata(self, decision_id):
        """Return only the exact parsed decision's immutable research request.

        Async adapters may issue independent wait decisions while generation runs.
        Those decisions must never inherit another decision's research grouping.
        The host snapshots this request during admission before any operation.
        """
        current = self._decision_metadata
        return current[1] if current is not None and current[0] == decision_id else DecisionResearchMetadata()



    def state(self):
        """StatefulPolicy: the model binding is the only state that changes after construction."""
        return {'model': self.model}

    def restore(self, state):
        self.update_model(state['model'])
        self._decision_metadata = None

    def update_model(self, model: ModelRef) -> None:
        if model.backend_id != self.model.backend_id:
            raise ValueError("Changing model backends requires explicit policy reconfiguration")
        self.model = model
