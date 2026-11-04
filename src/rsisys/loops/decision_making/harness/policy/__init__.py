from dataclasses import asdict, replace
from contextlib import contextmanager
from copy import deepcopy
import json
from inspect import getattr_static
from typing import Protocol
from uuid import uuid4

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.coordination import ReviewDecision
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef, ModelBackend
from ...action_space.directives import DirectiveDefinitions

from ..decision import Decision
from ..attempts import requested_attempt, known_attempts, AttemptContract
from ...action_space import OperationCall
from ..state import DecisionContext
from ..context import BasicContextCompactor, IdentityContextCompactor
from .prompts import DecisionPrompts


from rsisys.runtime.json_tools import load_object as _load_object


class DecisionPolicy(Protocol):
    def decide(self, context: DecisionContext) -> Decision: ...


class ModelDecisionPolicy:
    """Example JSON parser. Provider response parsing stays in policy."""

    def __init__(self, backend: ModelBackend, model: ModelRef, *, allowed_actions: tuple[str, ...] = ("echo",),
                 directives: DirectiveDefinitions | None = None, max_validation_attempts: int = 1,
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
        self.allowed_actions = allowed_actions
        self.directives = directives if directives is not None else DirectiveDefinitions()
        self._context_compactor = context_compactor if context_compactor is not None else BasicContextCompactor()
        self._explicit_compactor = context_compactor is not None
        self._context_runtime = None
        self.context_token_count_mode = "unmeasured"
        self._usage_recorder = None
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

    def _research_instructions(self):
        return self.prompts.render('research') + ' '

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
        for spec in self.directives.describe(context.directive_modes):
            input_ids = {'type': 'array', 'items': {'type': 'string'}}
            if not spec.accepts_input_refs:
                input_ids['maxItems'] = 0
            branches.append(self._envelope({
                'kind': {'const': 'explore'}, 'mode': {'const': spec.mode},
                'strategy': {'type': 'string'},
                'arguments': self._input_schema(spec.input_schema, f'directive {spec.mode!r}'),
                'input_ids': input_ids,
                'review_after_seconds': {'type': ['number', 'null'], 'exclusiveMinimum': 0},
            }, ('kind', 'mode', 'strategy', 'arguments')))
        branches.extend((
            self._envelope({'kind': {'const': 'wait'}, 'job': {
                'type': ['object', 'null'],
                'properties': {'job_id': {'type': 'string'}, 'backend': {'type': 'string'}},
                'required': ['job_id', 'backend'], 'additionalProperties': False,
            }}, ('kind',)),
            self._envelope({'kind': {'const': 'finish'}}, ('kind',)),
        ))
        return {'oneOf': self._attempt_branches(branches, context)}

    def review_schema(self, context: DecisionContext) -> dict:
        """Expose permitted review queries and controls, never new directives."""
        branches = self._operation_branches(context)
        branches.extend(self._envelope({
            'action': {'const': action},
            'next_review_seconds': {'type': ['number', 'null'], 'exclusiveMinimum': 0},
        }, ('action',)) for action in context.review_actions)
        # An empty union is unsatisfiable; JSON Schema requires nonempty oneOf.
        return {'oneOf': self._attempt_branches(branches, context)} if branches else {'not': {}}

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
        inputs["directives"] = [asdict(spec) for spec in self.directives.describe(context.directive_modes)]
        return GenerationRequest(
            "decision_maker", self.model, json.dumps(inputs, separators=(',', ':')),
            system=self.prompts.render('decision', scheduling=self._scheduling_instructions(context),
                control=self._control_instructions(context), attempt=self._attempt_instructions(),
                research=self._research_instructions()),
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
        feedback = asdict(context.last_feedback) if context.last_feedback else None
        return {
            'candidate_evaluations': [asdict(row) for row in context.candidate_evaluations],
            'ledger_archives': deepcopy(context.research_agenda.get('ledger_archives', {})),
            'state_scope': 'Working selection; older evaluations and adoptions may be archived. '
                           'Missing records never establish a negative claim about the full history.',
            'last_feedback': None if feedback is None else {
                key: feedback[key] for key in ('directive_id', 'status', 'simulated',
                    'quality_metrics', 'training_metrics', 'execution_metrics',
                    'evidence_refs', 'adopted_candidate_ids')},
            'confirmed_adoptions': [
                {'decision_id': row['decision_id'], 'candidate_ids': row['adopted_candidate_ids']}
                for row in context.decision_graph if row.get('adopted_candidate_ids')],
            'execution_model': asdict(context.execution.model) if context.execution.model else None,
            'outstanding_directive_ids': [row.directive_id for row in context.outstanding],
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
        from ..ledger_context import working_ledger_context
        from ..context import with_working_decision_graph
        # Prompt selection must never shrink the context used by the parser.
        context = with_working_decision_graph(working_ledger_context(context))
        compactor, recent_limit = (self._context_runtime if self._context_runtime is not None
                                  else (self.context_compactor, 32))
        if isinstance(compactor, BasicContextCompactor):
            projected = compactor.compact(context, recent_limit=recent_limit,
                measure=lambda view: self._count_request(build(view)),
                summarize=lambda records, previous, limit: self._summarize_context(
                    records, previous, limit, current_state=self._summary_current_state(context)))
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
        refs = payload.get('feedback_proposal_ids', [])
        if (not isinstance(refs, list) or len(refs) > 16
                or any(not isinstance(ref, str) or not ref.strip() or len(ref) > 512 for ref in refs)
                or len(set(refs)) != len(refs)):
            raise ValueError('feedback_proposal_ids must contain distinct proposal IDs')
        result['feedback_proposal_ids'] = tuple(refs)
        return result

    def _parse_decision(self, response, context):
        payload = _load_object(response.text)
        kind = payload["kind"]
        reason = self._parse_reason(payload)
        lineage = self._parse_lineage(payload, context)
        lineage.update(self._parse_attempt(payload, context))
        common = {'kind', 'reason', 'depends_on_decision_ids', 'source_decision_ids',
                  'research_attempt_id', 'research_attempt_title', 'feedback_proposal_ids'}
        if kind in ('invoke', 'wait', 'finish'):
            allowed = common | {'invoke': {'operation', 'arguments'}, 'wait': {'job'}, 'finish': set()}[kind]
            if set(payload) - allowed:
                raise ValueError('Unexpected decision fields')
        if kind == "invoke":
            return Decision("invoke", operation=OperationCall(payload["operation"], payload["arguments"]),
                            reason=reason, simulated=response.simulated, usage=response.usage, **lineage)
        if kind == "explore":
            mode = payload.get('mode', 'execute')
            if mode not in context.directive_modes:
                raise ValueError(f'Disallowed exploration mode: {mode}')
            fields = {key: value for key, value in payload.items()
                      if key not in ('kind', 'strategy', 'mode', 'input_ids', 'arguments', 'review_after_seconds', 'reason',
                                     'depends_on_decision_ids', 'source_decision_ids',
                                     'research_attempt_id', 'research_attempt_title', 'feedback_proposal_ids')}
            if 'arguments' in payload and fields:
                raise ValueError('Use either arguments or top-level directive parameters, not both')
            candidates = {c.candidate_id: c for c in context.candidates}
            if context.last_feedback:
                for candidate in context.last_feedback.candidates:
                    candidates.setdefault(candidate.candidate_id, candidate)
            request = self.directives.parse(mode, payload.get('arguments', fields),
                candidates=tuple(candidates.values()))
            accepts_inputs = self.directives.describe((mode,))[0].accepts_input_refs
            if not accepts_inputs and payload.get('input_ids', []) != []:
                raise ValueError(f'{mode} does not accept input_ids; use [] or omit the field')
            sources = tuple((ref, result.simulated) for result in context.operation_results
                            for ref in result.artifacts) if accepts_inputs else ()
            if 'input_ids' in payload:
                available = {ref.artifact_id: (ref, simulated) for ref, simulated in sources}
                if context.last_feedback:
                    available.update({ref.artifact_id: (ref, context.last_feedback.simulated)
                                      for ref in context.last_feedback.artifact_refs})
                ids = payload['input_ids']
                if not isinstance(ids, list) or any(not isinstance(i, str) or i not in available for i in ids):
                    raise ValueError('input_ids must identify available artifacts')
                sources = tuple(available[i] for i in ids)
            directive = Directive(
                directive_id=str(uuid4()), objective=context.objective, strategy=payload["strategy"],
                mode=mode, payload=request,
                allowed_actions=self.allowed_actions,
                input_refs=tuple(ref for ref, _ in sources),
                simulated=response.simulated or any(simulated for _, simulated in sources),
                review_after_seconds=payload.get('review_after_seconds'),
            )
            return Decision("explore", directive=directive, reason=reason,
                            simulated=response.simulated, usage=response.usage, **lineage)
        if kind == "wait":
            return Decision("wait", job=JobRef(**payload["job"]) if payload.get("job") is not None else None,
                            reason=reason, simulated=response.simulated, usage=response.usage, **lineage)
        if kind == "finish":
            return Decision("finish", reason=reason, simulated=response.simulated, usage=response.usage, **lineage)
        raise ValueError(f"Unsupported decision kind: {kind}")

    def _review_request(self, context):
        """Use externally permitted operations or advise on the active execution."""
        model = self.model
        return GenerationRequest(
            "decision_maker", model, json.dumps(self._context_inputs(context)),
            system=self.prompts.render('review', attempt=self._attempt_instructions(),
                research=self._research_instructions()),
            **({'response_schema': self.review_schema(context)} if self.structured_output else {}),
        )

    def review(self, context: DecisionContext) -> ReviewDecision | Decision:
        model = self.model
        response = self._generate_context_request(context, self._review_request)
        self._received(response)
        if response.model != model:
            raise ValueError('Model response identity does not match the requested version')
        payload = _load_object(response.text)
        if isinstance(payload, dict) and payload.get('kind') == 'invoke':
            required = {'kind', 'operation', 'arguments'}
            if not required <= set(payload) or set(payload) - required - {
                    'reason', 'depends_on_decision_ids', 'source_decision_ids',
                    'research_attempt_id', 'research_attempt_title', 'feedback_proposal_ids'}:
                raise ValueError('A review invocation requires kind, operation and arguments, with optional reason')
            allowed = {spec.name for spec in context.operations}
            if (not isinstance(payload['operation'], str) or payload['operation'] not in allowed
                    or not isinstance(payload['arguments'], dict)):
                raise ValueError('Review may invoke only an available operation')
            return Decision('invoke', operation=OperationCall(payload['operation'], payload['arguments']),
                            reason=self._parse_reason(payload), simulated=response.simulated, usage=response.usage,
                            **self._parse_lineage(payload, context), **self._parse_attempt(payload, context))
        if (not isinstance(payload, dict) or 'action' not in payload
                or set(payload) - {'action', 'next_review_seconds', 'reason'}):
            raise ValueError('Expected a review action, optional next_review_seconds and reason')
        if payload['action'] not in context.review_actions:
            raise ValueError('Review control is not currently permitted')
        return ReviewDecision(action=payload['action'], next_review_seconds=payload.get('next_review_seconds'),
                              reason=self._parse_reason(payload), simulated=response.simulated, usage=response.usage)

    def update_model(self, model: ModelRef) -> None:
        if model.backend_id != self.model.backend_id:
            raise ValueError("Changing model backends requires explicit policy reconfiguration")
        self.model = model
