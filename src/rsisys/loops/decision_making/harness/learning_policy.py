"""Optional memory-prefix ordering for the existing JSON decision policy."""

from dataclasses import replace
import json

from .policy import ModelDecisionPolicy


class MemoryPrefixDecisionPolicy(ModelDecisionPolicy):
    """Place stable capabilities and memory before changing turn observations.

    Caching is backend-dependent: byte stability is not a measurement of KV cache
    hits. Memory stays untrusted user content in the prompt; the inherited system
    instructions, schemas and generation settings retain their authority.

    The learning harness assembles research_state before these builders. They
    only order the request; inherited compaction prices the complete request,
    including research state, capabilities and feedback. No model call is added.
    """

    _prefix_fields = (
        'objective', 'project_context', 'directives', 'operations', 'directive_modes',
        'research_state', 'memories', 'context_summary',
    )

    @classmethod
    def _ordered_request(cls, request):
        payload = json.loads(request.prompt)
        fields = [name for name in cls._prefix_fields if name in payload]
        fields.extend(name for name in payload if name not in cls._prefix_fields)
        # Canonicalize each value recursively, never sort the top-level object.
        # Preserve array order (including capability and schema-required lists)
        # and keep all remaining fields in the superclass's relative order.
        prompt = '{' + ','.join(
            json.dumps(name) + ':' + json.dumps(
                payload[name], sort_keys=True, separators=(',', ':'))
            for name in fields
        ) + '}'
        return replace(request, prompt=prompt)

    def _decision_request(self, context):
        return self._ordered_request(super()._decision_request(context))

    def _review_request(self, context):
        return self._ordered_request(super()._review_request(context))
