"""Standard candidate input boundary, shared by diagnosis and formal evaluation.

This is an observable adapter in trusted candidate code, not backend attestation
or a security boundary. The candidate still owns its model and inference code.
Only this module reads the protocol and assigns prediction IDs.
"""
from copy import deepcopy
import hashlib
import json
import sys


CONTRACT = 'medical-input-adapter-v1'
BINDING = 'adapter_boundary_only; backend_input_consumption_not_attested'


class CandidateInputError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def user_contents(messages):
    if (not isinstance(messages, list) or not messages
            or any(not isinstance(m, dict) or set(m) != {'role', 'content'}
                   or m['role'] not in ('system', 'user', 'assistant')
                   or not isinstance(m['content'], str) for m in messages)):
        raise CandidateInputError('Expected complete role/content messages, not messages[0] or a bare string')
    users = [m['content'] for m in messages if m['role'] == 'user']
    if not users or any(not text.strip() for text in users):
        raise CandidateInputError('Missing user question/options: messages[0] is the system instruction')
    return users


class CandidateInput:
    """One fixed sample. Use render() locally or chat_messages() for a service.

    Reading .messages alone does not record a prepared input. Batch inference can
    prepare each request, generate together, and return answers in request order.
    """
    def __init__(self, item):
        self.benchmark_id, self.id = item['benchmark_id'], item['id']
        self._messages = deepcopy(item['messages'])
        self._users = user_contents(self._messages)
        self._receipt = None

    @property
    def messages(self):
        return deepcopy(self._messages)

    def _prepare(self, messages):
        messages = self.messages if messages is None else deepcopy(messages)
        actual_users = user_contents(messages)
        if any(not any(text in actual for actual in actual_users) for text in self._users):
            raise CandidateInputError('Prepared user messages omit or alter the fixed question/options')
        return messages

    def _record(self, messages, kind, prepared):
        self._receipt = {'benchmark_id': self.benchmark_id, 'id': self.id,
            'contract': CONTRACT, 'binding': BINDING, 'kind': kind,
            'source_messages_sha256': digest(self._messages),
            'prepared_messages_sha256': digest(messages),
            'roles': [m['role'] for m in messages],
            'fixed_user_sha256': [digest(text) for text in self._users],
            'prepared_sha256': digest(prepared),
            'prepared_characters': len(prepared) if isinstance(prepared, str) else None}

    def chat_messages(self, *, messages=None):
        """Return the complete messages to place in the actual HTTP request."""
        prepared = self._prepare(messages)
        self._record(prepared, 'chat_messages', prepared)
        return prepared

    def render(self, tokenizer, *, messages=None, **options):
        """Return the exact text to tokenize; reject templates that drop the user."""
        if options.get('tokenize', False) is not False:
            raise CandidateInputError('render returns text; tokenize that returned text for inference')
        prepared = self._prepare(messages)
        options = {'add_generation_prompt': True, **options, 'tokenize': False}
        text = tokenizer.apply_chat_template(prepared, **options)
        if not isinstance(text, str) or any(user not in text for user in self._users):
            raise CandidateInputError('Rendered inference input omits the fixed user question/options')
        self._record(prepared, 'rendered_prompt', text)
        return text

    def receipt(self):
        if self._receipt is None:
            raise CandidateInputError('No prepared input recorded. Use request.render(tokenizer, ...) '
                'for local inference or request.chat_messages() for the actual service request')
        return deepcopy(self._receipt)


def validate_receipts(receipts, payload):
    items = payload['items']
    if not isinstance(receipts, list) or len(receipts) != len(items):
        raise CandidateInputError('Exactly one adapter input receipt is required per fixed sample')
    expected = {(i['benchmark_id'], i['id']): i for i in items}
    seen = set()
    for row in receipts:
        if not isinstance(row, dict):
            raise CandidateInputError('Malformed adapter input receipt')
        key = (row.get('benchmark_id'), row.get('id'))
        item = expected.get(key)
        if item is None or key in seen:
            raise CandidateInputError('Unknown or duplicate adapter input receipt')
        seen.add(key)
        if (row.get('contract') != CONTRACT or row.get('binding') != BINDING
                or row.get('kind') not in ('chat_messages', 'rendered_prompt')
                or row.get('source_messages_sha256') != digest(item['messages'])
                or row.get('fixed_user_sha256') != [digest(t) for t in user_contents(item['messages'])]
                or 'user' not in row.get('roles', [])
                or any(not isinstance(row.get(k), str) or len(row[k]) != 64
                       for k in ('prepared_messages_sha256', 'prepared_sha256'))):
            raise CandidateInputError('Adapter receipt does not match the fixed sample input')
    return {'contract': CONTRACT, 'status': 'adapter_verified', 'binding': BINDING,
        'samples': len(items), 'receipts': deepcopy(receipts)}


def predict_payload(payload, infer=None, *, infer_batch=None):
    requests = [CandidateInput(item) for item in payload['items']]
    if infer_batch is not None:
        answers = infer_batch(requests)
    elif infer is not None:
        answers = []
        for request in requests:
            answer = infer(request)
            request.receipt()  # Reject a missing boundary after the first sample.
            answers.append(answer)
    else:
        raise CandidateInputError('Supply infer(request) or infer_batch(requests)')
    if not isinstance(answers, list) or len(answers) != len(requests) or any(
            not isinstance(text, str) for text in answers):
        raise CandidateInputError('Return one answer string per input request, in input order')
    receipts = [request.receipt() for request in requests]
    validate_receipts(receipts, payload)
    return {'predictions': [{'benchmark_id': request.benchmark_id, 'id': request.id, 'text': text}
                           for request, text in zip(requests, answers)], 'input_receipts': receipts}


def run(infer=None, *, infer_batch=None):
    print(json.dumps(predict_payload(json.load(sys.stdin), infer, infer_batch=infer_batch),
                     ensure_ascii=False), flush=True)
