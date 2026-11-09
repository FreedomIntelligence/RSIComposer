"""Optional candidate helpers; algorithms remain replaceable. No eager ML imports."""
from collections.abc import Mapping
import operator


def token_ids(value):
    """Normalize tokenizer lists/BatchEncoding; never iterate dictionary keys."""
    if isinstance(value, Mapping):
        value = value.get('input_ids')
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError('Expected one tokenized example')
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('Tokenizer must return a nonempty input_ids sequence')
    try:
        if any(isinstance(v, bool) for v in value):
            raise TypeError
        values = [operator.index(v) for v in value]
    except TypeError:
        raise ValueError('Token IDs must be integers, not keys or strings') from None
    if any(v < 0 for v in values):
        raise ValueError('Input token IDs must be nonnegative')
    return values


def sft_example(tokenizer, messages, response, *, max_tokens=2048):
    if type(max_tokens) is not int or max_tokens < 2 or not isinstance(response, str) or not response.strip():
        raise ValueError('Require a response and max_tokens >=2')
    # Render text first so return_dict/tokenize API differences do not silently
    # turn input_ids into string keys. This baseline explicitly disables thinking.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full = tokenizer.apply_chat_template([*messages, {'role': 'assistant', 'content': response}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix = token_ids(tokenizer(prompt, add_special_tokens=False))
    ids = token_ids(tokenizer(full, add_special_tokens=False))
    if ids[:len(prefix)] != prefix:
        raise ValueError('Chat template is not prefix-stable; inspect the assistant mask for this tokenizer')
    ids = ids[:max_tokens]
    if len(prefix) >= len(ids):
        raise ValueError('Truncation removed every supervised response token')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prefix) + ids[len(prefix):]}


def padded_examples(batch, pad_token_id):
    if not batch or type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError('Require a nonempty batch and integer pad token ID')
    size = max(len(row['input_ids']) for row in batch)
    result = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for row in batch:
        ids = token_ids(row['input_ids'])
        labels = row['labels']
        if (len(labels) != len(ids) or any(type(v) is not int or v < -100 for v in labels)
                or any(v < 0 and v != -100 for v in labels) or not any(v != -100 for v in labels)):
            raise ValueError('Aligned integer labels must include supervised tokens')
        pad = size - len(ids)
        result['input_ids'].append(ids + [pad_token_id] * pad)
        result['attention_mask'].append([1] * len(ids) + [0] * pad)
        result['labels'].append(list(labels) + [-100] * pad)
    return result


def tensor_collator(pad_token_id):
    def collate(batch):
        import torch
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in padded_examples(batch, pad_token_id).items()}
    return collate
