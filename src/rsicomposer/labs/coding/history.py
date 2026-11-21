"""Bound worker history without duplicating the immutable task prefix."""

OMITTED = ('Earlier tool interactions were omitted for context capacity. '
           'Files persist in the container. Reinspect any details needed.')


def bounded_history(messages, max_chars=120000, recent_messages=18, max_bytes=48000):
    """Keep the original system/task pair and a bounded recent suffix.

    UTF-8 bytes give a conservative token bound for the deployed byte-level BPE
    tokenizer. A character limit alone cannot bound random data or Unicode.
    The default leaves room in the 65536-token context for output and framing.
    Original messages remain unchanged for complete on-disk trace retention.
    """
    def sizes(content):
        return len(content), len(content.encode('utf-8'))

    chars = sum(len(m['content']) for m in messages)
    byte_count = sum(len(m['content'].encode('utf-8')) for m in messages)
    if chars <= max_chars and byte_count <= max_bytes:
        return messages
    prefix = messages[:2]
    available_chars = max_chars - sum(len(m['content']) for m in prefix) - len(OMITTED)
    available_bytes = max_bytes - sum(len(m['content'].encode('utf-8')) for m in prefix) - len(OMITTED.encode('utf-8'))
    if min(available_chars, available_bytes) < 1:
        raise ValueError('Task and strategy exceed the worker history budget')
    suffix = []
    for message in reversed(messages[2:][-recent_messages:]):
        size_chars, size_bytes = sizes(message['content'])
        if size_chars > available_chars or size_bytes > available_bytes:
            if not suffix:
                tail = message['content'][-available_chars:]
                tail = tail.encode('utf-8')[-available_bytes:].decode('utf-8', errors='ignore')
                suffix.append(dict(message, content=tail))
            break
        suffix.append(message)
        available_chars -= size_chars
        available_bytes -= size_bytes
    return prefix + [{'role': 'user', 'content': OMITTED}] + list(reversed(suffix))
