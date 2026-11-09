"""Candidate-side, standard-library client for the fixed-model Unix broker.

Only this file is made available to generated programs, never the host Lab.
"""
import json
import os
import socket


def complete(messages, *, max_tokens=1024, temperature=0.0, seed=73):
    request = {'messages': messages, 'max_tokens': max_tokens,
               'temperature': temperature, 'seed': seed}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(300)
        client.connect(os.environ['AIDE_LLM_SOCKET'])
        client.sendall(json.dumps(request).encode() + b'\n')
        with client.makefile('rb') as stream:
            result = json.loads(stream.readline(2_000_000))
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result['content']
