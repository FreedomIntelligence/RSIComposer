"""Stable JSON representation for decision records."""
import json

def _json_details(value):
    """Match MemoryBuffer's JSON load representation before the first write."""
    return json.loads(json.dumps(value))

