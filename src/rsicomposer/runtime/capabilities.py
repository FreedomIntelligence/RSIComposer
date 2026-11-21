"""Optional declared methods, including real methods forwarded by wrappers."""
from inspect import getattr_static, isroutine


def optional_method(component, name):
    """Ignore fabricated callable objects; accept explicitly forwarded methods.

    A dynamic proxy forwarding a bound method preserves a real capability. A
    catch-all mock/RPC object manufacturing arbitrary callable attributes does
    not declare every optional capability in the system.
    """
    value = getattr(component, name, None)
    if callable(value) and (getattr_static(component, name, None) is not None or isroutine(value)):
        return value
    return None
