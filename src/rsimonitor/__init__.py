"""Reusable, read-only experiment monitoring from public monitor.json snapshots."""

__all__ = ["RunStore", "MonitorServer", "project_snapshot"]


def __getattr__(name):
    # Keep `python -m rsimonitor.server` from preloading its target module.
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
