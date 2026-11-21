"""Execution failures usable without optional execution backends installed."""


class WorkerLostError(RuntimeError):
    """A remote worker disappeared; its operation's effects are uncertain."""


class WorkerLaunchUncertainError(RuntimeError):
    """Submission crossed the executor boundary without a confirmed outcome."""
