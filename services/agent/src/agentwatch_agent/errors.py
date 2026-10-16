"""Run-ending conditions, each carrying a redactable reason for the live run."""


class RunCancelled(Exception):
    """Cancellation was requested; work stops and unfinished steps end as cancelled."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RunFailure(Exception):
    """The run cannot finish successfully. `reason` is shared live; `detail` stays local."""

    def __init__(self, reason: str, detail: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class RunNotStarted(Exception):
    """The live run could not be registered, so no task work was started."""
