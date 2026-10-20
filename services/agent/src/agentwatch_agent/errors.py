"""Run-ending conditions, each carrying a redactable reason for the live run."""


class RunCancelled(Exception):
    """Cancellation was requested; work stops and unfinished steps end as cancelled."""

    # False when raised before a blocking request was started, so nothing was abandoned.
    request_started = True

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RunFailure(Exception):
    """The run cannot finish successfully. `reason` is shared live; `detail` stays local."""

    request_started = True

    def __init__(self, reason: str, detail: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class RunNotStarted(Exception):
    """No task work started: local setup or live-run registration failed.

    `stage` is "setup" for local failures before the live API is contacted, and
    "registration" when the live API could not register the run.
    """

    def __init__(self, message: str, stage: str, closure_journaled: bool = False):
        super().__init__(message)
        self.stage = stage
        # True when the registration outcome is unknown and a closure was journaled, so
        # `resend` closes the run if the service did record it.
        self.closure_journaled = closure_journaled
