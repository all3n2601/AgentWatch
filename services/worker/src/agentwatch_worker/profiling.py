"""Evidence-based CPU queue attribution for the first local experiment."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StepMeasurement:
    step_id: str
    submitted_ns: int
    started_ns: int
    finished_ns: int
    cpu_seconds: float
    checksum: str

    @property
    def queue_seconds(self) -> float:
        return (self.started_ns - self.submitted_ns) / 1e9

    @property
    def execution_seconds(self) -> float:
        return (self.finished_ns - self.started_ns) / 1e9

    @property
    def duration_seconds(self) -> float:
        return (self.finished_ns - self.submitted_ns) / 1e9


def attribute_cpu_queue(step: StepMeasurement, threshold_seconds: float) -> dict:
    """Use measured queue delay, without access to the injection configuration.

    This identifies contention for sandbox worker capacity, not host CPU saturation.
    Execution wall time may itself include scheduling delays; it is not 'pure work'.
    """
    if not step.submitted_ns <= step.started_ns < step.finished_ns:
        raise ValueError("Step timestamps must be ordered with positive execution time")
    if threshold_seconds < 0:
        raise ValueError("Queue threshold must be nonnegative")
    return {
        **asdict(step),
        "duration_seconds": step.duration_seconds,
        "queue_seconds": step.queue_seconds,
        "execution_seconds": step.execution_seconds,
        "queue_fraction": step.queue_seconds / step.duration_seconds,
        "resource": "cpu_sandbox" if step.queue_seconds > threshold_seconds else "none",
        "evidence": "direct worker enqueue/start timestamps",
        "rule_version": "cpu-queue-v1",
        "threshold_seconds": threshold_seconds,
    }
