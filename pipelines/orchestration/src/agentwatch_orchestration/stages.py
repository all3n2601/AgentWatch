"""Run the dataset pipeline's existing commands as orchestrated stages.

Every stage runs the same module the matching make target runs, in that stage's own
environment (the shared workspace, the TFDV gate's, or DVC's), so orchestration adds ordering,
retries, and logs without a second implementation of any stage. The commands mirror the
Makefile's dataset targets; keep the two in step.

Stages pass the release they built to the next stage explicitly instead of asking for the
newest release, so a release published by someone else between two stages is never validated,
split, or versioned in this run's place.

This module does not import Airflow, so it is tested without it.
"""

import json
import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# src/agentwatch_orchestration/stages.py -> the repository root
DEFAULT_ROOT = Path(__file__).resolve().parents[4]
# Generous for today's data (a build takes seconds); a hung command must still end the task.
DEFAULT_STAGE_TIMEOUT_SECONDS = 30 * 60
STDERR_TAIL_LINES = 20  # how much of a failed command's stderr the error carries

# Prefixes of the lines the data commands print for later stages to read.
ADDED_PREFIX = "added: "
RELEASE_PREFIX = "release: "
REPORT_PREFIX = "report: "
SPLIT_PREFIX = "split: "
LEAKAGE_PREFIX = "leakage: "
RESULT_PREFIX = "result: "
REUSED_SUFFIX = " (reused)"
# dvc remote default exits 1 with this message when no default remote is set; any other failure
# is a broken configuration or environment, which must fail the push rather than skip it.
NO_REMOTE_EXIT_CODE = 1
NO_REMOTE_MESSAGE = "No default remote set"


class StageError(RuntimeError):
    """A stage command failed, timed out, or printed output the next stage cannot use."""


@dataclass(frozen=True)
class PipelineConfig:
    """Where the stores, contracts, and tools are. Paths are relative to root unless absolute."""

    root: Path = DEFAULT_ROOT
    evidence: Path = Path("data/evidence")
    datasets: Path = Path("data/datasets")
    schema: Path = Path("pipelines/validation/schema/steps.pbtxt")
    contract: Path = Path("pipelines/model/evaluation-contract.toml")
    api_url: str | None = None  # also build from runs saved through the ingestion API
    timeout_seconds: int = DEFAULT_STAGE_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "PipelineConfig":
        """Read AGENTWATCH_ROOT, AGENTWATCH_DATASET_API_URL, and AGENTWATCH_STAGE_TIMEOUT_SECONDS."""
        root = Path(env["AGENTWATCH_ROOT"]) if env.get("AGENTWATCH_ROOT") else DEFAULT_ROOT
        timeout_text = env.get("AGENTWATCH_STAGE_TIMEOUT_SECONDS")
        timeout_seconds = DEFAULT_STAGE_TIMEOUT_SECONDS
        if timeout_text:
            try:
                timeout_seconds = int(timeout_text)
            except ValueError as error:
                raise StageError(
                    f"AGENTWATCH_STAGE_TIMEOUT_SECONDS must be a whole number of seconds, "
                    f"got {timeout_text!r}"
                ) from error
            if timeout_seconds <= 0:
                raise StageError("AGENTWATCH_STAGE_TIMEOUT_SECONDS must be positive")
        if not (root / "Makefile").is_file():
            raise StageError(f"{root} is not an AgentWatch checkout; set AGENTWATCH_ROOT")
        return cls(
            root=root,
            api_url=env.get("AGENTWATCH_DATASET_API_URL") or None,
            timeout_seconds=timeout_seconds,
        )

    def path(self, relative: Path) -> Path:
        return relative if relative.is_absolute() else self.root / relative


@dataclass(frozen=True)
class BuildOutcome:
    release: str
    report: str
    rows: int
    runs: int  # runs that passed every stage and are in the release
    runs_rejected: int  # runs the build rejected individually, each with a reason in the report


@dataclass(frozen=True)
class SplitOutcome:
    path: str
    reused: bool
    leakage: str  # passed, or a blocked/inconclusive status with its reason


@dataclass(frozen=True)
class RunSummary:
    """What one pipeline run produced, so a run is traceable without reading its logs."""

    release: str
    build_report: str
    rows: int
    runs: int
    runs_rejected: int
    evidence_added: int
    split: str
    leakage: str
    drift: list[str]  # the gate's drift findings against the previous passing release
    versioned: bool
    pushed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def data_command(config: PipelineConfig, module: str, *args: str) -> list[str]:
    return [
        "uv", "run", "--project", str(config.root), "--package", "agentwatch-data",
        "python", "-m", f"agentwatch_data.{module}", *args,
    ]  # fmt: skip


def validation_command(config: PipelineConfig, *args: str) -> list[str]:
    project = config.root / "pipelines/validation"
    return ["uv", "run", "--project", str(project), "python", "-m", "agentwatch_validation", *args]


def dvc_command(config: PipelineConfig, *args: str) -> list[str]:
    project = config.root / "pipelines/versioning"
    return ["uv", "run", "--project", str(project), "--frozen", "dvc", *args]


def run_command(
    config: PipelineConfig, stage: str, argv: Sequence[str], check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one command from the repository root, logging its output for the task log."""
    logger.info("%s: %s", stage, " ".join(argv))
    # Airflow's own virtual environment must not leak into the stage's environment.
    env = {name: value for name, value in os.environ.items() if name != "VIRTUAL_ENV"}
    try:
        result = subprocess.run(
            argv,
            cwd=config.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise StageError(f"{stage} timed out after {config.timeout_seconds} s") from error
    except OSError as error:
        raise StageError(f"{stage} could not start {argv[0]}: {error}") from error
    for line in result.stdout.splitlines():
        logger.info("%s | %s", stage, line)
    for line in result.stderr.splitlines():
        logger.warning("%s | %s", stage, line)
    if check and result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-STDERR_TAIL_LINES:])
        raise StageError(f"{stage} failed with exit code {result.returncode}:\n{tail}")
    return result


def value_after(stage: str, output: str, prefix: str) -> str:
    """The value of the first line starting with prefix, which the next stage depends on."""
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    raise StageError(f"{stage} did not report a {prefix.strip()!r} line")


def acquire(config: PipelineConfig, run_dirs: Sequence[str]) -> int:
    """Add recorded runs to the evidence store. Returns how many were newly added.

    Adding evidence that is already stored does nothing, so a retried or repeated run is safe.
    """
    if not run_dirs:
        return 0
    # Run directories come from DAG parameters; one that looks like an option (--store=...,
    # -h) must never be parsed as one, so it is refused and the rest follow "--".
    options = [run_dir for run_dir in run_dirs if run_dir.startswith("-")]
    if options:
        raise StageError(f"acquire refused run directories that look like options: {options}")
    store = str(config.path(config.evidence))
    argv = data_command(config, "evidence", "--store", store, "--", *run_dirs)
    output = run_command(config, "acquire", argv).stdout
    return sum(line.startswith(ADDED_PREFIX) for line in output.splitlines())


def build(config: PipelineConfig) -> BuildOutcome:
    """Build (or reuse) the release of the current evidence, with its report's counts.

    Runs the build rejects do not fail it (each is recorded with a reason in the report), so
    the counts are carried forward for the run summary and alerts.
    """
    sources = ["--source", f"local:{config.path(config.evidence)}"]
    if config.api_url:
        sources += ["--source", f"api:{config.api_url}"]
    output = str(config.path(config.datasets))
    result = run_command(
        config, "build", data_command(config, "build", *sources, "--output", output)
    )
    release = value_after("build", result.stdout, RELEASE_PREFIX)
    report = value_after("build", result.stdout, REPORT_PREFIX)
    counts = read_build_report(config.path(Path(report)))
    return BuildOutcome(release=release, report=report, **counts)


def read_build_report(path: Path) -> dict[str, int]:
    """The row, run, and rejection counts of a build report the build just wrote."""
    try:
        report = json.loads(path.read_text())
        counts = {
            "rows": report["rows"],
            "runs": report["runs"],
            "runs_rejected": len(report["rejections"]),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StageError(f"build report {path} is unreadable: {error}") from error
    if not all(type(value) is int and value >= 0 for value in counts.values()):
        raise StageError(f"build report {path} has invalid counts: {counts}")
    return counts


def validate(config: PipelineConfig, release: str) -> list[str]:
    """Run the TFDV quality gate; anomalies fail the stage and stop the run.

    Drift does not fail the gate. Its findings are returned, one per line, for the run summary
    and alerts. They are read from the gate's durable result, not its printed lines, so a change
    to the printout cannot silently drop them.
    """
    schema = str(config.path(config.schema))
    output = run_command(
        config,
        "validate",
        validation_command(config, "validate", "--release", release, "--schema", schema),
    ).stdout
    return read_gate_drift(config.path(Path(value_after("validate", output, RESULT_PREFIX))))


def read_gate_drift(path: Path) -> list[str]:
    """The drift findings of a gate result the gate just wrote, as "feature: description"."""
    try:
        drift = json.loads(path.read_text())["drift"]
        lines = [f"{item['feature']}: {item['description']}" for item in drift]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StageError(f"gate result {path} is unreadable: {error}") from error
    return lines


def split(config: PipelineConfig, release: str) -> SplitOutcome:
    """Freeze the release's splits. A blocked leakage check is kept as evidence, not a failure."""
    argv = data_command(
        config, "splits",
        "--release", release,
        "--contract", str(config.path(config.contract)),
        "--frozen-schema", str(config.path(config.schema)),
    )  # fmt: skip
    output = run_command(config, "split", argv).stdout
    path = value_after("split", output, SPLIT_PREFIX)
    return SplitOutcome(
        path=path.removesuffix(REUSED_SUFFIX),
        reused=path.endswith(REUSED_SUFFIX),
        leakage=value_after("split", output, LEAKAGE_PREFIX),
    )


def version(config: PipelineConfig) -> None:
    """Verify both stores, then snapshot them with DVC, as make version-dataset does.

    Committing the updated data/*.dvc pointer files is left to a person or a later task with a
    commit identity; until then the version exists only in this checkout's DVC cache.
    """
    status_file = config.root / ".dvc/tmp/version-dataset-status.json"
    status = run_command(
        config,
        "version",
        dvc_command(config, "data", "status", "--granular", "--unchanged", "--json"),
    )
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(status.stdout)
    evidence = str(config.path(config.evidence))
    datasets = str(config.path(config.datasets))
    integrity = data_command(
        config,
        "integrity",
        "--evidence",
        evidence,
        "--datasets",
        datasets,
        "--dvc-status",
        str(status_file),
    )
    run_command(config, "version", integrity)
    run_command(config, "version", dvc_command(config, "add", evidence, datasets))


def remote_configured(config: PipelineConfig) -> bool:
    """Whether DVC has a default remote; without one there is nowhere to push.

    Only DVC's own "no default remote" answer means unconfigured. Any other failure raises, so
    the push task fails (and retries) instead of being skipped like a local-only setup.
    """
    argv = dvc_command(config, "remote", "default")
    result = run_command(config, "push", argv, check=False)
    if result.returncode == 0:
        return True
    output = result.stdout + result.stderr
    if result.returncode == NO_REMOTE_EXIT_CODE and NO_REMOTE_MESSAGE in output:
        return False
    tail = "\n".join(output.splitlines()[-STDERR_TAIL_LINES:])
    raise StageError(f"push could not read the DVC remote (exit code {result.returncode}):\n{tail}")


def push(config: PipelineConfig) -> None:
    run_command(config, "push", dvc_command(config, "push"))
