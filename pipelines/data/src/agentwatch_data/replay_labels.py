"""Labels for replay experiments, where each injected condition contends one or more resources.

Rules live in labels-v2.toml, frozen before any evaluation. Conditions come from the
experiment design and are used for labeling only; condition data is never a feature.
Only leaf steps of baseline, contended and clean-control replays are labeled, each against
the mean of its own usable baseline copies. Evidence that cannot be labeled is rejected
with one of REJECTION_CODES, never relabeled or dropped silently.

Usage: python -m agentwatch_data.replay_labels <experiment directory>
"""

import argparse
import dataclasses
import json
import logging
import math
import re
import statistics
import sys
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.extract import Rejection, canonical_sha256
from agentwatch_data.labels import (
    LabelingError,
    excess_fraction,
    require_keys,
    require_number,
    require_text,
)
from agentwatch_data.replay_evidence import (
    Experiment,
    Replay,
    ReplayEvidenceError,
    StepTiming,
    read_experiment,
)
from agentwatch_data.schema import MIXED, REPLAY_ROLES, RESOURCE_CLASSES

logger = logging.getLogger(__name__)

# --- Frozen rules -----------------------------------------------------------------

# Conditions that inject exactly one contention; each maps to one resource class.
SOLO_CONDITIONS = ("cpu", "inference", "retrieval")


@dataclass(frozen=True)
class ReplayLabelConfig:
    version: str
    stdev_multiplier: float
    min_seconds: float
    condition_resources: dict[str, str]
    fingerprint: str

    @property
    def label_version(self) -> str:
        return f"{self.version}+{self.fingerprint[:8]}"


def logic_source() -> str:
    """This module's source with normalized line endings, so checkouts agree."""
    return Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")


def fingerprint(rules: dict) -> str:
    """Identify the validated rules and the labeling code that applies them."""
    return canonical_sha256({"rules": rules, "logic": logic_source()})


def load_replay_config(path: Path | None = None) -> ReplayLabelConfig:
    source = path if path is not None else files("agentwatch_data").joinpath("labels-v2.toml")
    try:
        raw = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise LabelingError(f"Replay label rules {source} cannot be read: {error}") from error
    try:
        rules = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise LabelingError(f"Replay label rules are not valid TOML: {error}") from error
    require_keys(rules, "replay label rules", {"version", "noise", "condition_resources"})
    noise = require_keys(rules["noise"], "noise", {"stdev_multiplier", "min_seconds"})
    table = require_keys(rules["condition_resources"], "condition_resources", set(SOLO_CONDITIONS))
    contended_classes = set(RESOURCE_CLASSES) - {"none"}
    condition_resources = {}
    for condition in SOLO_CONDITIONS:
        name = f"condition_resources.{condition}"
        resource = require_text(table[condition], name)
        if resource not in contended_classes:
            raise LabelingError(f"{name} must be one of {sorted(contended_classes)}")
        condition_resources[condition] = resource
    version = require_text(rules["version"], "version")
    multiplier = require_number(noise["stdev_multiplier"], "noise.stdev_multiplier", positive=True)
    floor = require_number(noise["min_seconds"], "noise.min_seconds", positive=False)
    # Fingerprint the validated values, so equivalent spellings such as 3 and 3.0 agree.
    validated = {
        "version": version,
        "noise": {"stdev_multiplier": multiplier, "min_seconds": floor},
        "condition_resources": condition_resources,
    }
    return ReplayLabelConfig(
        version, multiplier, floor, condition_resources, fingerprint(validated)
    )


# --- Labeling ---------------------------------------------------------------------

REJECTION_CODES = (
    "experiment_not_completed",
    "protocol_checks_failed",
    "replay_not_succeeded",
    "injection_inactive",
    "resource_class_mismatch",
    "step_not_completed",
    "output_differs_from_source",
    "baseline_reference_missing",
    "mixed_solo_evidence_missing",
)
# Protocol-validity checks: when one fails (or was not recorded) no row of the experiment is
# trustworthy. They are not rule outcomes, so rejecting on them does not select data by its
# label. Every other failed check is recorded in failed_checks and does not block labels.
BLOCKING_CHECKS = (
    "baseline_replays_complete",
    "stable_baseline",
    "same_steps_every_replay",
    "injectors_cleaned_up",
)
CLEAN, MIXED_CONDITION = "clean", "mixed"
BASELINE, CONTENDED, CLEAN_CONTROL = REPLAY_ROLES
NO_INJECTION = "none"
MIN_BASELINE_COPIES = 2
REVISION_SUFFIX = re.compile(r"-r\d+$")
ROW_KEYS = (
    "experiment",
    "task_id",
    "task_group",
    "plan_sha256",
    "agent_model",
    "run_id",
    "replay_index",
    "role",
    "condition",
    "step_id",
    "step_type",
    "duration_seconds",
    "queue_seconds",
    "execution_seconds",
    "label_resource",
    "label_resources",
    "label_excess_fraction",
    "label_basis",
    "injection_method",
    "baseline_mean_seconds",
    "noise_threshold_seconds",
    "control_exceeded_noise",
    "failed_checks",
    "label_version",
)


class UnlabelableError(Exception):
    """Evidence that cannot be labeled; the message is a reason starting with its code."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in REJECTION_CODES:
            raise ValueError(f"Unknown rejection code {code!r}")
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class Reference:
    mean_seconds: float
    threshold_seconds: float

    def exceeded(self, step: StepTiming) -> bool:
        """Strictly above the threshold: a slowdown exactly at it is noise."""
        if step.total_seconds is None:
            raise ValueError("exceeded() needs a usable step")
        return step.total_seconds - self.mean_seconds > self.threshold_seconds


@dataclass(frozen=True)
class Outcome:
    resources: tuple[str, ...]  # sorted classes the label names
    basis: str
    injection_method: str


def task_group(task_id: str) -> str:
    """The task name without its `-r<revision>` suffix, so revisions share a split."""
    return REVISION_SUFFIX.sub("", task_id)


def step_problem(step: StepTiming) -> UnlabelableError | None:
    """Why a step cannot be labeled or serve as evidence, or None when it is usable."""
    total = step.total_seconds
    if step.state != "completed" or total is None or not math.isfinite(total) or total <= 0:
        return UnlabelableError(
            "step_not_completed", f"state {step.state!r}, total_seconds {total!r}"
        )
    # matches_source is None for steps the replay evidence has no entry for (such as the
    # root step); only a recorded difference means the step did different work.
    if step.matches_source is False:
        return UnlabelableError(
            "output_differs_from_source", "the replayed output differs from the source run"
        )
    return None


def replay_problem(replay: Replay, config: ReplayLabelConfig) -> UnlabelableError | None:
    """Why no step of a replay can be labeled, or None when its design is intact."""
    if replay.state != "completed" or replay.task_passed is not True:
        return UnlabelableError(
            "replay_not_succeeded", f"state {replay.state!r}, task_passed {replay.task_passed!r}"
        )
    classes = [injector.resource_class for injector in replay.injectors]
    if replay.role != CONTENDED:
        if replay.condition != CLEAN or classes:
            return UnlabelableError(
                "resource_class_mismatch",
                f"{replay.role} replay has condition {replay.condition!r}, injectors {classes}",
            )
        return None
    if replay.injection_active is not True:
        return UnlabelableError(
            "injection_inactive", f"injection_active is {replay.injection_active!r}"
        )
    if replay.condition == MIXED_CONDITION:
        contended_classes = set(RESOURCE_CLASSES) - {"none"}
        if len(classes) < 2 or len(set(classes)) != len(classes):
            return UnlabelableError(
                "resource_class_mismatch",
                f"mixed replay needs two or more distinct injected classes, got {classes}",
            )
        if not set(classes) <= contended_classes:
            return UnlabelableError(
                "resource_class_mismatch", f"mixed replay injects unknown classes {classes}"
            )
        return None
    expected = config.condition_resources.get(replay.condition)
    if expected is None or classes != [expected]:
        return UnlabelableError(
            "resource_class_mismatch",
            f"condition {replay.condition!r} expects one {expected!r} injector, got {classes}",
        )
    return None


def baseline_references(baselines: list[Replay], config: ReplayLabelConfig) -> dict[str, Reference]:
    """Per-step mean and noise threshold over the usable baseline copies of each step."""
    copies: dict[str, list[float]] = {}
    for replay in baselines:
        for step_id, step in replay.steps.items():
            if step.total_seconds is not None and step_problem(step) is None:
                copies.setdefault(step_id, []).append(step.total_seconds)
    return {
        step_id: Reference(
            statistics.fmean(values),
            max(config.min_seconds, config.stdev_multiplier * statistics.stdev(values)),
        )
        for step_id, values in copies.items()
        if len(values) >= MIN_BASELINE_COPIES
    }


@dataclass(frozen=True)
class Context:
    """What labeling one step needs from the rest of the experiment."""

    config: ReplayLabelConfig
    references: dict[str, Reference]
    usable: dict[int, Replay]  # labelable replays by index
    solo_by_class: dict[str, list[Replay]]  # labelable solo contended replays


def mixed_outcome(
    replay: Replay, step: StepTiming, exceeded: bool, reference: Reference, context: Context
) -> Outcome:
    members = tuple(sorted(injector.resource_class for injector in replay.injectors))
    method = "+".join(sorted(injector.method for injector in replay.injectors))
    confirmed = []
    for member in members:
        solos = context.solo_by_class.get(member, [])
        if len(solos) != 1:
            raise UnlabelableError(
                "mixed_solo_evidence_missing",
                f"{len(solos)} labelable solo {member} replays, need exactly one",
            )
        solo_step = solos[0].steps.get(step.step_id)
        if solo_step is None or not solo_step.is_leaf or step_problem(solo_step) is not None:
            raise UnlabelableError(
                "mixed_solo_evidence_missing",
                f"solo {member} replay {solos[0].index:02d} has no usable copy of the step",
            )
        if reference.exceeded(solo_step):
            confirmed.append(member)
    if not exceeded:
        return Outcome((), "mixed_below_threshold", method)
    if len(confirmed) == len(members):
        return Outcome(members, "mixed_confirmed", method)
    if confirmed:
        return Outcome(tuple(confirmed), "mixed_partial", method)
    # Slowed only when every contention ran together: each was necessary, so all are named.
    return Outcome(members, "mixed_joint_only", method)


def step_outcome(
    replay: Replay, step: StepTiming, exceeded: bool, reference: Reference, context: Context
) -> Outcome:
    if replay.role == BASELINE:
        return Outcome((), "baseline_reference", NO_INJECTION)
    if replay.role == CLEAN_CONTROL:
        basis = "clean_exceeded_noise" if exceeded else "clean_within_noise"
        return Outcome((), basis, NO_INJECTION)
    if replay.condition == MIXED_CONDITION:
        return mixed_outcome(replay, step, exceeded, reference, context)
    (injector,) = replay.injectors  # replay_problem admitted exactly one
    if exceeded:
        return Outcome((injector.resource_class,), "contention_exceeded_threshold", injector.method)
    return Outcome((), "contention_below_threshold", injector.method)


def control_exceeded_noise(replay: Replay, step_id: str, context: Context) -> bool | None:
    """Whether the clean control right after a contended replay drifted on this step.

    Reported beside the label and never used to change it; None when unknown.
    """
    if replay.role != CONTENDED:
        return None
    control = context.usable.get(replay.index + 1)
    if control is None or control.role != CLEAN_CONTROL:
        return None
    step = control.steps.get(step_id)
    if step is None or step_problem(step) is not None:
        return None
    return context.references[step_id].exceeded(step)


def label_resource(resources: tuple[str, ...]) -> str:
    if not resources:
        return "none"
    return resources[0] if len(resources) == 1 else MIXED


def label_step(
    experiment: Experiment,
    replay: Replay,
    step: StepTiming,
    failed_checks: str | None,
    context: Context,
) -> dict:
    # A step without a reference is rejected in every replay, its baselines included.
    reference = context.references.get(step.step_id)
    if reference is None:
        raise UnlabelableError(
            "baseline_reference_missing",
            f"fewer than {MIN_BASELINE_COPIES} usable baseline copies of the step",
        )
    problem = step_problem(step)
    if problem is not None or step.total_seconds is None:
        raise problem or UnlabelableError("step_not_completed", "no total_seconds")
    exceeded = reference.exceeded(step)
    outcome = step_outcome(replay, step, exceeded, reference, context)
    return {
        "experiment": experiment.name,
        "task_id": experiment.task_id,
        "task_group": task_group(experiment.task_id),
        "plan_sha256": experiment.plan_sha256,
        "agent_model": experiment.model,
        "run_id": replay.run_id,
        "replay_index": replay.index,
        "role": replay.role,
        "condition": replay.condition,
        "step_id": step.step_id,
        "step_type": step.step_type,
        "duration_seconds": step.total_seconds,
        "queue_seconds": step.queue_seconds,
        "execution_seconds": step.execution_seconds,
        "label_resource": label_resource(outcome.resources),
        "label_resources": ",".join(outcome.resources),
        "label_excess_fraction": excess_fraction(step.total_seconds, reference.mean_seconds),
        "label_basis": outcome.basis,
        "injection_method": outcome.injection_method,
        "baseline_mean_seconds": reference.mean_seconds,
        "noise_threshold_seconds": reference.threshold_seconds,
        "control_exceeded_noise": control_exceeded_noise(replay, step.step_id, context),
        "failed_checks": failed_checks,
        "label_version": context.config.label_version,
    }


def experiment_problem(experiment: Experiment) -> UnlabelableError | None:
    if experiment.status != "completed":
        return UnlabelableError("experiment_not_completed", f"status is {experiment.status!r}")
    # A check the manifest did not record is not a pass.
    failed = [name for name in BLOCKING_CHECKS if experiment.checks.get(name) is not True]
    if failed:
        return UnlabelableError(
            "protocol_checks_failed", f"failed or missing checks {', '.join(failed)}"
        )
    return None


def label_experiment(
    experiment: Experiment, config: ReplayLabelConfig
) -> tuple[list[dict], list[Rejection]]:
    """Label every leaf step of one experiment, ordered by (replay_index, step_id).

    Each replay and step that cannot be labeled is rejected with its reason and the rest
    continue; a failed experiment-level check rejects the whole experiment once.
    """
    origin = f"experiment {experiment.name}"
    rejections: list[Rejection] = []

    def reject(where: str, error: UnlabelableError) -> None:
        logger.warning("Rejected %s at labeling: %s", where, error)
        rejections.append(Rejection(where, str(error)))

    problem = experiment_problem(experiment)
    if problem is not None:
        reject(origin, problem)
        return [], rejections
    failed_checks = ",".join(
        sorted(
            name
            for name, passed in experiment.checks.items()
            if not passed and name not in BLOCKING_CHECKS
        )
    )
    # Warm-ups are never labeled, so they are neither rows nor rejections.
    measured = sorted(
        (replay for replay in experiment.replays if replay.role in REPLAY_ROLES),
        key=lambda replay: replay.index,
    )
    usable: dict[int, Replay] = {}
    for replay in measured:
        problem = replay_problem(replay, config)
        if problem is not None:
            reject(f"{origin} replay {replay.index:02d}", problem)
        else:
            usable[replay.index] = replay
    solo_by_class: dict[str, list[Replay]] = {}
    for replay in usable.values():
        if replay.role == CONTENDED and replay.condition != MIXED_CONDITION:
            solo_by_class.setdefault(replay.injectors[0].resource_class, []).append(replay)
    baselines = [replay for replay in usable.values() if replay.role == BASELINE]
    context = Context(config, baseline_references(baselines, config), usable, solo_by_class)

    rows = []
    for index, replay in usable.items():
        for step_id, step in sorted(replay.steps.items()):
            # Parent steps cover their children's time; labeling them would count one
            # slowdown several times.
            if not step.is_leaf:
                continue
            try:
                rows.append(label_step(experiment, replay, step, failed_checks or None, context))
            except UnlabelableError as error:
                reject(f"{origin} replay {index:02d} step {step_id}", error)
    return rows, rejections


def label_directory(
    directory: Path, config: ReplayLabelConfig | None = None
) -> tuple[list[dict], list[Rejection]]:
    """Read and label one experiment directory; unreadable evidence raises."""
    rules = config if config is not None else load_replay_config()
    return label_experiment(read_experiment(directory), rules)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", type=Path, help="replay experiment directory")
    parser.add_argument("--rules", type=Path, help="label rules (default: packaged labels-v2.toml)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_replay_config(args.rules)
        rows, rejections = label_directory(args.experiment, config)
    except (ReplayEvidenceError, LabelingError) as error:
        print(f"Replay labeling failed: {error}", file=sys.stderr)
        return 1
    output = {"rows": rows, "rejections": [dataclasses.asdict(r) for r in rejections]}
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
