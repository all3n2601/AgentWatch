"""Preregistration manifests for optimization experiments (O04), checked against opt-contract-v1.

An optimization experiment compares the baseline concurrency policy with preregistered
candidate limits in paired batches (docs/optimization-contract.md). Its manifest is written
before any batch runs and fixes everything a comparison depends on: the contract version and
hash, the frozen task set with its content and configuration hashes, the model, hardware and
power identity, the conditions and their injectors, the arrival schedule, the candidate
policies, queue bounds, repetitions, warm-up and the randomization seed.

The manifest is identified by the SHA-256 of its parsed contents as canonical JSON. Every batch
record names that hash, so a manifest edited after batches ran no longer matches them. The
batch order is derived from the recorded seed by `schedule`, so it can be reproduced and
checked rather than trusted.

Policies, conditions and injectors are experiment design. They identify what was compared and
must never become attribution features (DESIGN_FIELDS).
"""

import math
import random
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from agentwatch_data.extract import TIMESTAMP, canonical_sha256, parse_json
from agentwatch_data.files import read_bounded_text

MANIFEST_FORMAT = "opt-experiment-manifest-v1"
CONTRACT_VERSION = "opt-contract-v1"
CONTRACT_STATUSES = ("proposed", "accepted")
POLICY_TARGET = "coding-agent"  # the only target O02 serves
PURPOSES = ("development", "evaluation")
MODEL_BACKENDS = ("ollama", "vllm")
POWER_SOURCES = ("ac", "battery")
INJECTOR_KINDS = ("cpu", "inference", "retrieval")
ARRIVAL_METHOD = "fixed_offsets_v1"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SEED = 2**63 - 1
SHA256 = re.compile(r"[0-9a-f]{64}")
TASK_ID = re.compile(r"[a-z0-9][a-z0-9-]*-r\d+")
NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
BASELINE, CANDIDATE = "baseline", "candidate"
WARMUP, PAIR = "warmup", "pair"

# Manifest and batch fields that describe the experiment design or the policy under test.
# A model that saw them would learn the experiment, not the resource that slowed a step.
DESIGN_FIELDS = frozenset(
    {
        "candidate",
        "condition",
        "configuration",
        "contention",
        "injectors",
        "inference_requests",
        "limits",
        "pair_index",
        "policy",
        "policy_sha256",
        "purpose",
        "queue_bounds",
        "test_jobs",
    }
)


class ManifestError(ValueError):
    """The manifest or the contract cannot support a comparison; nothing was compared."""


@dataclass(frozen=True)
class OptimizationContract:
    """The opt-contract-v1 values a manifest and its comparison depend on."""

    version: str
    status: str
    sha256: str  # full hex digest; reports show the first 8 digits, as the contract page says
    controls: tuple[str, ...]
    min_limit: int
    candidate_limits: tuple[int, ...]
    conditions: tuple[str, ...]
    contended_conditions: tuple[str, ...]  # conditions that preregister an injector set
    min_evaluation_tasks: int
    min_paired_batches: int
    min_completed_tasks: int
    bootstrap_resamples: int
    interval_level: float
    max_baseline_cv: float
    enforcement_scope: str


@dataclass(frozen=True)
class Task:
    task_id: str
    content_sha256: str
    configuration_digest: str


@dataclass(frozen=True)
class Policy:
    name: str
    limits: dict[str, int]  # an absent control has no limit
    sha256: str


@dataclass(frozen=True)
class Condition:
    name: str
    injectors: tuple[dict, ...]  # empty for an uncontended condition


@dataclass(frozen=True)
class Manifest:
    sha256: str
    experiment_id: str
    purpose: str
    tasks: tuple[Task, ...]
    arrivals: dict[str, float]  # task_id -> seconds after the batch's first submission
    model: dict
    hardware: dict
    power: dict
    state: dict
    conditions: tuple[Condition, ...]
    baseline: Policy
    candidates: tuple[Policy, ...]
    pairs: int
    warmup_batches: int
    seed: int

    def policy(self, name: str | None) -> Policy:
        if name is None:
            return self.baseline
        return next(candidate for candidate in self.candidates if candidate.name == name)


@dataclass(frozen=True)
class ScheduledBatch:
    index: int
    kind: str  # warmup or pair
    condition: str
    configuration: str  # baseline or candidate
    candidate: str | None  # the candidate the pair compares; None for warm-up batches
    pair_index: int | None
    policy_sha256: str

    @property
    def pair_id(self) -> str | None:
        if self.kind != PAIR:
            return None
        return f"{self.condition}:{self.candidate}:{self.pair_index}"


def contract_section(parsed: dict, name: str) -> dict:
    value = parsed.get(name)
    if not isinstance(value, dict):
        raise ManifestError(f"Optimization contract has no [{name}] table")
    return value


def load_optimization_contract(path: Path) -> OptimizationContract:
    """Read opt-contract-v1 and identify it by the SHA-256 of its parsed canonical JSON."""
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ManifestError(f"Optimization contract {path} cannot be read: {error}") from error
    if parsed.get("version") != CONTRACT_VERSION:
        raise ManifestError(f"Optimization contract version must be {CONTRACT_VERSION!r}")
    if parsed.get("status") not in CONTRACT_STATUSES:
        raise ManifestError(f"Optimization contract status must be one of {CONTRACT_STATUSES}")
    policy = contract_section(parsed, "policy")
    workload = contract_section(parsed, "workload")
    repetitions = contract_section(parsed, "repetitions")
    gates = contract_section(parsed, "gates")
    try:
        return OptimizationContract(
            version=parsed["version"],
            status=parsed["status"],
            sha256=canonical_sha256(parsed),
            controls=tuple(policy["controls"]),
            min_limit=policy["min_limit"],
            candidate_limits=tuple(workload["candidate_limits"]),
            conditions=tuple(workload["conditions"]),
            contended_conditions=tuple(gates["improvement_conditions"]),
            min_evaluation_tasks=workload["min_evaluation_tasks"],
            min_paired_batches=repetitions["min_paired_batches"],
            min_completed_tasks=repetitions["min_completed_tasks_per_configuration"],
            bootstrap_resamples=gates["bootstrap_resamples"],
            interval_level=gates["interval_level"],
            max_baseline_cv=gates["max_baseline_root_step_cv"],
            enforcement_scope=policy["enforcement_scope"],
        )
    except KeyError as error:
        raise ManifestError(f"Optimization contract lacks {error.args[0]!r}") from error


# --- Field readers: every manifest value is untrusted ----------------------------------------


def require_object(value, name: str, keys: set[str], optional: set[str] = frozenset()) -> dict:
    """An object with exactly the required keys, plus any of the optional ones."""
    if not isinstance(value, dict):
        raise ManifestError(f"{name} must be an object")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys - optional)
    if missing or unknown:
        raise ManifestError(f"{name} keys: missing {missing}, unknown {unknown}")
    return value


def require_text(value, name: str, pattern: re.Pattern | None = None) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ManifestError(f"{name} must be a non-empty string of at most 256 characters")
    if pattern is not None and not pattern.fullmatch(value):
        raise ManifestError(f"{name} {value!r} has an invalid format")
    return value


def require_choice(value, name: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ManifestError(f"{name} must be one of {choices}, got {value!r}")
    return value


def require_int(value, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{name} must be an integer of at least {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ManifestError(f"{name} must be at most {maximum}, got {value!r}")
    return value


def require_seconds(value, name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ManifestError(f"{name} must be a finite number of seconds, got {value!r}")
    if value < 0 or (positive and value == 0):
        raise ManifestError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def require_list(value, name: str, minimum: int = 0) -> list:
    if not isinstance(value, list) or len(value) < minimum:
        raise ManifestError(f"{name} must be a list of at least {minimum} entries")
    return value


def require_unique(values: list, name: str) -> None:
    if len(set(values)) != len(values):
        raise ManifestError(f"{name} must not repeat entries")


def require_uuid(value, name: str) -> str:
    try:
        return str(UUID(require_text(value, name)))
    except ValueError as error:
        raise ManifestError(f"{name} must be a UUID") from error


def require_timestamp(value, name: str) -> str:
    text = require_text(value, name)
    if not TIMESTAMP.fullmatch(text):
        raise ManifestError(f"{name} must be an RFC 3339 timestamp with a time zone")
    try:
        datetime.fromisoformat(text)
    except ValueError as error:
        raise ManifestError(f"{name} is not a valid timestamp: {error}") from error
    return text


# --- Identity sections, compared exactly between the manifest and every batch ----------------


def model_identity(value, name: str = "model") -> dict:
    model = require_object(value, name, {"backend", "name", "digest"})
    require_choice(model["backend"], f"{name}.backend", MODEL_BACKENDS)
    require_text(model["name"], f"{name}.name")
    require_text(model["digest"], f"{name}.digest")
    return model


def hardware_identity(value, name: str = "hardware") -> dict:
    keys = {"host_id", "cpu_model", "logical_cpus", "memory_bytes", "os", "kernel", "gpus"}
    hardware = require_object(value, name, keys)
    for key in ("host_id", "cpu_model", "os", "kernel"):
        require_text(hardware[key], f"{name}.{key}")
    require_int(hardware["logical_cpus"], f"{name}.logical_cpus", 1)
    require_int(hardware["memory_bytes"], f"{name}.memory_bytes", 1)
    for index, gpu in enumerate(require_list(hardware["gpus"], f"{name}.gpus")):
        gpu_name = f"{name}.gpus[{index}]"
        gpu = require_object(gpu, gpu_name, {"name", "driver_version", "memory_total_bytes"})
        require_text(gpu["name"], f"{gpu_name}.name")
        require_text(gpu["driver_version"], f"{gpu_name}.driver_version")
        require_int(gpu["memory_total_bytes"], f"{gpu_name}.memory_total_bytes", 1)
    return hardware


def power_identity(value, name: str = "power") -> dict:
    # An unknown power state cannot show that both batches of a pair ran under the same one.
    power = require_object(value, name, {"source", "profile"})
    require_choice(power["source"], f"{name}.source", POWER_SOURCES)
    require_text(power["profile"], f"{name}.profile")
    return power


def state_identity(value, name: str = "state") -> dict:
    state = require_object(value, name, {"tests", "retrieval"})
    require_text(state["tests"], f"{name}.tests")
    require_text(state["retrieval"], f"{name}.retrieval")
    return state


# --- Policies ---------------------------------------------------------------------------------


def policy_sha256(limits: dict[str, int], contract: OptimizationContract) -> str:
    """Identify a policy by what it enforces, not by who set it or when.

    Operator, reason and timestamps of a stored policy version are audit data; two versions
    with the same limits configure the runner identically.
    """
    return canonical_sha256(
        {
            "target": POLICY_TARGET,
            "limits": limits,
            "enforcement_scope": contract.enforcement_scope,
        }
    )


def policy_limits(value, name: str, contract: OptimizationContract) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ManifestError(f"{name} must be an object")
    unknown = sorted(set(value) - set(contract.controls))
    if unknown:
        raise ManifestError(f"{name} names unknown controls {unknown}")
    # A present limit is a positive integer; "no limit" is an absent key, never null or zero.
    return {
        control: require_int(limit, f"{name}.{control}", contract.min_limit)
        for control, limit in value.items()
    }


def read_policies(value, contract: OptimizationContract) -> tuple[Policy, tuple[Policy, ...]]:
    policies = require_object(value, "policies", {"baseline", "candidates"})
    baseline_entry = require_object(policies["baseline"], "policies.baseline", {"limits"})
    baseline_limits = policy_limits(baseline_entry["limits"], "policies.baseline.limits", contract)
    if baseline_limits:
        # opt-contract-v1: the baseline is the runner without admission limits.
        raise ManifestError("policies.baseline.limits must be empty (no limit on any control)")
    baseline = Policy(BASELINE, {}, policy_sha256({}, contract))
    candidates = []
    for index, entry in enumerate(require_list(policies["candidates"], "policies.candidates", 1)):
        name = f"policies.candidates[{index}]"
        entry = require_object(entry, name, {"name", "limits"})
        policy_name = require_text(entry["name"], f"{name}.name", NAME)
        limits = policy_limits(entry["limits"], f"{name}.limits", contract)
        if len(limits) != 1:
            # Factorial candidates are reported as not evaluated in v1, so none is preregistered.
            raise ManifestError(f"{name} must change exactly one control from the baseline")
        ((control, limit),) = limits.items()
        if limit not in contract.candidate_limits:
            raise ManifestError(
                f"{name}.limits.{control} must be a preregistered candidate limit "
                f"{list(contract.candidate_limits)}, got {limit}"
            )
        candidates.append(Policy(policy_name, limits, policy_sha256(limits, contract)))
    require_unique([candidate.name for candidate in candidates], "policies.candidates names")
    require_unique([candidate.sha256 for candidate in candidates], "policies.candidates limits")
    return baseline, tuple(candidates)


def read_queue_bounds(value, contract: OptimizationContract) -> None:
    bounds = require_object(value, "queue_bounds", set(contract.controls))
    for control in contract.controls:
        name = f"queue_bounds.{control}"
        entry = require_object(bounds[control], name, {"max_length", "max_wait_seconds"})
        require_int(entry["max_length"], f"{name}.max_length", 1)
        require_seconds(entry["max_wait_seconds"], f"{name}.max_wait_seconds", positive=True)


# --- Workload and design ----------------------------------------------------------------------


def read_tasks(value) -> tuple[Task, ...]:
    tasks = []
    for index, entry in enumerate(require_list(value, "workload.tasks", 1)):
        name = f"workload.tasks[{index}]"
        entry = require_object(entry, name, {"task_id", "content_sha256", "configuration_digest"})
        tasks.append(
            Task(
                task_id=require_text(entry["task_id"], f"{name}.task_id", TASK_ID),
                content_sha256=require_text(
                    entry["content_sha256"], f"{name}.content_sha256", SHA256
                ),
                configuration_digest=require_text(
                    entry["configuration_digest"], f"{name}.configuration_digest", SHA256
                ),
            )
        )
    require_unique([task.task_id for task in tasks], "workload.tasks task_id")
    return tuple(tasks)


def read_arrivals(value, tasks: tuple[Task, ...]) -> dict[str, float]:
    arrivals = require_object(value, "workload.arrivals", {"method", "offsets"})
    require_choice(arrivals["method"], "workload.arrivals.method", (ARRIVAL_METHOD,))
    offsets = {}
    for index, entry in enumerate(require_list(arrivals["offsets"], "workload.arrivals.offsets")):
        name = f"workload.arrivals.offsets[{index}]"
        entry = require_object(entry, name, {"task_id", "offset_seconds"})
        task_id = require_text(entry["task_id"], f"{name}.task_id", TASK_ID)
        if task_id in offsets:
            raise ManifestError(f"workload.arrivals schedules {task_id} more than once")
        offsets[task_id] = require_seconds(entry["offset_seconds"], f"{name}.offset_seconds", False)
    if set(offsets) != {task.task_id for task in tasks}:
        raise ManifestError("workload.arrivals must schedule every task exactly once")
    if min(offsets.values()) != 0:
        # A batch's time runs from its first submission, so the schedule starts at zero.
        raise ManifestError("workload.arrivals must start at offset 0")
    return offsets


def read_workload(document: dict, contract: OptimizationContract) -> tuple[tuple[Task, ...], dict]:
    workload = require_object(
        document["workload"], "workload", {"unit", "tasks", "arrivals"}, {"development_tasks"}
    )
    require_choice(workload["unit"], "workload.unit", ("batch",))
    tasks = read_tasks(workload["tasks"])
    task_ids = {task.task_id for task in tasks}
    development = [
        require_text(task_id, "workload.development_tasks entry", TASK_ID)
        for task_id in require_list(workload.get("development_tasks", []), "development_tasks")
    ]
    if document["purpose"] == "evaluation":
        # Settings are chosen on development tasks; frozen evaluation tasks are used once.
        if not development:
            raise ManifestError("An evaluation manifest must name its development_tasks")
        if task_ids & set(development):
            raise ManifestError("Evaluation tasks must be disjoint from development_tasks")
        if len(tasks) < contract.min_evaluation_tasks:
            raise ManifestError(
                f"An evaluation needs at least {contract.min_evaluation_tasks} frozen tasks"
            )
    elif development:
        raise ManifestError("Only an evaluation manifest names development_tasks")
    return tasks, read_arrivals(workload["arrivals"], tasks)


def read_conditions(value, contract: OptimizationContract) -> tuple[Condition, ...]:
    conditions = []
    for index, entry in enumerate(require_list(value, "conditions", 1)):
        name = f"conditions[{index}]"
        entry = require_object(entry, name, {"name", "injectors"})
        condition = require_choice(entry["name"], f"{name}.name", contract.conditions)
        injectors = require_list(entry["injectors"], f"{name}.injectors")
        if condition in contract.contended_conditions and not injectors:
            raise ManifestError(f"{name} must preregister its injector set and intensity")
        if condition not in contract.contended_conditions and injectors:
            raise ManifestError(f"{name} is uncontended and must not declare injectors")
        for position, injector in enumerate(injectors):
            injector_name = f"{name}.injectors[{position}]"
            injector = require_object(injector, injector_name, {"kind", "method", "intensity"})
            require_choice(injector["kind"], f"{injector_name}.kind", INJECTOR_KINDS)
            require_text(injector["method"], f"{injector_name}.method")
            if not isinstance(injector["intensity"], dict) or not injector["intensity"]:
                raise ManifestError(f"{injector_name}.intensity must be a non-empty object")
        require_unique([injector["kind"] for injector in injectors], f"{name}.injectors kinds")
        conditions.append(Condition(condition, tuple(injectors)))
    require_unique([condition.name for condition in conditions], "conditions names")
    if {condition.name for condition in conditions} != set(contract.conditions):
        raise ManifestError(f"conditions must cover exactly {list(contract.conditions)}")
    return tuple(conditions)


def read_repetitions(value, tasks: tuple[Task, ...], contract: OptimizationContract) -> dict:
    repetitions = require_object(value, "repetitions", {"pairs", "warmup_batches", "seed"})
    pairs = require_int(repetitions["pairs"], "repetitions.pairs", contract.min_paired_batches)
    require_int(repetitions["warmup_batches"], "repetitions.warmup_batches", 0)
    require_int(repetitions["seed"], "repetitions.seed", 0, MAX_SEED)
    if pairs * len(tasks) < contract.min_completed_tasks:
        raise ManifestError(
            f"{pairs} pairs of {len(tasks)} tasks cannot reach the contract's "
            f"{contract.min_completed_tasks} completed tasks per configuration"
        )
    return repetitions


TOP_LEVEL_KEYS = {
    "format",
    "experiment_id",
    "created_at",
    "contract",
    "purpose",
    "workload",
    "runner",
    "model",
    "hardware",
    "power",
    "state",
    "conditions",
    "policies",
    "queue_bounds",
    "repetitions",
}


def parse_manifest(document, contract: OptimizationContract) -> Manifest:
    document = require_object(document, "manifest", TOP_LEVEL_KEYS)
    require_choice(document["format"], "format", (MANIFEST_FORMAT,))
    declared = require_object(document["contract"], "contract", {"version", "sha256"})
    if declared["version"] != contract.version or declared["sha256"] != contract.sha256:
        # A stale or edited contract changes what the gates mean; the comparison must not mix them.
        raise ManifestError(
            f"Manifest contract {declared['version']!r} "
            f"{str(declared['sha256'])[:8]} does not match {contract.version} "
            f"{contract.sha256[:8]}"
        )
    require_choice(document["purpose"], "purpose", PURPOSES)
    require_timestamp(document["created_at"], "created_at")
    require_object(document["runner"], "runner", {"version"})
    require_text(document["runner"]["version"], "runner.version")
    tasks, arrivals = read_workload(document, contract)
    baseline, candidates = read_policies(document["policies"], contract)
    read_queue_bounds(document["queue_bounds"], contract)
    repetitions = read_repetitions(document["repetitions"], tasks, contract)
    return Manifest(
        sha256=canonical_sha256(document),
        experiment_id=require_uuid(document["experiment_id"], "experiment_id"),
        purpose=document["purpose"],
        tasks=tasks,
        arrivals=arrivals,
        model=model_identity(document["model"]),
        hardware=hardware_identity(document["hardware"]),
        power=power_identity(document["power"]),
        state=state_identity(document["state"]),
        conditions=read_conditions(document["conditions"], contract),
        baseline=baseline,
        candidates=candidates,
        pairs=repetitions["pairs"],
        warmup_batches=repetitions["warmup_batches"],
        seed=repetitions["seed"],
    )


def read_manifest(path: Path, contract: OptimizationContract) -> Manifest:
    text = read_bounded_text(path, MAX_MANIFEST_BYTES, ManifestError)
    try:
        document = parse_json(text)
    except ValueError as error:
        raise ManifestError(f"{path.name}: {error}") from error
    return parse_manifest(document, contract)


def schedule(manifest: Manifest) -> tuple[ScheduledBatch, ...]:
    """The preregistered batch order, reproducible from the manifest's seed.

    For each condition in manifest order: its warm-up batches at baseline limits, then every
    (candidate, repetition) pair in a seeded random order. A pair's two batches run back to
    back, and a seeded coin decides which configuration runs first.
    """
    generator = random.Random(manifest.seed)
    batches: list[ScheduledBatch] = []

    def add(kind, condition, configuration, candidate, pair_index, policy):
        batches.append(
            ScheduledBatch(
                len(batches), kind, condition, configuration, candidate, pair_index, policy
            )
        )

    for condition in manifest.conditions:
        for _ in range(manifest.warmup_batches):
            add(WARMUP, condition.name, BASELINE, None, None, manifest.baseline.sha256)
        pairs = [
            (candidate, repetition)
            for candidate in manifest.candidates
            for repetition in range(manifest.pairs)
        ]
        generator.shuffle(pairs)
        for candidate, repetition in pairs:
            order = [(BASELINE, manifest.baseline), (CANDIDATE, candidate)]
            if generator.random() < 0.5:
                order.reverse()
            for configuration, policy in order:
                add(PAIR, condition.name, configuration, candidate.name, repetition, policy.sha256)
    return tuple(batches)
