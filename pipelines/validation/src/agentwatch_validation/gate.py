"""Dataset quality gate: TFDV statistics checked against a frozen schema, plus drift.

Runs in its own environment (TensorFlow Data Validation is too heavy for the shared one), so it
reads everything it needs from the release itself: steps.parquet, the column contract in
schema.json, and manifest.json for integrity.

    python -m agentwatch_validation validate --release <dir> --schema <steps.pbtxt>
    python -m agentwatch_validation freeze-schema --release <dir> --schema <steps.pbtxt>

A release whose Parquet columns or types disagree with its own contract is refused before any
conversion. Schema anomalies (a missing or unexpected column, a type change, a value outside
its domain, an unknown category, a required value missing) fail the gate. Drift against the
previous passing release is reported but does not fail it: new contention or tasks
legitimately shift distributions.

Each validation is an attempt, committed as one immutable directory,
<datasets>/validations/<release>/<attempt>/, holding result.json and the statistics it was
computed from. A release's verdict is its latest committed attempt; earlier attempts are kept
as evidence. A committed attempt that cannot be read, or claims to come from the future, makes
the verdict unknown rather than silently falling back to an older attempt.
"""

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tensorflow_data_validation as tfdv
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2, schema_pb2, statistics_pb2

from agentwatch_validation import __version__

VALIDATIONS_DIR = "validations"
RESULT_FILE, STATISTICS_FILE = "result.json", "statistics.pbtxt"
DATASET_FILE, MANIFEST_FILE, CONTRACT_FILE = "steps.parquet", "manifest.json", "schema.json"
PASSED, FAILED = "passed", "failed"

# Categorical columns whose distribution is compared with the previous passing release. The
# L-infinity threshold (largest change in any category's share) is a proposed value.
DRIFT_COLUMNS = ("label_resource", "condition", "label_basis", "telemetry_status", "kind")
DRIFT_L_INFINITY_THRESHOLD = 0.25

# Contract dtype -> (the Arrow type the builder writes, the TFDV feature type it becomes).
DTYPES = {
    "string": (pa.string(), schema_pb2.BYTES),
    "float64": (pa.float64(), schema_pb2.FLOAT),
    "int64": (pa.int64(), schema_pb2.INT),
    "bool": (pa.bool_(), schema_pb2.INT),
    "timestamp[us, tz=UTC]": (pa.timestamp("us", tz="UTC"), schema_pb2.INT),
}


class GateError(RuntimeError):
    """The release or schema cannot be validated at all (not a data anomaly)."""


class UnknownVerdict(GateError):
    """A committed attempt is damaged or impossible, so the release's verdict is unknown."""


@dataclass(frozen=True)
class Release:
    path: Path
    manifest: dict
    contract: dict
    frame: pd.DataFrame


@dataclass(frozen=True)
class Outcome:
    status: str
    result: dict
    result_path: Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def text_sha256(path: Path) -> str:
    """Hash of a text file with normalized line endings, so checkouts agree."""
    return hashlib.sha256(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    ).hexdigest()


def read_json(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GateError(f"{what} {path.name} is missing; rebuild the release") from error
    except (OSError, ValueError) as error:
        raise GateError(f"{what} {path.name} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"{what} {path.name} is not a JSON object")
    return value


def load_release(path: Path) -> Release:
    """Read a release, verify it against its manifest, and prepare it for TFDV."""
    if not (path / DATASET_FILE).is_file():
        raise GateError(f"{path} is not a dataset release")
    manifest = read_json(path / MANIFEST_FILE, "Manifest")
    contract = read_json(path / CONTRACT_FILE, "Column contract")
    if manifest.get("parquet_sha256") != file_sha256(path / DATASET_FILE):
        raise GateError(f"{path.name}: {DATASET_FILE} does not match its manifest")
    if manifest.get("schema_sha256") != file_sha256(path / CONTRACT_FILE):
        raise GateError(f"{path.name}: {CONTRACT_FILE} does not match its manifest")
    columns = [column["name"] for column in contract.get("columns", [])]
    table = pq.read_table(path / DATASET_FILE)
    if table.column_names != columns:
        raise GateError(f"{path.name}: Parquet columns do not match the column contract")
    # TFDV checks nothing, not even dataset constraints, on zero examples.
    if table.num_rows == 0:
        raise GateError(f"{path.name}: {DATASET_FILE} has no rows")
    check_types(path.name, table.schema, contract)
    check_values(path.name, table, contract)
    return Release(path, manifest, contract, to_frame(table.to_pandas(), contract))


def check_values(release: str, table: pa.Table, contract: dict) -> None:
    """Refuse values the contract forbids but TFDV cannot see once the frame is prepared.

    pandas turns NaN into a missing value, so a malformed reading would pass as "not
    measured"; and TFDV has no domain for "nonempty string".
    """
    for column in contract["columns"]:
        name, values = column["name"], table.column(column["name"])
        if column["dtype"] == "float64" and pc.any(pc.is_nan(values)).as_py():
            raise GateError(f"{release}: column {name} contains NaN")
        if column["dtype"] == "string":
            blank = pc.equal(pc.utf8_trim_whitespace(values), "")
            if pc.any(blank).as_py():
                raise GateError(f"{release}: column {name} contains an empty string")


def check_types(release: str, schema: pa.Schema, contract: dict) -> None:
    """Refuse Parquet columns whose physical type differs from the contract.

    to_frame converts declared integer and boolean columns, and that conversion would quietly
    turn a string column holding "1" into a valid integer, hiding the type change from TFDV.
    """
    for column in contract["columns"]:
        name, dtype = column["name"], column["dtype"]
        if dtype not in DTYPES:
            raise GateError(f"{release}: column {name} has unsupported contract dtype {dtype!r}")
        actual = schema.field(name).type
        if not actual.equals(DTYPES[dtype][0]):
            raise GateError(
                f"{release}: Parquet column {name} is {actual}, but the contract declares {dtype}"
            )


def to_frame(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Represent every column the way TFDV can type it.

    Nullable integers use pandas' Int64 so missing values do not turn them into floats;
    booleans become 0/1; timestamps become integer microseconds since the epoch, because TFDV
    would otherwise drop datetime columns.
    """
    prepared = frame.copy()
    for column in contract["columns"]:
        name, dtype = column["name"], column["dtype"]
        if dtype.startswith("timestamp"):
            prepared[name] = prepared[name].map(
                lambda value: None if pd.isna(value) else value.value // 1000
            )
            prepared[name] = prepared[name].astype("Int64")
        elif dtype in ("int64", "bool"):
            prepared[name] = prepared[name].astype("Int64")
    return prepared


def statistics(release: Release):
    return tfdv.generate_statistics_from_dataframe(release.frame)


def freeze_schema(release: Release) -> schema_pb2.Schema:
    """Build the reviewed reference schema: inferred features, constrained by the contract.

    Inferred domains only describe values seen so far (for example only two of the four
    resource classes), so they are replaced by the contract's allowed values and bounds.
    Free-text columns get no domain, so new tasks and runs are not anomalies.
    """
    schema = tfdv.infer_schema(statistics(release), infer_feature_shape=True)
    del schema.string_domain[:]
    features = {feature.name: feature for feature in schema.feature}
    for column in release.contract["columns"]:
        feature = features.get(column["name"]) or schema.feature.add(name=column["name"])
        feature.type = DTYPES[column["dtype"]][1]
        feature.ClearField("domain_info")
        feature.presence.min_fraction = 0.0 if column["nullable"] else 1.0
        feature.presence.min_count = 0 if column["nullable"] else 1
        if column["nullable"]:
            # A fixed shape means "exactly one value in every row", so a nullable column
            # inferred from data without nulls would reject the first legitimate null.
            feature.ClearField("shape")
        minimum, maximum = column["minimum"], column["maximum"]
        if column["allowed"] is not None:
            feature.string_domain.value.extend(column["allowed"])
        elif column["dtype"] == "bool":
            feature.int_domain.min, feature.int_domain.max = 0, 1
        elif column["dtype"] == "int64" and (minimum is not None or maximum is not None):
            if minimum is not None:
                feature.int_domain.min = int(minimum)
            if maximum is not None:
                feature.int_domain.max = int(maximum)
        elif column["dtype"] == "float64":
            # The contract requires finite numbers; without this, +-inf passes a domain.
            feature.float_domain.disallow_inf = True
            if minimum is not None:
                feature.float_domain.min = float(minimum)
            if maximum is not None:
                feature.float_domain.max = float(maximum)
        if column["name"] in DRIFT_COLUMNS:
            feature.drift_comparator.infinity_norm.threshold = DRIFT_L_INFINITY_THRESHOLD
    known = {column["name"] for column in release.contract["columns"]}
    unknown = [feature.name for feature in schema.feature if feature.name not in known]
    if unknown:
        raise GateError(f"Columns outside the contract: {unknown}")
    return schema


def write_schema(schema: schema_pb2.Schema, path: Path, schema_version: str) -> None:
    header = (
        f"# Frozen TFDV schema for {schema_version}. Generated by\n"
        "#   python -m agentwatch_validation freeze-schema\n"
        "# from a release and its column contract. Change it only in a reviewed PR.\n"
    )
    write_atomic(path, header + text_format.MessageToString(schema))


def load_schema(path: Path) -> schema_pb2.Schema:
    # TFDV reads through TensorFlow's file API, which raises its own NotFoundError rather
    # than FileNotFoundError, so existence is checked first.
    if not path.is_file():
        raise GateError(f"Frozen schema {path} is missing; run freeze-schema")
    try:
        return tfdv.load_schema_text(str(path))
    except (OSError, text_format.ParseError) as error:
        raise GateError(f"Frozen schema {path} is unreadable: {error}") from error


def classify(anomalies) -> tuple[list[dict], list[dict]]:
    """Split TFDV anomalies into blocking schema problems and informational drift."""
    blocking, drift = [], []
    for feature, info in sorted(anomalies.anomaly_info.items()):
        kinds = sorted({anomalies_pb2.AnomalyInfo.Type.Name(r.type) for r in info.reason})
        record = {"feature": feature, "types": kinds, "description": info.description}
        is_drift = all(kind.startswith("COMPARATOR_") for kind in kinds)
        (drift if is_drift else blocking).append(record)
    if anomalies.HasField("dataset_anomaly_info"):
        info = anomalies.dataset_anomaly_info
        kinds = sorted({anomalies_pb2.AnomalyInfo.Type.Name(r.type) for r in info.reason})
        blocking.append({"feature": "<dataset>", "types": kinds, "description": info.description})
    return blocking, drift


def validated_time(result: dict) -> datetime | None:
    try:
        moment = datetime.fromisoformat(result["validated_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else None


def committed_attempts(
    datasets: Path, release: str, now: datetime | None = None
) -> list[tuple[dict, Path]]:
    """A release's committed validation attempts, oldest first, as (result, directory).

    Staging directories are hidden and are not attempts. Every other directory was committed
    by a rename, so one whose result is missing, unreadable, or names another attempt or
    release is damaged evidence, and one validated after now is impossible; either raises
    UnknownVerdict instead of letting an older attempt stand in for it.
    """
    now = now or datetime.now(UTC)
    attempts, damaged = [], []
    base = datasets / VALIDATIONS_DIR / release
    for directory in sorted(base.iterdir()) if base.is_dir() else []:
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        try:
            result = json.loads((directory / RESULT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = None
        moment = validated_time(result) if isinstance(result, dict) else None
        if (
            moment is None
            or result.get("attempt") != directory.name
            or result.get("release") != release
            or result.get("status") not in (PASSED, FAILED)
        ):
            damaged.append(f"{directory.name} is unreadable or malformed")
        elif moment > now:
            damaged.append(f"{directory.name} was validated in the future ({moment.isoformat()})")
        else:
            attempts.append((moment, result, directory))
    if damaged:
        raise UnknownVerdict(
            f"Verdict of {release} is unknown: {'; '.join(damaged)}. Inspect the attempt; "
            "a verdict returns once it is repaired or removed"
        )
    attempts.sort(key=lambda item: (item[0], item[1]["attempt"]))
    return [(result, directory) for _, result, directory in attempts]


def latest_result(
    datasets: Path, release: str, now: datetime | None = None
) -> tuple[dict, Path] | tuple[None, None]:
    """A release's verdict: its latest committed attempt, or (None, None) if never validated."""
    attempts = committed_attempts(datasets, release, now)
    return attempts[-1] if attempts else (None, None)


def previous_passing(datasets: Path, current: str, schema_sha256: str, now: datetime | None = None):
    """The most recently validated release, other than current, whose verdict is a pass.

    Returns its name and the exact statistics that attempt passed with. Only a pass against
    the same frozen schema counts, so drift is never measured against a release checked by
    other rules. A release whose latest attempt failed, or whose verdict is unknown, is not a
    baseline, even though its earlier passing attempts are kept.
    """
    candidates = []
    validations = datasets / VALIDATIONS_DIR
    for release_dir in validations.iterdir() if validations.is_dir() else []:
        if release_dir.name == current or not release_dir.is_dir():
            continue
        try:
            result, attempt = latest_result(datasets, release_dir.name, now)
        except UnknownVerdict:
            continue
        if (
            result is None
            or result.get("status") != PASSED
            or result.get("schema_sha256") != schema_sha256
        ):
            continue
        try:
            text = (attempt / STATISTICS_FILE).read_text(encoding="utf-8")
        except OSError:
            continue
        if sha256(text) == result.get("statistics_sha256"):
            moment = validated_time(result)
            candidates.append((moment, result["attempt"], result["release"], text))
    if not candidates:
        return None, None
    *_, release, text = max(candidates)
    return release, text_format.Parse(text, statistics_pb2.DatasetFeatureStatisticsList())


def validate(release_path: Path, schema_path: Path, now: datetime | None = None) -> Outcome:
    release = load_release(release_path)
    schema = load_schema(schema_path)
    datasets = release_path.parent
    validated_at = now or datetime.now(UTC)
    schema_sha256 = text_sha256(schema_path)
    baseline_release, baseline_stats = previous_passing(
        datasets, release_path.name, schema_sha256, validated_at
    )
    stats = statistics(release)
    anomalies = tfdv.validate_statistics(stats, schema, previous_statistics=baseline_stats)
    blocking, drift = classify(anomalies)
    status = FAILED if blocking else PASSED
    attempt = f"{validated_at:%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"
    statistics_text = text_format.MessageToString(stats)
    result = {
        "attempt": attempt,
        "release": release_path.name,
        "status": status,
        "validated_at": validated_at.isoformat(),
        "examples": int(stats.datasets[0].num_examples),
        # What this verdict is bound to: the exact data, contract, frozen schema, and statistics.
        "release_parquet_sha256": release.manifest["parquet_sha256"],
        "release_contract_sha256": release.manifest["schema_sha256"],
        "schema_file": schema_path.name,
        "schema_sha256": schema_sha256,
        "schema_version": release.contract["schema_version"],
        "statistics_sha256": sha256(statistics_text),
        "baseline_release": baseline_release,
        "anomalies": blocking,
        "drift": drift,
        "gate_version": __version__,
        "tfdv_version": tfdv.__version__,
    }
    directory = commit_attempt(
        datasets / VALIDATIONS_DIR / release_path.name,
        attempt,
        {
            STATISTICS_FILE: statistics_text,
            RESULT_FILE: json.dumps(result, indent=2, sort_keys=True) + "\n",
        },
    )
    return Outcome(status, result, directory / RESULT_FILE)


def commit_attempt(parent: Path, attempt: str, files: dict[str, str]) -> Path:
    """Write an attempt's files into private staging, then rename the directory into place.

    The rename is the commit: readers see all of an attempt's files or none, and concurrent
    attempts never share a staging path. A failure leaves earlier attempts untouched.
    """
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".attempt-", dir=parent))
    try:
        staging.chmod(0o755)  # mkdtemp is owner-only; Airflow and DVC may run as other users
        for name, text in files.items():
            with (staging / name).open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())  # durable before the rename makes them visible
            (staging / name).chmod(0o644)
        fsync_directory(staging)
        os.replace(staging, parent / attempt)
        fsync_directory(parent)  # and the rename itself survives a crash
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return parent / attempt


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path: Path, text: str) -> None:
    """Write text so readers never see a partial file; readable by other users."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        staged.chmod(0o644)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
