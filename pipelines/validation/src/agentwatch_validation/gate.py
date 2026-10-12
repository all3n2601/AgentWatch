"""Dataset quality gate: TFDV statistics checked against a frozen schema, plus drift.

Runs in its own environment (TensorFlow Data Validation is too heavy for the shared one), so it
reads everything it needs from the release itself: steps.parquet, the column contract in
schema.json, and manifest.json for integrity.

    python -m agentwatch_validation validate --release <dir> --schema <steps.pbtxt>
    python -m agentwatch_validation freeze-schema --release <dir> --schema <steps.pbtxt>

Schema anomalies (a missing or unexpected column, a type change, a value outside its domain,
an unknown category, a required value missing) fail the gate. Drift against the previous
passing release is reported but does not fail it: new contention or tasks legitimately shift
distributions. Results are written to <datasets>/validations/<release>/.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import tensorflow_data_validation as tfdv
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2, schema_pb2

from agentwatch_validation import __version__

VALIDATIONS_DIR = "validations"
RESULT_FILE, STATISTICS_FILE = "result.json", "statistics.pbtxt"
DATASET_FILE, MANIFEST_FILE, CONTRACT_FILE = "steps.parquet", "manifest.json", "schema.json"
PASSED, FAILED = "passed", "failed"

# Categorical columns whose distribution is compared with the previous passing release. The
# L-infinity threshold (largest change in any category's share) is a proposed value.
DRIFT_COLUMNS = ("label_resource", "condition", "label_basis", "telemetry_status", "kind")
DRIFT_L_INFINITY_THRESHOLD = 0.25

FEATURE_TYPES = {
    "string": schema_pb2.BYTES,
    "float64": schema_pb2.FLOAT,
    "int64": schema_pb2.INT,
    "bool": schema_pb2.INT,
    "timestamp[us, tz=UTC]": schema_pb2.INT,
}


class GateError(RuntimeError):
    """The release or schema cannot be validated at all (not a data anomaly)."""


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
    return Release(path, manifest, contract, to_frame(table.to_pandas(), contract))


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
        feature.type = FEATURE_TYPES[column["dtype"]]
        feature.ClearField("domain_info")
        feature.presence.min_fraction = 0.0 if column["nullable"] else 1.0
        feature.presence.min_count = 0 if column["nullable"] else 1
        if column["allowed"] is not None:
            feature.string_domain.value.extend(column["allowed"])
        elif column["dtype"] == "bool":
            feature.int_domain.min, feature.int_domain.max = 0, 1
        elif column["dtype"] == "int64" and column["minimum"] is not None:
            feature.int_domain.min = int(column["minimum"])
            if column["maximum"] is not None:
                feature.int_domain.max = int(column["maximum"])
        elif column["dtype"] == "float64" and column["minimum"] is not None:
            feature.float_domain.min = float(column["minimum"])
            if column["maximum"] is not None:
                feature.float_domain.max = float(column["maximum"])
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


def previous_passing(datasets: Path, current: str):
    """The most recently validated passing release other than current, with its statistics."""
    candidates = []
    for result_path in (datasets / VALIDATIONS_DIR).glob(f"*/{RESULT_FILE}"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stats_path = result_path.with_name(STATISTICS_FILE)
        if (
            isinstance(result, dict)
            and result.get("status") == PASSED
            and result.get("release") != current
            and stats_path.is_file()
        ):
            candidates.append((str(result.get("validated_at", "")), result["release"], stats_path))
    if not candidates:
        return None, None
    _, release, stats_path = max(candidates)
    return release, tfdv.load_stats_text(str(stats_path))


def validate(release_path: Path, schema_path: Path, now: datetime | None = None) -> Outcome:
    release = load_release(release_path)
    schema = load_schema(schema_path)
    datasets = release_path.parent
    baseline_release, baseline_stats = previous_passing(datasets, release_path.name)
    stats = statistics(release)
    anomalies = tfdv.validate_statistics(stats, schema, previous_statistics=baseline_stats)
    blocking, drift = classify(anomalies)
    status = FAILED if blocking else PASSED
    result = {
        "release": release_path.name,
        "status": status,
        "validated_at": (now or datetime.now(UTC)).isoformat(),
        "examples": int(stats.datasets[0].num_examples),
        "schema_file": schema_path.name,
        "schema_sha256": text_sha256(schema_path),
        "schema_version": release.contract["schema_version"],
        "baseline_release": baseline_release,
        "anomalies": blocking,
        "drift": drift,
        "gate_version": __version__,
        "tfdv_version": tfdv.__version__,
    }
    out = datasets / VALIDATIONS_DIR / release_path.name
    write_atomic(out / STATISTICS_FILE, text_format.MessageToString(stats))
    write_atomic(out / RESULT_FILE, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return Outcome(status, result, out / RESULT_FILE)


def write_atomic(path: Path, text: str) -> None:
    """Write text so readers never see a partial file; readable by other users."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    try:
        staged.write_text(text, encoding="utf-8")
        staged.chmod(0o644)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
