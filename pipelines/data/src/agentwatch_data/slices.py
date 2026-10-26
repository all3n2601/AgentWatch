"""Slice a dataset release to find where its data is skewed, and say how to mitigate it.

    python -m agentwatch_data.slices --release data/datasets/<version> [--split <split file>]
    python -m agentwatch_data.slices --release data/datasets/<version> --newest-split

The dataset has no demographic attributes. Its slices are the attributes along which the data,
or a model trained on it, could be skewed: the task, the experiment kind, the label, the
condition and injection method, the agent model, the source, the telemetry status, and the
availability of PSI and cgroup readings. Availability is not the host platform: Linux hosts can
lack cgroup readings too, so a slice without them is not a macOS slice.

For each slice the report records its rows, runs, and share of the release; its label balance;
how often each nullable reading is missing (missing stays distinct from a measured zero); its
rows in each split; and how often the rule baseline agrees with the label. It then flags
slices against thresholds declared before any model is evaluated, and lists a mitigation for
each flag, with per-slice balancing weights for re-weighting.

This is a data-level analysis. The rule-baseline agreement is not a model evaluation: on
today's CPU-only data the label is the rule's prediction on every row. Per-slice model metrics
(Fairlearn's MetricFrame on the frozen test split) belong to model evaluation, once a model
and a non-empty training split exist.

The report is derived from the release, so it is written outside the versioned datasets store
(by default under .data/reports/<release>/), as slices.json and a readable slices.md.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agentwatch_data.build import DATASET_FILE
from agentwatch_data.files import json_text, write_atomic
from agentwatch_data.splits import GROUP_KEYS, SPLITS, SPLITS_DIR

SLICE_METHOD = "slices-v1"  # bump when attributes, metrics, or thresholds change
ATTRIBUTES = (
    "task_id",
    "kind",
    "label_resource",
    "condition",
    "injection_method",
    "agent_model",
    "source",
    "telemetry_status",
    "cpu_pressure_available",
    "cgroup_available",
)
TARGET = "label_resource"
NO_RESOURCE = "none"  # the label of a step no resource slowed
RULE = "rule_prediction"
# Nullable readings: null means the reading was missing or unsupported, never zero.
READINGS = ("telemetry_coverage", "cpu_pressure_max_avg10", "cgroup_throttled_usec_delta")
REQUIRED_COLUMNS = tuple(dict.fromkeys(("run_id", RULE, *ATTRIBUTES, *READINGS, *GROUP_KEYS)))
# condition and injection_method decide the label by design (contended runs are the positives),
# so single-class slices and label-rate gaps across them are expected, not bias; the target
# cannot be compared with itself.
LABEL_GAP_EXEMPT = (TARGET, "condition", "injection_method")
# They also split the rows exactly as the label does, so their missing-readings gaps repeat the
# label's; the label alone reports them, so one finding is not counted three times.
MISSING_GAP_EXEMPT = ("condition", "injection_method")
# An availability flag defines whether its reading is missing, so that gap is a tautology.
READING_AVAILABILITY = {
    "cpu_pressure_available": "cpu_pressure_max_avg10",
    "cgroup_available": "cgroup_throttled_usec_delta",
}
# Slice values share one namespace with these markers and with "true"/"false" for booleans;
# an attribute whose real values could equal one of them would need distinct markers.
UNRECORDED = "(unrecorded)"  # a null attribute, such as agent_model on non-agent runs
UNASSIGNED = "unassigned"  # rows whose task the split file does not place
EXCLUDED = "excluded"  # rows of runs the split excluded (a source not allowed in test)

# Declared 2026-10-26, before any model is evaluated; never tuned to clear a flag.
MIN_SLICE_ROWS = 30  # fewer rows cannot support a per-slice metric
MIN_SLICE_SHARE = 0.05
MAX_POSITIVE_RATE_GAP = 0.20  # between values of one attribute
MAX_MISSING_RATE_GAP = 0.20  # between values of one attribute, per reading

UNDER_REPRESENTED = "under_represented"
SINGLE_CLASS = "single_class"
POSITIVE_RATE_GAP = "positive_rate_gap"
MISSING_READINGS_GAP = "missing_readings_gap"
ABSENT_FROM_SPLIT = "absent_from_split"
MITIGATIONS = {
    UNDER_REPRESENTED: (
        "Collect more runs for this slice (the only fix that adds evidence); until then, "
        "re-weight training with the slice's balancing weight."
    ),
    SINGLE_CLASS: (
        "Collect runs of the missing classes in this slice; a model cannot learn or be "
        "evaluated on a class the slice never shows."
    ),
    POSITIVE_RATE_GAP: (
        "Stratify or re-weight training by this attribute, and collect contended runs for "
        "the slices with the lower positive rate."
    ),
    MISSING_READINGS_GAP: (
        "Keep the readings missing (never zero-filled); report model metrics per slice so a "
        "platform without the reading is judged separately; collect on Linux where it exists."
    ),
    ABSENT_FROM_SPLIT: (
        "The split cannot train on or evaluate this slice; regroup once step events carry a "
        "real task_id, and collect more tasks so every slice reaches each split."
    ),
}


class SliceError(RuntimeError):
    """The release or split cannot be sliced; nothing was written."""


@dataclass(frozen=True)
class Assignment:
    """Where a split file put each row: by its group (the split's group key), unless its run
    was excluded."""

    groups: dict[str, str]
    excluded_runs: frozenset[str]
    group_key: str = GROUP_KEYS[0]

    def split_of(self, row: dict[str, object]) -> str:
        if row["run_id"] in self.excluded_runs:
            return EXCLUDED
        return self.groups.get(str(row[self.group_key]), UNASSIGNED)


@dataclass(frozen=True)
class SliceReport:
    report: dict[str, object]
    json_path: Path
    markdown_path: Path


def load_rows(release: Path) -> list[dict[str, object]]:
    path = release / DATASET_FILE
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        raise SliceError(f"{path} is unreadable: {error}") from error
    missing = [name for name in REQUIRED_COLUMNS if name not in table.column_names]
    if missing:
        raise SliceError(f"{path} lacks columns the slices need: {missing}")
    if table.num_rows == 0:
        raise SliceError(f"{path} has no rows to slice")
    return table.select(list(REQUIRED_COLUMNS)).to_pylist()


def load_assignment(split: Path, release: Path) -> Assignment:
    """The split each row belongs to, from a split file made for this release."""
    try:
        document = json.loads(split.read_text())
        groups = document["groups"]
        split_release = document["release"]
        group_key = document["contract"]["group_key"]
        excluded_runs = frozenset(str(run["run_id"]) for run in document["excluded_runs"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SliceError(f"split file {split} is unreadable: {error}") from error
    if split_release != release.name:
        raise SliceError(f"split file {split} is for {split_release}, not {release.name}")
    if group_key not in GROUP_KEYS:
        raise SliceError(f"split file {split} groups by {group_key!r}, not one of {GROUP_KEYS}")
    # An empty assignment would leave every row unassigned: a report that looks computed but
    # places nothing.
    if (
        not isinstance(groups, dict)
        or not groups
        or not all(isinstance(task, str) and name in SPLITS for task, name in groups.items())
    ):
        raise SliceError(f"split file {split} has an invalid task assignment")
    return Assignment(groups, excluded_runs, group_key)


def newest_split(release: Path) -> Path | None:
    """The most recently written split file of a release in its datasets store, if any."""
    directory = release.parent / SPLITS_DIR / release.name
    try:
        candidates = [(path.stat().st_mtime, path) for path in directory.glob("*.json")]
    except OSError as error:
        raise SliceError(f"split directory {directory} is unreadable: {error}") from error
    return max(candidates)[1] if candidates else None


def slice_value(value: object) -> str:
    if value is None:
        return UNRECORDED
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def describe(rows: list[dict[str, object]], total: int, splits: Assignment | None) -> dict:
    """Counts and rates for one slice's rows."""
    labels = Counter(str(row[TARGET]) for row in rows)
    positives = sum(count for label, count in labels.items() if label != NO_RESOURCE)
    described: dict[str, object] = {
        "rows": len(rows),
        "runs": len({row["run_id"] for row in rows}),
        "share": round(len(rows) / total, 6),
        "labels": dict(sorted(labels.items())),
        "positive_rate": round(positives / len(rows), 6),
        "missing_rate": {
            reading: round(sum(row[reading] is None for row in rows) / len(rows), 6)
            for reading in READINGS
        },
        "rule_agreement": round(sum(row[RULE] == row[TARGET] for row in rows) / len(rows), 6),
    }
    if splits is not None:
        per_split = Counter(splits.split_of(row) for row in rows)
        described["splits"] = {
            name: per_split.get(name, 0) for name in (*SPLITS, EXCLUDED, UNASSIGNED)
        }
    return described


def flags_for(attribute: str, slices: dict[str, dict], split_rows: Counter | None) -> list[dict]:
    flags = []

    def flag(kind: str, value: str | None, detail: str) -> None:
        flags.append(
            {
                "attribute": attribute,
                "value": value,
                "kind": kind,
                "detail": detail,
                "mitigation": MITIGATIONS[kind],
            }
        )

    for value, described in slices.items():
        if described["rows"] < MIN_SLICE_ROWS or described["share"] < MIN_SLICE_SHARE:
            flag(
                UNDER_REPRESENTED,
                value,
                f"{described['rows']} rows ({described['share']:.1%} of the release); the "
                f"declared minimum is {MIN_SLICE_ROWS} rows and {MIN_SLICE_SHARE:.0%}",
            )
        if attribute not in LABEL_GAP_EXEMPT and len(described["labels"]) == 1:
            (only,) = described["labels"]
            flag(SINGLE_CLASS, value, f"every row is labeled {only!r}")
        if split_rows is not None:
            for name in ("train", "test"):
                if split_rows[name] and not described["splits"][name]:
                    flag(ABSENT_FROM_SPLIT, value, f"no rows in the {name} split")
    if len(slices) < 2:
        return flags
    if attribute not in LABEL_GAP_EXEMPT:
        rates = {value: described["positive_rate"] for value, described in slices.items()}
        gap = max(rates.values()) - min(rates.values())
        if gap > MAX_POSITIVE_RATE_GAP:
            flag(
                POSITIVE_RATE_GAP,
                None,
                f"positive rate ranges {min(rates.values()):.1%}–{max(rates.values()):.1%} "
                f"(gap {gap:.1%}, declared maximum {MAX_POSITIVE_RATE_GAP:.0%})",
            )
    for reading in READINGS:
        if attribute in MISSING_GAP_EXEMPT or READING_AVAILABILITY.get(attribute) == reading:
            continue
        rates = {value: described["missing_rate"][reading] for value, described in slices.items()}
        gap = max(rates.values()) - min(rates.values())
        if gap > MAX_MISSING_RATE_GAP:
            flag(
                MISSING_READINGS_GAP,
                None,
                f"{reading} is missing on {min(rates.values()):.1%}–{max(rates.values()):.1%} "
                f"of rows (gap {gap:.1%}, declared maximum {MAX_MISSING_RATE_GAP:.0%})",
            )
    return flags


def balancing_weights(slices: dict[str, dict], total: int) -> dict[str, float]:
    """Row weights that give every value of an attribute the same total weight."""
    return {
        value: round(total / (len(slices) * described["rows"]), 6)
        for value, described in slices.items()
    }


def analyze(rows: list[dict[str, object]], release: str, splits: Assignment | None) -> dict:
    total = len(rows)
    split_rows = None
    if splits is not None:
        split_rows = Counter(splits.split_of(row) for row in rows)
    attributes, flags = {}, []
    for attribute in ATTRIBUTES:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(slice_value(row[attribute]), []).append(row)
        slices = {value: describe(grouped[value], total, splits) for value in sorted(grouped)}
        attributes[attribute] = {
            "slices": slices,
            "balancing_weights": balancing_weights(slices, total),
        }
        flags.extend(flags_for(attribute, slices, split_rows))
    return {
        "method": SLICE_METHOD,
        "release": release,
        "rows": total,
        "runs": len({row["run_id"] for row in rows}),
        "split_rows": dict(sorted(split_rows.items())) if split_rows is not None else None,
        "thresholds": {
            "min_slice_rows": MIN_SLICE_ROWS,
            "min_slice_share": MIN_SLICE_SHARE,
            "max_positive_rate_gap": MAX_POSITIVE_RATE_GAP,
            "max_missing_rate_gap": MAX_MISSING_RATE_GAP,
        },
        "attributes": attributes,
        "flags": flags,
    }


def flag_location(flag: dict, separator: str) -> str:
    """Where a flag applies: one slice of an attribute, or the attribute as a whole."""
    if flag["value"] is None:
        return flag["attribute"]
    return f"{flag['attribute']}{separator}{flag['value']}"


def table_cell(text: str) -> str:
    """Text that stays inside one Markdown table cell."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def markdown(report: dict) -> str:
    lines = [
        f"# Slices of {report['release']}",
        "",
        f"{report['rows']} rows from {report['runs']} runs, sliced with {report['method']}.",
    ]
    if report["split_rows"] is not None:
        counts = ", ".join(f"{name} {count}" for name, count in report["split_rows"].items())
        lines.append(f"Rows per split: {counts}.")
    lines += ["", f"## Flags ({len(report['flags'])})", ""]
    if not report["flags"]:
        lines.append("No slice crossed a declared threshold.")
    for flag in report["flags"]:
        where = flag_location(flag, " = ")
        lines.append(f"- **{flag['kind']}** — {where}: {flag['detail']}. {flag['mitigation']}")
    for attribute, analysis in report["attributes"].items():
        has_splits = report["split_rows"] is not None
        header = "| Value | Rows | Share | Positive rate | Rule agreement | Missing readings |"
        lines += ["", f"## {attribute}", "", header + (" Train / test |" if has_splits else "")]
        lines.append("|---" * (7 if has_splits else 6) + "|")
        for value, described in analysis["slices"].items():
            missing = ", ".join(
                f"{reading} {rate:.0%}" for reading, rate in described["missing_rate"].items()
            )
            cells = [
                table_cell(value),
                str(described["rows"]),
                f"{described['share']:.1%}",
                f"{described['positive_rate']:.1%}",
                f"{described['rule_agreement']:.1%}",
                missing,
            ]
            if has_splits:
                cells.append(f"{described['splits']['train']} / {described['splits']['test']}")
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "Rule agreement is not a model evaluation: on CPU-only data the label is the "
            "rule's prediction on every row."
        ),
    ]
    return "\n".join(lines) + "\n"


def slice_release(release: Path, split: Path | None, output: Path) -> SliceReport:
    rows = load_rows(release)
    splits = load_assignment(split, release) if split is not None else None
    report = analyze(rows, release.name, splits)
    json_path, markdown_path = output / "slices.json", output / "slices.md"
    try:
        write_atomic(json_path, json_text(report))
        write_atomic(markdown_path, markdown(report))
    except OSError as error:
        # Never leave this run's JSON beside an older run's Markdown.
        json_path.unlink(missing_ok=True)
        raise SliceError(f"the report could not be written to {output}: {error}") from error
    return SliceReport(report, json_path, markdown_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--split", type=Path, help="a split file of this release")
    choice.add_argument(
        "--newest-split",
        action="store_true",
        help="use the release's most recently written split file, if it has one",
    )
    parser.add_argument(
        "--output", type=Path, help="report directory (default: .data/reports/<release>)"
    )
    args = parser.parse_args(argv)
    output = args.output or Path(".data/reports") / args.release.name
    try:
        split = newest_split(args.release) if args.newest_split else args.split
        result = slice_release(args.release, split, output)
    except SliceError as error:
        print(f"Slicing failed: {error}", file=sys.stderr)
        return 1
    print(f"slices: {result.json_path}")
    print(f"summary: {result.markdown_path}")
    for flag in result.report["flags"]:
        print(f"  flag {flag['kind']}: {flag_location(flag, '=')}: {flag['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
