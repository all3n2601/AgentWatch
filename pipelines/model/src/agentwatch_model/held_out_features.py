"""Feature availability: how often each model input is present, held-out steps against train.

The A06 sweeps ran on vLLM, most of dataset v1 on Ollama, so a feature the classifier learned
from can be missing or constant on held-out steps. This summary puts that next to the scores;
it is descriptive and changes no score.

Missing means null or NaN, the values `features.feature_matrix` passes to LightGBM as NaN; a
measured zero is present. A group with no rows, or a feature with no present value, has None
fractions, never 0.
"""

import math
from collections import Counter

from agentwatch_model.features import FEATURES

FEATURE_AVAILABILITY_DEFINITION = (
    "Per model feature, for the train split, the scored held-out steps and each held-out "
    "configuration: rows, the fraction whose value is null or NaN (missing_fraction), the "
    "number of distinct present values, and the share of present values equal to the most "
    "common one (most_common_fraction). Fractions are null when there is nothing to divide by."
)


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def availability(rows: list[dict], name: str) -> dict:
    present = [row.get(name) for row in rows if not is_missing(row.get(name))]
    counts = Counter(present)
    return {
        "rows": len(rows),
        "missing_fraction": (len(rows) - len(present)) / len(rows) if rows else None,
        "distinct_values": len(counts),
        "most_common_fraction": counts.most_common(1)[0][1] / len(present) if present else None,
    }


def feature_availability(held_out_rows: list[dict], train_rows: list[dict]) -> dict:
    """Availability of every feature in FEATURES order, held-out rows also by configuration."""
    by_configuration: dict[str, list[dict]] = {}
    for row in held_out_rows:
        by_configuration.setdefault(row["configuration"], []).append(row)
    return {
        name: {
            "train": availability(train_rows, name),
            "held_out": availability(held_out_rows, name),
            "held_out_by_configuration": {
                configuration: availability(by_configuration[configuration], name)
                for configuration in sorted(by_configuration)
            },
        }
        for name in FEATURES
    }
