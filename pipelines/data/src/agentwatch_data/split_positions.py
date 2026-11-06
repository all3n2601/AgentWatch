"""The split names and the `split-v1` hash position shared by every split method.

`split-v1` assigns each group by this hash alone; `split-v2` (split_assignment) keeps the same
position for every group its assignment file does not send to test. Both splits and
split_assignment import it, so neither has to import the other's internals.
"""

import hashlib

HASH_METHOD = "split-v1"  # hashed with each group key; a new value redraws every split
SPLITS = ("train", "validation", "test")


def assign(group: str, fractions: dict[str, float]) -> str:
    """The split of one group, from a hash of its key: stable across releases."""
    digest = hashlib.sha256(f"{HASH_METHOD}:{group}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name in SPLITS[:-1]:
        cumulative += fractions[name]
        if position < cumulative:
            return name
    return SPLITS[-1]
