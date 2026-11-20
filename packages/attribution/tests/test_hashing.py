import hashlib
import math
from pathlib import Path

import agentwatch_data.extract
import agentwatch_data.files
import agentwatch_data.schema
import agentwatch_model.metrics
import pytest
from agentwatch_attribution import classes, hashing


def test_canonical_hash_is_key_order_independent() -> None:
    assert hashing.canonical_sha256({"a": 1, "b": [1, 2]}) == hashing.canonical_sha256(
        {"b": [1, 2], "a": 1}
    )
    assert hashing.canonical_sha256({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_canonical_hash_refuses_nan() -> None:
    with pytest.raises(ValueError):
        hashing.canonical_sha256({"a": math.nan})


def test_file_hash_matches_hashlib(tmp_path: Path) -> None:
    content = bytes(range(256)) * (5 * 1024 + 3)  # more than one 1 MiB block
    path = tmp_path / "blob.bin"
    path.write_bytes(content)
    assert len(content) > 1 << 20
    assert hashing.file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_is_number_rejects_bool_nan_inf_and_strings() -> None:
    assert classes.is_number(1)
    assert classes.is_number(0.5)
    for value in (True, False, math.nan, math.inf, -math.inf, "1", None):
        assert not classes.is_number(value)


def test_class_constants_match_the_contract() -> None:
    assert classes.RESOURCE_CLASSES == ("none", "cpu_sandbox", "inference", "retrieval")
    assert classes.NONE == "none"


def test_data_extract_hash_agrees_with_the_shared_hash() -> None:
    # agentwatch_data.extract keeps its own definition: that module's source is fingerprinted
    # into the frozen labels-v2 label version, so it cannot import this one without changing
    # that version. The two must therefore agree byte for byte.
    for value in ({"b": [1, 2.5, None], "a": "é"}, [], 3, "text"):
        assert agentwatch_data.extract.canonical_sha256(value) == hashing.canonical_sha256(value)


def test_data_package_reexports_the_same_functions() -> None:
    assert agentwatch_data.files.file_sha256 is hashing.file_sha256
    assert agentwatch_data.schema.RESOURCE_CLASSES is classes.RESOURCE_CLASSES
    assert agentwatch_model.metrics.CLASSES is classes.RESOURCE_CLASSES
    assert agentwatch_model.metrics.NONE is classes.NONE
    assert agentwatch_model.metrics.is_number is classes.is_number
