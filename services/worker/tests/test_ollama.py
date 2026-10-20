import argparse
import http.client
import io
import urllib.error
from datetime import datetime, timedelta

import pytest
from agentwatch_worker import ollama
from agentwatch_worker.ollama import (
    model_placement,
    normalized_model_name,
    observe_model_placement,
    ollama_base_url,
)


def loaded(size=1000, size_vram=1000, name="qwen3:1.7b"):
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "size": size,
                "size_vram": size_vram,
                "digest": "8f68893c685c",
                "details": {"quantization_level": "Q4_K_M"},
                "context_length": 4096,
            }
        ]
    }


@pytest.mark.parametrize(
    ("size_vram", "fraction"),
    [(1000, 1.0), (0, 0.0), (250, 0.25)],
)
def test_model_placement_reports_ollama_memory_split(size_vram, fraction):
    placement = model_placement(loaded(size_vram=size_vram), "qwen3:1.7b")
    assert placement["status"] == "measured"
    assert placement["gpu_memory_fraction"] == fraction
    assert placement["size_vram_bytes"] == size_vram
    assert placement["digest"] == "8f68893c685c"


def test_unloaded_model_is_unavailable_not_cpu():
    placement = model_placement({"models": []}, "qwen3:1.7b")
    assert placement["status"] == "unavailable"
    assert "gpu_memory_fraction" not in placement


@pytest.mark.parametrize(
    ("name", "normalized"),
    [
        ("qwen3", "qwen3:latest"),
        ("qwen3:1.7b", "qwen3:1.7b"),
        (" qwen3:1.7b ", "qwen3:1.7b"),
        ("library/qwen3:1.7b", "qwen3:1.7b"),
        ("registry.ollama.ai/library/qwen3", "qwen3:latest"),
        ("user/custom", "user/custom:latest"),
        ("localhost:5000/team/model", "localhost:5000/team/model:latest"),
        ("localhost:5000/team/model:v2", "localhost:5000/team/model:v2"),
    ],
)
def test_model_names_are_compared_in_ollamas_canonical_form(name, normalized):
    assert normalized_model_name(name) == normalized


@pytest.mark.parametrize("requested", ["qwen3", "qwen3:latest", "library/qwen3"])
def test_untagged_or_namespaced_model_matches_its_listed_name(requested):
    placement = model_placement(loaded(name="qwen3:latest"), requested)
    assert placement["status"] == "measured"


def test_different_tag_of_the_same_model_is_not_matched():
    placement = model_placement(loaded(name="qwen3:latest"), "qwen3:1.7b")
    assert placement["status"] == "unavailable"


def test_non_string_listed_names_are_ignored():
    response = loaded()
    response["models"].insert(0, {"name": None, "model": 7, "size": 1, "size_vram": 1})
    assert model_placement(response, "qwen3:1.7b")["status"] == "measured"


@pytest.mark.parametrize(
    "response",
    [
        {},
        [],
        loaded(size=0, size_vram=0),
        loaded(size_vram=2000),
        loaded(size_vram=-1),
        loaded(size_vram=True),
        loaded(size="1000"),
    ],
)
def test_inconsistent_placement_is_malformed(response):
    placement = model_placement(response, "qwen3:1.7b")
    assert placement["status"] == "malformed"
    assert "gpu_memory_fraction" not in placement


@pytest.mark.parametrize(
    ("failure", "reported"),
    [
        (urllib.error.URLError("connection refused"), "connection refused"),
        (ConnectionResetError("connection reset"), "connection reset"),
        (http.client.IncompleteRead(b"partial"), "IncompleteRead"),
    ],
)
def test_placement_request_failure_is_recorded_as_unavailable(failure, reported):
    def unreachable():
        raise failure

    placement = observe_model_placement(unreachable, "qwen3:1.7b")
    assert placement["status"] == "unavailable"
    assert placement["reason"].startswith("placement request failed:")
    assert reported in placement["reason"]
    assert "gpu_memory_fraction" not in placement
    assert datetime.fromisoformat(placement["observed_at"]).utcoffset() == timedelta(0)
    assert placement["read_seconds"] >= 0


def respond_with(body: bytes):
    def urlopen(*_args, **_kwargs):
        return io.BytesIO(body)

    return urlopen


@pytest.mark.parametrize("body", [b"\xff", b"not json", b'{"models": '])
def test_undecodable_placement_response_is_malformed(monkeypatch, body):
    monkeypatch.setattr(ollama.urllib.request, "urlopen", respond_with(body))
    placement = observe_model_placement(
        lambda: ollama.ollama_loaded_models("http://ollama.test"), "qwen3:1.7b"
    )
    assert placement["status"] == "malformed"
    assert placement["reason"].startswith("placement response is not valid UTF-8 JSON:")
    assert "gpu_memory_fraction" not in placement


def test_placement_programming_errors_are_not_contained():
    def broken():
        raise KeyError("models")

    with pytest.raises(KeyError):
        observe_model_placement(broken, "qwen3:1.7b")


@pytest.mark.parametrize(
    "url", ["127.0.0.1:11434", "ftp://127.0.0.1:11434", "http://", "localhost", ""]
)
def test_ollama_url_must_be_http(url):
    with pytest.raises(argparse.ArgumentTypeError):
        ollama_base_url(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11435", "https://ollama.example:443/"])
def test_valid_ollama_urls_are_accepted(url):
    assert ollama_base_url(url) == url
