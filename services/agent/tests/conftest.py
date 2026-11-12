import pytest
from agent_test_doubles import FakeLiveServer
from agentwatch_agent.tasks import load_task
from test_sweep_steps import synthetic_sweep


@pytest.fixture
def mean_task():
    return load_task("fix-mean-empty")


@pytest.fixture
def server():
    return FakeLiveServer()


@pytest.fixture
def capacity_sweep_dir(tmp_path, monkeypatch):
    """A synthetic held-out capacity sweep: the stub driver's folder with fixture run folders."""
    return synthetic_sweep(tmp_path, monkeypatch)
