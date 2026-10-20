import pytest
from agent_test_doubles import FakeLiveServer
from agentwatch_agent.tasks import load_task


@pytest.fixture
def mean_task():
    return load_task("fix-mean-empty")


@pytest.fixture
def server():
    return FakeLiveServer()
