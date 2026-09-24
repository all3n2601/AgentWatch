from model_passport_worker.app import health


def test_health() -> None:
    assert health() == {"status": "ok", "service": "worker"}

