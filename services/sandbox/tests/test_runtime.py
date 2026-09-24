from model_passport_sandbox import runtime_name


def test_runtime_name() -> None:
    assert runtime_name() == "model-passport-sandbox"

