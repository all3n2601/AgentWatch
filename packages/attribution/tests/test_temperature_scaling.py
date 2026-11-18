import numpy as np
from agentwatch_attribution.calibration import PROBABILITY_FLOOR, scale


def test_scale_keeps_the_top_class_and_sums_to_one() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    for temperature in (0.5, 1.0, 3.0):
        scaled = scale(probabilities, temperature)
        assert scaled.argmax(axis=1).tolist() == probabilities.argmax(axis=1).tolist()
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
    np.testing.assert_allclose(scale(probabilities, 1.0), probabilities)


def test_scale_floor_survives_zero_probability() -> None:
    scaled = scale(np.array([[1.0, 0.0, 0.0]]), 2.0)
    assert np.isfinite(scaled).all()
    assert (scaled > 0).all()
    assert PROBABILITY_FLOOR == 1e-12
