from agentwatch_agent.stats import p90, summarize


def test_p90_is_one_observed_element_with_one_definition():
    values = [float(v) for v in range(1, 11)]  # 1..10
    # Index round(0.9 * 9) = 8: the ninth value, never an interpolation.
    assert p90(values) == 9.0
    assert summarize(values)["p90"] == p90(values)
    assert p90([5.0]) == 5.0
    assert p90([]) is None
