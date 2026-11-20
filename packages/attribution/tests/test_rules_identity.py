from agentwatch_attribution.rules import load_rules


def test_packaged_rules_keep_their_published_identity():
    # Moving the file must not change the version: it hashes the parsed thresholds, not the path.
    assert load_rules().version == "rules-v1+1bf54c19"
