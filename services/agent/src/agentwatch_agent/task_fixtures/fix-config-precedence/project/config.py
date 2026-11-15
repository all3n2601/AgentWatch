"""Combine a job's settings from its three sources."""

from defaults import DEFAULTS


def load_settings(file_settings, env_settings):
    """Return the job's settings from the defaults, the settings file and the environment."""
    merged = dict(env_settings)
    merged.update(file_settings)
    for key, value in DEFAULTS.items():
        merged.setdefault(key, value)
    return merged
