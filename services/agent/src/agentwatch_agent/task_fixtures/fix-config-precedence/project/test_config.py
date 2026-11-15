import unittest

from config import load_settings
from defaults import DEFAULTS


class SettingsPrecedenceTests(unittest.TestCase):
    def test_unset_keys_come_from_the_defaults(self):
        self.assertEqual(load_settings({}, {}), DEFAULTS)

    def test_settings_file_overrides_the_defaults(self):
        self.assertEqual(load_settings({"retries": 5}, {})["retries"], 5)

    def test_environment_overrides_the_settings_file(self):
        settings = load_settings({"log_level": "warning"}, {"log_level": "debug"})
        self.assertEqual(settings["log_level"], "debug")

    def test_sources_are_not_modified(self):
        file_settings, env_settings = {"retries": 5}, {"retries": 7}
        load_settings(file_settings, env_settings)
        self.assertEqual(file_settings, {"retries": 5})
        self.assertEqual(env_settings, {"retries": 7})
        self.assertEqual(DEFAULTS["retries"], 3)
