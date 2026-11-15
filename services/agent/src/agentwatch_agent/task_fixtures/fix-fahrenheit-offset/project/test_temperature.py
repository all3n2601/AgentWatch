import unittest

from temperature import to_fahrenheit


class FahrenheitTests(unittest.TestCase):
    def test_freezing_point(self):
        self.assertEqual(to_fahrenheit(0), 32)

    def test_boiling_point(self):
        self.assertEqual(to_fahrenheit(100), 212)

    def test_body_temperature(self):
        self.assertAlmostEqual(to_fahrenheit(37), 98.6)
