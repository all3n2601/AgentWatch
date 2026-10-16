import unittest

from stats import mean, total


class TotalTests(unittest.TestCase):
    def test_total_of_values(self):
        self.assertEqual(total([1, 2, 3]), 6)

    def test_total_of_empty_input_is_zero(self):
        self.assertEqual(total([]), 0)


class MeanTests(unittest.TestCase):
    def test_mean_of_values(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_mean_of_signed_values(self):
        self.assertEqual(mean([-3, 1]), -1)

    def test_mean_of_empty_input_is_none(self):
        self.assertIsNone(mean([]))
