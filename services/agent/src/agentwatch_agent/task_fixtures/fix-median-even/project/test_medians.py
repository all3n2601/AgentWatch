import unittest

from medians import median


class MedianTests(unittest.TestCase):
    def test_median_of_odd_count(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_median_of_single_value(self):
        self.assertEqual(median([5]), 5)

    def test_median_of_even_count_is_mean_of_middle_pair(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_median_does_not_modify_its_input(self):
        values = [3, 1, 2]
        median(values)
        self.assertEqual(values, [3, 1, 2])
