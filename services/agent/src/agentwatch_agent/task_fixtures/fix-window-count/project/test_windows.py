import unittest

from windows import windows


class WindowTests(unittest.TestCase):
    def test_every_window_is_returned_in_order(self):
        self.assertEqual(windows([1, 2, 3], 2), [[1, 2], [2, 3]])

    def test_window_equal_to_input_length(self):
        self.assertEqual(windows([1, 2], 2), [[1, 2]])

    def test_window_longer_than_input_is_empty(self):
        self.assertEqual(windows([1, 2], 3), [])

    def test_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            windows([1], 0)
