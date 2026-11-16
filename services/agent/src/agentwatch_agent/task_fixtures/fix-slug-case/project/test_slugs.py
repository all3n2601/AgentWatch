import unittest

from slugs import slugify


class SlugTests(unittest.TestCase):
    def test_words_are_joined_with_hyphens(self):
        self.assertEqual(slugify("release notes"), "release-notes")

    def test_slug_is_lowercase(self):
        self.assertEqual(slugify("Release Notes"), "release-notes")

    def test_extra_spaces_are_ignored(self):
        self.assertEqual(slugify("  Release   Notes "), "release-notes")
