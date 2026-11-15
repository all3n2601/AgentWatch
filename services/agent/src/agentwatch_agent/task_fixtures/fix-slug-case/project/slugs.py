"""URL slugs for article titles."""


def slugify(title):
    """Return the URL slug for an article title."""
    return "-".join(title.split())
