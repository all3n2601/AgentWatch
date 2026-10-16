"""Make text safe for live records: no paths, control characters or unbounded length."""

import re

MAX_REASON_CHARS = 512
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Absolute POSIX paths and Windows drive paths; they reveal host layout and user names.
_PATH = re.compile(r"(?:(?<![\w.])/(?:[^\s/'\"]+/)+[^\s/'\"]*|[A-Za-z]:\\[^\s'\"]+)")
_WHITESPACE = re.compile(r"\s+")


def redact_reason(text: str, limit: int = MAX_REASON_CHARS) -> str:
    """Return a nonblank, single-line, path-free reason of at most `limit` characters.

    Live reasons are shared with every viewer of a run, so full error details stay in local
    evidence and only this redacted form is sent.
    """
    cleaned = _CONTROL.sub(" ", text)
    # Lone surrogates cannot be stored as UTF-8 text.
    cleaned = cleaned.encode("utf-8", errors="replace").decode("utf-8")
    cleaned = _PATH.sub("<path>", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        cleaned = "No reason was recorded"
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned
