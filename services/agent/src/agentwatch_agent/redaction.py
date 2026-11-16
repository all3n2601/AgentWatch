"""Make text safe to share: no credentials, email addresses, host paths or control characters.

`redact_text` (method `trace-redaction-v1`) replaces each sensitive value with a placeholder
and counts replacements by kind. `redact_reason` applies it to live reasons, which every
viewer of a run sees, and also removes control characters and bounds the length.

The rules are deliberately specific (known token shapes, `name=value` where the name ends in
a secret word, URL user info, email addresses, absolute paths) so identifiers, hashes, code
and relative paths survive. They cannot find every secret: an unlabeled random string is
indistinguishable from data, so shared evidence still needs review.
"""

import re
from collections import Counter

REDACTION_METHOD = "trace-redaction-v1"
MAX_REASON_CHARS = 512
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_SECRET_NAME = r"[\w-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)"
# Top-level folders of macOS, Linux and container filesystems that hold host-specific paths.
_HOST_ROOTS = (
    "Users|home|root|private|var|tmp|opt|mnt|media|Volumes|srv|etc|usr|run|proc|sys|data|app"
    "|workspaces?|Library|Applications|System|nix|snap"
)
# Applied in order; earlier rules replace text that later, broader rules would split.
_RULES: tuple[tuple[str, re.Pattern, str], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
        ),
        "<private-key>",
    ),
    # scheme://user:password@host keeps the scheme and host.
    (
        "url_credentials",
        re.compile(r"(\b[a-zA-Z][\w+.-]*://)[^/\s:@]+:[^/\s@]+@"),
        r"\1<credentials>@",
    ),
    ("bearer", re.compile(r"(?i)\b(bearer|basic)\s+[\w.~+/=-]{8,}"), r"\1 <credential>"),
    (
        "token",
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|sk-[\w-]{20,}"
            r"|AKIA[0-9A-Z]{16}|xox[abposr]-[\w-]{10,}|hf_[A-Za-z0-9]{30,}|AIza[\w-]{35})"
        ),
        "<credential>",
    ),
    # The name must end in the secret word, so `max_tokens: 512` is not a credential.
    (
        "secret_assignment",
        re.compile(rf"(?i)\b({_SECRET_NAME})(\s*[=:]\s*)([\"']?)[^\s\"',;]{{4,}}"),
        r"\1\2\3<credential>",
    ),
    ("email", re.compile(r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<email>"),
    # Absolute paths under filesystem roots, and Windows drive paths, reveal host layout and
    # user names. Only known roots count, so API routes such as `/api/ps` are kept. A slash
    # after `:` or `/` belongs to a URL, and a path after a placeholder (`<project>/tests/x.py`)
    # is already relative; both are kept.
    (
        "path",
        re.compile(
            rf"(?:(?<![\w.>:/])/(?:{_HOST_ROOTS})(?:/[^\s'\"]*)?(?![\w-])"
            r"|[A-Za-z]:\\[^\s'\"]+)"
        ),
        "<path>",
    ),
)


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Return the text with sensitive values replaced, and the replacements by kind."""
    counts: Counter[str] = Counter()
    for kind, pattern, replacement in _RULES:
        text, replaced = pattern.subn(replacement, text)
        if replaced:
            counts[kind] += replaced
    return text, dict(counts)


def redact_reason(text: str, limit: int = MAX_REASON_CHARS) -> str:
    """Return a nonblank, single-line, redacted reason of at most `limit` characters.

    Live reasons are shared with every viewer of a run, so full error details stay in local
    evidence and only this redacted form is sent.
    """
    cleaned = _CONTROL.sub(" ", text)
    # Lone surrogates cannot be stored as UTF-8 text.
    cleaned = cleaned.encode("utf-8", errors="replace").decode("utf-8")
    cleaned, _ = redact_text(cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        cleaned = "No reason was recorded"
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned
