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
# Top-level folders of macOS, Linux and container filesystems that hold host-specific paths,
# plus the MSYS/Git Bash drive form (`/c/Users/...`).
_HOST_ROOTS = (
    "Users|home|root|private|var|tmp|opt|mnt|media|Volumes|srv|etc|usr|run|proc|sys|data|app"
    "|workspaces?|Library|Applications|System|nix|snap|[A-Za-z]/Users|[A-Za-z]/home"
)
# Characters that end an unquoted value: whitespace, quotes, separators and call/index syntax.
_VALUE_END = r"\s\"',;()\[\]{}"
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
    # In an Authorization header any scheme value is a credential.
    (
        "bearer",
        re.compile(r"(?i)\b(authorization\s*[:=]\s*[\"']?)(bearer|basic)\s+[\w.~+/=-]{8,}"),
        r"\1\2 <credential>",
    ),
    # Elsewhere only a token-shaped value (16+ characters with a digit) counts, so prose such as
    # "basic configuration mismatch" or "Basic authentication failed" is kept.
    (
        "bearer",
        re.compile(r"(?i)\b(bearer|basic)\s+(?=[\w.~+/=-]*\d)[\w.~+/=-]{16,}"),
        r"\1 <credential>",
    ),
    (
        "token",
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|sk-[\w-]{20,}"
            r"|AKIA[0-9A-Z]{16}|xox[abposr]-[\w-]{10,}|hf_[A-Za-z0-9]{30,}|AIza[\w-]{35})"
        ),
        "<credential>",
    ),
    # `name = value` where the name ends in a secret word (so `max_tokens: 512` is not one),
    # including a quoted JSON name (`"password": "..."`). A quoted value is always redacted.
    (
        "secret_assignment",
        re.compile(rf"(?i)\b({_SECRET_NAME})([\"']?)(\s*[=:]\s*)([\"'])[^\"'\n]{{4,}}\4"),
        r"\1\2\3\4<credential>\4",
    ),
    # An unquoted value must look like a token: 8+ characters with a letter and a digit, not
    # followed by call or index syntax. Prose and code (`token: expired`, `next_token =
    # self.advance()`, `secret=None`) are kept.
    (
        "secret_assignment",
        re.compile(
            rf"(?i)\b({_SECRET_NAME})([\"']?)(\s*[=:]\s*)"
            rf"(?=[^{_VALUE_END}]*\d)(?=[^{_VALUE_END}]*[A-Za-z])[^{_VALUE_END}]{{8,}}(?![(\[])"
        ),
        r"\1\2\3<credential>",
    ),
    ("email", re.compile(r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<email>"),
    # A file URL names a host path, unlike other URLs.
    ("path", re.compile(r"(?i)\bfile://[^\s'\"]*"), "<path>"),
    # UNC paths: \\server\share\...
    ("path", re.compile(r"(?<![\w\\])\\\\[^\s\\'\"]+\\[^\s'\"]*"), "<path>"),
    # Absolute paths under filesystem roots, and Windows drive paths with either separator,
    # reveal host layout and user names. Only known roots count, so API routes such as
    # `/api/ps` are kept. A slash after `:` or `/` belongs to a URL, and a path after a
    # placeholder (`<project>/tests/x.py`) is already relative; both are kept. A path containing
    # spaces is replaced only up to the first space (see the limitations).
    (
        "path",
        re.compile(
            rf"(?:(?<![\w.>:/])/(?:{_HOST_ROOTS})(?:/[^\s'\"]*)?(?![\w-])"
            r"|(?<![\w])[A-Za-z]:[\\/](?![\\/])[^\s'\"]+)"
        ),
        "<path>",
    ),
)
LIMITATIONS = (
    "Pattern-based: an unlabeled random secret is indistinguishable from data",
    "A path containing spaces is replaced only up to the first space; later folder names stay",
    "Host names inside URLs are kept",
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
