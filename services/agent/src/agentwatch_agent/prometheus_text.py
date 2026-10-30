"""Read a service's Prometheus text exposition: one bounded HTTP fetch and a strict parser.

Used by the runner's vLLM and DCGM probes. The exposition is untrusted input: its size and
line count are capped, a line that does not parse is counted rather than guessed at, and
non-finite values are kept so callers can report them as malformed instead of as readings.
"""

import re
import urllib.request
from dataclasses import dataclass

# vLLM's exposition was about 64 KB on the laptop (October 27, 2026); this leaves headroom
# without letting a misbehaving endpoint fill the runner's memory every round.
MAX_EXPOSITION_BYTES = 4 * 1024 * 1024
MAX_EXPOSITION_LINES = 50_000
_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_ESCAPES = {"\\": "\\", '"': '"', "n": "\n"}


class ExpositionError(ValueError):
    """The response cannot be read as a bounded Prometheus text exposition."""


@dataclass(frozen=True)
class Series:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label(self, name: str) -> str | None:
        return dict(self.labels).get(name)


@dataclass(frozen=True)
class Exposition:
    series: tuple[Series, ...]
    unparsed_lines: int
    # Metric names whose line was readable up to the name but not beyond: such a series is
    # malformed, which differs from a series the service never exported.
    unparsed_names: frozenset[str] = frozenset()

    def named(self, name: str) -> list[Series]:
        return [entry for entry in self.series if entry.name == name]


def _labels(text: str, start: int) -> tuple[tuple[tuple[str, str], ...], int]:
    """Parse `{name="value",...}` beginning at `start` (just after `{`); return the end index."""
    labels: list[tuple[str, str]] = []
    position = start
    while True:
        while position < len(text) and text[position] in " \t":
            position += 1
        if position < len(text) and text[position] == "}":
            return tuple(labels), position + 1
        match = _LABEL_NAME.match(text, position)
        if match is None:
            raise ValueError("bad label name")
        name = match.group()
        position = match.end()
        if text[position : position + 2] != '="':
            raise ValueError("label without a quoted value")
        position += 2
        value: list[str] = []
        while True:
            if position >= len(text):
                raise ValueError("unterminated label value")
            character = text[position]
            if character == "\\":
                escaped = _ESCAPES.get(text[position + 1 : position + 2])
                if escaped is None:
                    raise ValueError("bad escape in label value")
                value.append(escaped)
                position += 2
            elif character == '"':
                position += 1
                break
            else:
                value.append(character)
                position += 1
        labels.append((name, "".join(value)))
        if position < len(text) and text[position] == ",":
            position += 1


def _sample(line: str) -> Series:
    match = _NAME.match(line)
    if match is None:
        raise ValueError("bad metric name")
    labels: tuple[tuple[str, str], ...] = ()
    position = match.end()
    if position < len(line) and line[position] == "{":
        labels, position = _labels(line, position + 1)
    fields = line[position:].split()
    # A value, optionally followed by a timestamp the probes do not use.
    if not 1 <= len(fields) <= 2:
        raise ValueError("expected a value and an optional timestamp")
    return Series(match.group(), labels, float(fields[0]))


def parse_exposition(text: str) -> Exposition:
    lines = text.splitlines()
    if len(lines) > MAX_EXPOSITION_LINES:
        raise ExpositionError(f"exposition has more than {MAX_EXPOSITION_LINES} lines")
    series: list[Series] = []
    unparsed = 0
    names: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            series.append(_sample(line))
        except ValueError:
            unparsed += 1
            name = _NAME.match(line)
            if name is not None:
                names.add(name.group())
    return Exposition(tuple(series), unparsed, frozenset(names))


def fetch_exposition(url: str, timeout_seconds: float) -> Exposition:
    """GET a /metrics URL within the timeout. Transport errors propagate to the caller."""
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise ExpositionError(f"metrics endpoint answered HTTP {response.status}")
        body = response.read(MAX_EXPOSITION_BYTES + 1)
    if len(body) > MAX_EXPOSITION_BYTES:
        raise ExpositionError(f"exposition is larger than {MAX_EXPOSITION_BYTES} bytes")
    try:
        return parse_exposition(body.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ExpositionError(f"exposition is not UTF-8: {error}") from error
