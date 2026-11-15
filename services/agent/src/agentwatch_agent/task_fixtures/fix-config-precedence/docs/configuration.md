# Configuration

## Sources

A job reads its settings from three sources: built-in defaults, the settings file, and
environment variables.

## Precedence

Environment variables override the settings file, and the settings file overrides the built-in
defaults. A key that no source sets keeps its default.

## Sources stay unchanged

Combining settings never modifies any of the sources.
