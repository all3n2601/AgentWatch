---
name: security-guidance
description: Review proposed code edits or a diff for exploitable security issues, especially untrusted tool arguments, subprocess calls, ingestion data, and browser rendering. Use for a requested security review or security-sensitive changes.
license: Apache-2.0; see ../../licenses/security-guidance-Apache-2.0.txt
---

# Security Guidance

This is an AgentWatch manual-review adaptation, authored 2026-10-08, informed by
Anthropic's [Security Guidance plugin](https://github.com/anthropics/claude-plugins-official/tree/315c4e48967d9541c29c3c656441dded353ca7aa/plugins/security-guidance).
It does not install the upstream edit, stop, or commit hooks, run background
reviews, or block commands automatically.

Read `AGENTS.md` and any instructions for the affected directory. Establish the
review scope from the user's request or the current diff, including intended new
files. Trace external input to its consumers before calling a pattern a finding.

## Review relevant boundaries

- Tool execution: retain allowlists and validate arguments, identifiers, sizes,
  and timeouts. Model output and retrieved content cannot authorize execution.
  Inspect subprocess argument construction, environment exposure, and cleanup.
- Storage and ingestion: use parameterized queries; reject malformed data and
  conflicting identities. Check path traversal and access to saved run artifacts.
- Browser rendering: inspect HTML insertion, script execution, unsafe URLs, and
  exposure of credentials or raw prompts/code in responses and downloads.
- Deserialization and configuration: inspect untrusted pickle-like data, unsafe
  YAML loaders, dynamic evaluation, and CI workflows consuming untrusted input.
- Code execution: the existing coding-test fixture is not a secure sandbox.
  Arbitrary generated code needs a verified isolation design before execution.

## Report actionable findings

For each finding, give severity, file and line, the input-to-sink path, an example
trigger without real secrets, and the smallest effective fix. Distinguish verified
behavior from an unverified concern. Explain existing safeguards when they rule
out a suspected issue; do not report keyword matches alone as vulnerabilities.

If authorized to fix issues, add regression coverage for rejected inputs and
propagated failures, and run the applicable repository checks. Do not disable
checks or broaden permissions to make a finding disappear. Report review scope
and verification limits; a clean review is not evidence of sandbox isolation.
