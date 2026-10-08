---
name: superpowers
description: Start the repo-local Superpowers development workflow for brainstorming, planning, implementation, debugging, or review when the user asks to use Superpowers.
license: MIT; see ../../licenses/superpowers-MIT.txt
---

# Superpowers

AgentWatch entrypoint, authored 2026-10-08, for the vendored
[Superpowers skill library](https://github.com/obra/superpowers/tree/8ca22dba9a94f28898bbce59f2537ff4d87c747d).

Read [using-superpowers](../using-superpowers/SKILL.md), then select the relevant
repo-local skill. Use its actual frontmatter name without the `superpowers:`
plugin namespace. Supporting references and scripts live alongside each skill.

Typical entrypoints:

- New feature: [brainstorming](../brainstorming/SKILL.md), then
  [writing-plans](../writing-plans/SKILL.md).
- Bug or failing check: [systematic-debugging](../systematic-debugging/SKILL.md).
- Approved plan: [executing-plans](../executing-plans/SKILL.md), or
  [subagent-driven-development](../subagent-driven-development/SKILL.md) when
  delegation is authorized and available.
- Before completion: [verification-before-completion](../verification-before-completion/SKILL.md).

Follow `AGENTS.md` and explicit user instructions over this workflow. The library
does not authorize destructive cleanup, external messages, merges, or permission
changes. Without the plugin's startup hook, invoke `/superpowers` to enter this
workflow explicitly. Never claim a dispatched agent or check ran if it did not.
