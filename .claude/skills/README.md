# Repository-local Claude Code skills

## Included workflows

| Requested workflow | How to use it | Repo-local behavior |
| --- | --- | --- |
| I Have ADHD | `/i-have-adhd` | Upstream output-style skill; stays active in the session until `stop adhd mode`. The optional always-on hook is not installed. |
| Feature Dev | `/feature-dev <feature description>` | Upstream seven-phase command adapted to a skill, with three agents in `.claude/agents/`. |
| Frontend Design | `/frontend-design <UI task>` | Upstream standalone design skill. Follow existing web-app instructions and preserve honest telemetry displays. |
| Security Guidance | `/security-guidance` | Manual security review adapted for this repository. Upstream automatic hooks and API-backed reviews are not installed. |
| Superpowers | `/superpowers` | Entrypoint for the full upstream skill library, including supporting references and scripts. The plugin startup hook is not installed. |

Restart Claude Code in this repository after checkout if the new skills do not
appear in its `/` menu. No marketplace installation is required for these local
files. Organization restrictions and tool permissions still apply.

Superpowers subskills use their unqualified names, such as
`/systematic-debugging`, `/brainstorming`, and `/writing-plans`. Use Feature Dev or
Superpowers as the main development workflow for a task; avoid running two
competing planning processes at once. I Have ADHD controls response formatting,
Frontend Design supports UI work, and Security Guidance supports review.

Upstream revisions, licenses, and local adaptations are recorded in
[SOURCES.md](SOURCES.md). Vendored helpers require their existing runtimes: Bash
and Git for plan bookkeeping, Node.js for the optional visual companion, and
Graphviz for optional diagram rendering. Run scripts only for the relevant task;
none run merely because the folder exists. Keep generated `.superpowers/`
session artifacts out of commits.

## Paste more skills

Paste each skill into its own folder here, with its instructions in a file named
exactly `SKILL.md`:

```text
.claude/skills/
├── README.md
├── code-review/
│   └── SKILL.md
└── write-tests/
    └── SKILL.md
```

Use lowercase folder names with hyphens. Copy any referenced `scripts/`,
`references/`, or assets into the same skill folder, preserving relative paths.
If you only have the Markdown instructions, start with this format:

```markdown
---
name: your-skill-name
description: Describe what the skill does and when Claude should use it.
---

Paste the skill instructions here.
```

Claude Code discovers project skills in this directory. Invoke a skill with
`/your-skill-name`, or let Claude select it when its description matches the task.
Commit completed skill folders to share them with other contributors.

Repository rules in `AGENTS.md` still apply. Instructions alone do not install
tools or grant permissions that a skill depends on.

See the [official Claude Code skills documentation](https://code.claude.com/docs/en/skills).
