# Repository-local Claude Code skills

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
