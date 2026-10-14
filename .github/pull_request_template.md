## Change and purpose

<!-- Explain the problem and resulting behavior. Link the task/issue if available. -->

## Scope and dependencies

<!-- Link the task and related open PRs. Explain why this is a separate task if
related work is already open, and list affected components and merge order. -->

## Verification

<!-- List commands actually run and their results. State skipped, unavailable,
or failing checks explicitly. For behavior changes, include reproduction steps. -->

| Check or manual scenario | Result / evidence |
| --- | --- |
| | |

## UI evidence (if applicable)

<!-- Include screenshots and widths actually checked: mobile 360–390 px,
tablet 768 px, desktop 1280–1440 px, and 200% zoom. Describe keyboard checks,
loading/empty/error states, and any overflow. Write N/A for non-UI changes. -->

## Contracts, data, and rollout (if applicable)

<!-- Describe API/SDK changes, migrations, dependencies, configuration, and
rollback steps. For profiling changes, explain effects on measurement meanings,
coverage, reproducibility, and overhead. Write N/A when there is no impact. -->

## Review checklist

- [ ] This is the task's one active PR, or the independent scope/split is explained above.
- [ ] Related audit findings and review fixes are included; the planned audit is complete.
- [ ] Changes are focused; unrelated work and secrets are excluded.
- [ ] PR title describes the resulting behavior using `<type>: <specific change>`.
- [ ] At merge: squash only, with the final PR title and a concise description;
      no work-in-progress commit log.
- [ ] Relevant local checks passed; limitations are stated above.
- [ ] Required GitHub CI checks passed on the final version.
- [ ] Owner `all3n2601` and another contributor approved the final code changes
      (owner-authored PRs require two other reviewers or an explicit admin bypass).
- [ ] Review conversations are resolved and affected documentation is updated.

