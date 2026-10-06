<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

# Web code and UI rules

These rules supplement the repository-root `AGENTS.md`.

## Structure and data flow

- Keep App Router pages and API handlers focused on routing and orchestration.
  Put reusable views in `components/`, reusable UI primitives in the existing
  `components/ui/` or `packages/ui/` system, and data/schema utilities in `lib/`.
  Extend an existing abstraction before creating a parallel component library.
- Keep server-only data access, environment variables, and credentials on the
  server. Add `"use client"` only where interactivity requires it; keep client
  boundaries small and pass serializable, validated data across them.
- Reuse the existing schemas and data adapters. Keep calculations out of repeated
  JSX expressions and preserve measurement meanings, units, and precision.
- Keep state minimal and scoped to its owner. Derive values rather than storing
  duplicate state. Clean up effects, timers, and subscriptions; prevent stale
  requests from replacing data for a newly selected run. Avoid unnecessary polling
  and duplicate requests.

## Responsive layouts and accessibility

- Build layouts from narrow screens upward using the existing Tailwind styles
  and design tokens. Prefer flexible grids, wrapping, and content-driven sizing
  over fixed page widths or absolute positioning for primary layout.
- Verify affected flows at approximately 360–390 px mobile, 768 px tablet,
  1280–1440 px desktop, and 200% browser zoom. Check intermediate widths when
  layouts change. Record which viewports were actually checked in the PR.
- Avoid page-level horizontal overflow. Wide evidence tables and trace timelines
  may scroll inside labeled containers; keep navigation and essential actions
  reachable. Wrap long IDs and messages or provide an accessible way to view/copy
  their full values. Do not hide required information just to fit mobile.
- Make charts resize with their containers and keep legends, units, and tooltips
  readable. Provide accessible textual summaries or tables for essential chart
  evidence; hover alone must not be the only way to inspect measurements.
- Use semantic headings, landmarks, buttons, links, and table headers. Label
  inputs and icon-only actions. Prefer existing accessible primitives for dialogs,
  menus, tabs, and tooltips; preserve keyboard operation and focus management.
- Keep focus visible, touch targets usable (aim for at least 44 by 44 CSS pixels),
  and text contrast readable. Do not communicate status through color alone.
  Respect reduced-motion preferences and avoid unnecessary animation.

## Complete interaction states

- Implement loading, empty, error, success, and unavailable-data states for
  affected flows. Distinguish measured zero from missing evidence. Preserve the
  selected run and filters across refreshes when still valid.
- Prevent duplicate submissions while an action is pending, expose progress and
  failures, and allow recovery where practical. Confirm destructive actions;
  keep read-only navigation immediate.
- Use stable IDs as list keys. Avoid rendering or downloading all history to
  display one page; preserve pagination and limit expensive chart/table work.
- Verify changed interactions in a browser, including keyboard navigation,
  responsive navigation, dialogs, filtering, run selection, and downloads as
  applicable. Unit tests alone do not verify layout or usability. If browser
  verification is unavailable, report that limitation before pushing.
