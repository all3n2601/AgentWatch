# AgentWatch glass design system

Status: implemented 2026-11-11 in `apps/web`. On the glass shell: Overview (`/`,
`components/model-overview.tsx`), Runs, Run detail, Experiments, Compare, Data quality and
Settings (`components/pages/<page>/`, with tested logic in `lib/<page>-page*.ts`), and Live runs
(`components/live-feed.tsx`, `live-run.tsx`). Controls has no approved design and keeps its earlier
inner styling inside the glass shell.

The approved direction is the "aggregate glass workspace" and its six supporting screens
(Overview, Runs, Run diagnosis, Experiments, Compare, Data quality, Settings): frosted
panels on a soft green-grey canvas, one brand green, an isometric campus on the Overview,
and solid-versus-hatched bars for work versus waiting.

Review every component at `/design-system` while running `pnpm --filter @agentwatch/web dev`.
The page shows illustrative values only and returns 404 in production builds.

## Tokens

Defined in `apps/web/app/design-system.css`, with light values on `:root` and dark values on
`:root.dark` (the class set by the existing theme script). All are prefixed `--aw-` and mapped
to Tailwind colours:

| Purpose | Tailwind classes |
|---|---|
| Canvas and surfaces | `aw-canvas`, `bg-glass`, `bg-glass-strong`, `bg-glass-inset` |
| Text | `text-ink`, `text-ink-muted`, `text-ink-subtle` |
| Lines | `border-line` |
| Brand | `bg-brand`, `bg-brand-strong` (white text, 5.3:1), `bg-brand-soft`, `text-brand-ink` |
| Tones | `{success,warning,caution,danger,info}` plus `-soft`, `-line`, `-solid` |
| Bars | `bg-bar`, `bg-bar-track`, `hatch` (waiting time) |
| Elevation | `shadow-panel`, `shadow-raised`, `shadow-glow` |

Layout: every workspace page lives in the `app/(workspace)/` route group, whose layout renders
`WorkspaceFrame` (components/shell/app-shell.tsx) once — header and labelled sidebar stay mounted
and keep their position on every page; the sidebar lists all pages (Overview, Runs, Experiments,
Compare, Data quality, Live runs, Controls, Settings). Pages render inside `WorkspacePage` (main
column plus optional assistant rail). The `fit:` variant applies at viewports at least 1280×760
(`@custom-variant fit` in `design-system.css`): the frame fills the viewport and **pages do not
scroll**. Content beyond one screen goes into tabs — `TabbedView` (components/shell/tabbed-view.tsx)
keeps the active tab in `?view=` and keeps inactive panels mounted — and long tables scroll inside
their own labelled panel. Below `fit`, pages stack and the document scrolls.

Settings preferences apply to every workspace page (components/shell/workspace-preferences.tsx):
Sidebar Expanded/Collapsed (labels or icons only), Density Compact (`data-density` on <html>
scales the interface), and Auto refresh (re-renders the page's server data every N s while the
browser tab is visible).

Surface utilities: `glass-panel` (blurred container), `glass-inset` (nested card, no blur),
`glass-raised` (chips and controls). Without `backdrop-filter`, or when the user asks for
reduced transparency, panels become opaque.

## Rules the components enforce

- **Status is never colour alone.** Every tone is paired with an icon or dot and a text label
  (`StatusBadge`, `StatusIcon`, `Checklist`).
- **Missing is not zero.** `Measurement` and `formatMeasurement` (`lib/measurement-format.ts`)
  render a dash plus a state (unavailable, not instrumented) for null or non-finite readings,
  keep a recorded `0` as `0`, and prefix estimates with `~`.
- **Waiting is hatched, work is solid**, in `SegmentBar` and `WorkWaitLegend`, so the split
  survives greyscale and colour-vision differences.
- **Ratios are observations.** `RatioChip` and `DeltaChip` grade emphasis with
  `slowdownTone` (`lib/tone.ts`: above 1.1× warning, 1.5× caution, 8× danger). These are
  display thresholds, not attribution rules; pair them with "Observed, not causal" copy.
- **Invalid proportions are refused.** `layoutSegments` (`lib/segments.ts`) rejects negative,
  non-finite or duplicate segments and reports an all-zero total as empty instead of drawing it.
- **Every page states its scope** through `PageHeader`'s `scope` line.

## Components

Primitives in `components/ui/` (extending the existing shadcn set):

- `Button` gains `brand`, `brand-soft` and `glass` variants and `pill` / `icon-round` sizes.
- `Tabs` gains a `glass` list variant (Diagnosis | Timeline | ...).
- `GlassPanel`, `InsetCard`, `PanelHeader`; `StatusBadge`, `StatusIcon`; `Chip`, `CopyChip`;
  `FilterSelect` (over the existing Select); `SearchField`; `SegmentedControl` (radio group);
  `Switch`; `FieldRow`, `FieldSection`, `GlassInput`.

Widgets in `components/widgets/`:

- `CampusScene` — the Overview render with labels pinned to building anchors.
- `StatCallout`, `StatStrip`; `AlertBanner`; `Measurement`.
- `BarList`, `SegmentBar`, `WorkWaitLegend`, `DonutChart`; `RatioChip`, `DeltaChip`.
- `KeyValueList`, `Checklist`, `RankedList`.
- `TimelineChart` — a Gantt of steps on one time axis (hatched queue, solid execution), with
  selectable numbered rows and a text description per bar. Narrow containers put each bar on its
  own line under the step name. Geometry comes from `lib/run-timeline.ts`.
- `EmptyState`, `UnavailableState`, `ErrorState`, `Skeleton`.

Tabs that lay out a main panel beside an evidence column share `components/shell/panel-layout.ts`
(`TAB_GRID`, `FILL_PANEL`, `PANEL_SCROLL`), so each fills its tab without page scroll.

Shell in `components/shell/`: `AppShell`, `TopBar` (+ `RoundIconButton`), `NavRail`,
`AssistantRail`, `PageHeader`, `BrandMark`. `NavRail` is a client component, so its items take
rendered icon elements (`icon: <House />`), not icon components. Below `xl` the assistant rail
sits above the content as a collapsed bar; below `md` the nav rail is a scrolling bar under the
header.

Not included on purpose: the search microphone and notification bell from the mockups. Neither
has a backend, and the shell does not show controls that do nothing.

## Overview campus scene

The Overview's campus is a real-time three.js scene (`components/widgets/campus-live-scene.tsx`
and `campus-canvas.tsx`, using `three` and `@react-three/fiber`; loaded only on that page).

**Source.** `apps/web/scenes/overview_campus.py` builds the campus in Blender (5.2 LTS) and writes:

- `public/scenes/overview-campus.glb` — geometry (about 1.1 MB, 33k triangles). Each pipeline
  stage is a node with `extras.aw_stage` (`gateway`, `model`, `retrieval`, `cpuTool`,
  `complete`) and its own glow material `aw_glow_<stage>`, so the page can tint stages.
- `lib/overview-campus-3d.json` — camera, stage anchors and road centre lines (glTF Y-up metres),
  validated by `campus3dSchema` in `lib/overview-scene.ts`.
- `public/scenes/overview-campus-{light,dark}.webp` (2880×1560) and
  `lib/overview-scene-anchors.json` — the static render, shown while the 3D scene loads, when
  WebGL is unavailable, or when the GLB fails to load.

**What the colours mean.** Everything is computed from the dataset release by
`lib/campus-status.ts`; nothing is simulated.

| Element | Meaning |
|---|---|
| Building halo and glow | Share of that stage's calls labelled with excess wait (model calls, retrieval calls, tool calls); thresholds 5% / 15% / 35% for attention / degraded / severe. Completion uses the failed-run share. The gateway is informational. |
| Traffic beads | One stream per road. Amber beads are the stage's excess-wait share; on the road to completion, red beads are the failed-run share and green the passed share. A non-zero share always shows at least one bead. |

The thresholds choose emphasis only; they are not attribution rules. The legend states that
traffic shows recorded proportions, not live runs.

**Interaction.** The camera never rotates or pans freely. Hover or focus a stage label (or the
building) for its figures. Clicking a stage eases the camera onto it: the other buildings fold
into the ground, roads and campus-wide overlays step aside, and the building is replaced by its
component model with the stage card docked beside it. "Back to campus", the card's close button,
Escape or a click on empty ground returns. Motion pauses with "Pause traffic", under
`prefers-reduced-motion` (the zoom then jumps instead of easing) and while the scene is off
screen.

**Component models.** `apps/web/scenes/component_models.py` builds one GLB per stage
(`public/scenes/component-<stage>.glb`, 60–160 KB each) and `lib/overview-component-models.json`
(packet routes), validated by `componentModelsSchema`:

| Stage | Model | Motion |
|---|---|---|
| Model runtime | GPU card | three spinning fans, status LEDs |
| CPU tool | CPU board with chip, RAM and tower cooler | cooler fan spins, cores pulse, packets run the board traces from the network card |
| Retrieval | database stack and vector-index rack | rotating scan ring, packets between store and index |
| Agent gateway | network router | port lights blink, signal ring turns, packets in and out |
| Complete | output store | drive lights blink, check mark turns, packets arriving |

Nodes named `spin_*` rotate at `extras.aw_spin` rad/s; meshes named `blink_*` pulse; materials
named `aw_glow_<stage>` take the stage's status colour, exactly as on the campus. The motion is
decorative: only colour carries a measured status.

**Regenerating.** Edit the script, then run it so the GLB, manifest and fallback renders change
together:

```sh
blender --background --python apps/web/scenes/overview_campus.py
blender --background --python apps/web/scenes/component_models.py
```

From an open Blender session (for example through the Blender MCP add-on), import the file with
`importlib.util.spec_from_file_location` and call `main()`. In `next dev`, clear
`apps/web/.next/dev/cache/images` afterwards; the image optimizer otherwise keeps serving the
previous render.
