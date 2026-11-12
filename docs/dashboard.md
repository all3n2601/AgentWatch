# AgentWatch workspace dashboard

The dashboard uses Tailwind, shadcn components, and Recharts. Every navigation item
opens a distinct route; run selections and comparisons are represented in the URL.
All displayed measurements come from completed recorded experiments.

## Pages

The sidebar groups pages by job: **Investigate** (Overview, Runs, Experiments, Compare),
**Trust** (Data quality) and **Operate** (Controls, Settings). Run selections and comparisons
are represented in the URL. See [the UI redesign plan](ui-redesign-plan.md) for the reasoning.

| Route | What works |
|---|---|
| `/` (`/overview` redirects here) | On screens at least 1280×760, one screen with no page scroll, as in the approved design (title and callouts over the scene, evidence panel overlapping its foot); stacked below that. The glass-shell Overview for one agent model's runs in the newest dataset release (`docs/design-system.md`): an interactive 3D campus (three.js) whose building halos and moving traffic show each stage's measured excess-wait share and the failed-run share, with hover/click stage cards and a static fallback; callouts for run outcomes, the largest observed slowdown (median over experiments of injected-replay time over the same experiment's clean replays) and steps with host telemetry; call types, steps labelled with excess wait by resource, and a run-outcome ring; and the investigation assistant chat in the right rail, which opens with three suggested questions centred in an otherwise empty conversation. Computed on the server by `lib/model-overview.ts`. Defaults to the model with the most runs; `?model=<id>` selects another when the release has several. Runs that record no model (coding and CPU experiments) are excluded and counted in a note. A failed run opens its record only when `.data/live-import-map.json` (or `AGENTWATCH_IMPORT_MAP`) maps it to a live run; otherwise it is marked "Not imported". The earlier command-center panels (replay dot plot, slowest runs, per-task slowdown) are not on this page |
| `/runs` | One list of live and saved runs (one page of newest live runs, loaded saved runs): search, source, state group and kind filters, paging, "Older live runs" and "Older saved runs". Live rows show "—" for steps and waiting because the live list API does not carry them |
| `/runs/UUID` | One run's page, whichever store holds it (the live API is asked first, then saved history; an unknown UUID is 404). Tabs: **Diagnosis** (steps ranked by queue delay, the resource involved and what that rests on), **Summary and scope**, **Timeline** (every step on one time axis beside the selected step's readings), **Resources** and **Evidence**. Saved runs add **Agent** (agent runs); their Resources tab charts CPU pressure from `samples.jsonl` with the clean and contended windows marked, and Evidence lists the three artifacts, the acceptance checks and the attribution settings. Live runs add **Live status** (connection, run state, feed freshness, totals, and the concurrency policy: versions linked while open, limits the runner applied (not recorded), current policy); all four live tabs share one poll of the live API, and the live timeline uses the producer's wall clock (approximate) while step readings use the runner's monotonic ticks. `?tab=` selects a tab |
| `/experiments` | Replay experiments from the dataset release: median step time per condition against the same experiment's clean replays, and the median slowdown per condition across experiments (ratios, observed, not causal) |
| `/compare?run=UUID&baseline=UUID` | Exactly the two saved runs named by UUID (a missing or invalid UUID is reported, never replaced); a baseline and candidate picker and swap; tabs **Summary** (where time changed, interpretation, quality and coverage, largest step differences, evidence exports), **Configuration** and **Steps**. One compatibility check, shared with the Runs selection (`lib/run-compatibility.ts`), lists every setting as match, mismatch or unknown, including competing iterations (contention) and the concurrency policy. A mismatch suppresses every delta, ratio and headline and lists the differing settings; when every recorded setting matches but the policy is not recorded (true of every saved run today), neutral observed differences are shown with "Policy not recorded; full compatibility cannot be verified", never "Configuration match". No faster/slower claims or better/worse colouring; throughput and p95 task latency are "Not recorded" |
| `/quality` | The dataset release summary: labels, label basis, workload mix, telemetry coverage, `split-v2` composition, held-out checks, single-feature AUC, empty-value shares, quality-gate result. Reads `data/datasets` (or `AGENTWATCH_DATASET_DIR`); shows "Dataset unavailable" with a `make pull-dataset` hint when files are missing, malformed, or disagree with the manifest counts |
| `/controls` | Versioned manual concurrency limits for coding-agent runs (`docs/concurrency-policy.md`): the active version with its read time and Read again, limits applied by runners (not recorded by the API), a form that starts from the active limits and previews the change, disable and restore with confirmation, paginated history; conflicts, rejected and unconfirmed changes as banners. Nothing changes limits automatically |
| `/settings` | Persistent table density, sidebar and automatic completed-history refresh preferences, observed source/connection details |
| `/live?liveRun=UUID` | Live runs (in the sidebar): the newest runs from the live API beside the selected run's Status, Timeline, Resources and Evidence tabs (`?tab=`). Follows the newest run when no `liveRun` is given; an unknown or invalid UUID is reported, never replaced by another run |

### Routing

- Each page is its own App Router segment in the `app/(workspace)/` route group, whose layout
  keeps the shell mounted: `page.tsx` (Overview), `runs`, `runs/[runId]`, `experiments`,
  `compare`, `quality`, `controls`, `settings`, `live` and the development-only
  `design-system`. Pages read their data on the server through the loaders in `lib/`
  (`runs-page-load.ts`, `run-load.ts`, `experiments-page-load.ts`, `dataset-load.ts`, and
  `experiments.ts` for saved history and requested runs).
- Links are built from `lib/routes.ts`. `typedRoutes` is on in `next.config.ts`, and
  `pnpm typecheck` runs `next typegen` first, so a link to a page that does not exist fails the
  typecheck.
- Old URLs redirect in `next.config.ts` (307): `/overview` to `/`, `/dataset` to `/quality`, and
  `/traces`, `/agents` and `/resources` to that run's Timeline, Agent or Resources tab when
  `?run=UUID` is given, otherwise to Runs.
- The header search submits to `/runs?q=…`, which opens Runs filtered by that text.
- Unknown routes and runs no store holds render `app/not-found.tsx` (404); a page that fails to
  render shows `app/error.tsx` with a retry. Every route sets its own tab title.

The Diagnosis tab never shows a confidence figure and never presents the resource a live step ran
on as an attribution of waiting: no per-step attribution is stored for live runs, and this
dashboard does not report how accurate the saved-run rule is.

## AI summaries (optional)

### Investigation assistant (Overview)

The Overview's right rail is a chat over AgentWatch's run data (`components/assistant-chat.tsx`,
`POST /api/assistant`). It uses the same endpoint settings as the summaries below and is off
without them. The model reads data only through four read-only tools (`lib/assistant-tools.ts`):

| Tool | Reads |
|---|---|
| `get_overview` | One agent model's population in the dataset release: outcomes, call types, excess-wait labels and per-stage shares, telemetry coverage, largest observed slowdown |
| `search_runs` | Dataset runs (outcome, condition, task, model, slowdown vs the experiment's clean replays) and the ingestion database's live and saved runs |
| `get_run` | One run by UUID or 8-character prefix: dataset totals and labels, plus the stored run's top steps by queue time and by duration (an imported run is labelled as the same run) |
| `compare_conditions` | Median slowdown per injected condition against each experiment's clean replays |

How it stays honest:

- Tool arguments are schema-checked (unknown tools, extra keys and out-of-range values are
  refused) and results carry only timing, outcome and label fields, never prompts or code.
- At most four tool rounds and eight calls per question; each request times out after 60 s.
- Every sentence of the answer must quote only figures (including run ids) found in the tool
  results or the question. Sentences that do not are removed, and the chat says how many.
- Runs the answer names are listed with links to their run pages; a dataset run without an
  imported live run is marked "Not imported" rather than linked elsewhere.
- If the endpoint rejects tool definitions, the chat falls back once to a fixed context (the
  overview and the failed runs).
- The endpoint must answer with one JSON completion; requests send `stream: false` because
  9router streams otherwise.

When `AGENTWATCH_INSIGHTS_URL` and `AGENTWATCH_INSIGHTS_MODEL` are set (plus `AGENTWATCH_INSIGHTS_API_KEY`
if the endpoint needs one), each run's Diagnosis tab shows a **Summarize** button. (The glass Overview does not show it yet: its facts still describe the earlier command-center figures.) Any
OpenAI-compatible `/chat/completions` endpoint works, such as a 9router gateway
(`http://localhost:20128/v1` by default there, with a dashboard-issued key and a model or combo name).
Unset, the feature is absent.

How it stays honest:

- The server rebuilds numbered facts from the same figures the page shows (`lib/insights.ts`); the
  browser only says what to summarize. Only these aggregates leave the machine: no run records,
  conversations, code, or file paths.
- The prompt tells the model to use only the facts, quote figures exactly, cite fact ids, and make
  no causal claim, confidence figure or purchase advice.
- The reply is parsed as JSON and checked: a point is kept only if it cites known facts and every
  figure in it appears in those facts. Statements that fail are left out and counted on screen.
  The reply is shown as plain text and never executed.
- The panel is labelled as generated text, names the model, and lists the figures it was given.
- Requests time out after 60 s, replies over 256 KiB are refused, one request starts per 3 s,
  and identical facts reuse a result for 5 minutes. Errors never show the key, the prompt or the
  endpoint's error body.

## Measurement definitions

The overview summarizes the **loaded history**, filtered by recorded completion time.
“All recorded” includes all loaded runs; “Last 24h” and “Last 7 days” restrict that set.
If the API returns more history, use the Runs page's older-history control.

- Profiled tool steps: all baseline and evaluation steps in the loaded experiments.
- Worker waiting/execution: sums of directly measured queue/execution timings for those steps.
- Model requests and inference wall time: preserved agent requests; tracked separately
  from worker timings so inference is not counted as tool execution.
- Token totals: server-reported counts; missing values remain unavailable rather than zero.
- Comparison: candidate minus baseline, observed and not causal. Every recorded setting
  (workload kind, workload, model, worker capacity, iterations, competing iterations, tests per
  step, baseline repeats, output checksum, attribution rule) and the concurrency policy are
  checked. A mismatch suppresses differences; an unknown setting is never counted as a match.
- PSI: raw Linux CPU pressure avg10 readings, not queue fractions. Unsupported local Mac
  readings are shown as unavailable, while recorded sample and valid-reading counts are shown.

GPU and retrieval are explicitly marked uninstrumented. No synthetic saturation graphs,
learned-model confidence, team accounts, or user identities are presented.

## Settings

Density changes the run/span table padding. Refresh preferences poll completed history
manually or every 15, 30, or 60 seconds; they do not stream unfinished runs. Both settings
persist in this browser and apply across pages. Reset returns to comfortable/manual.
The sidebar's Collapse button (desktop only) shrinks it to an icon rail and is saved with the
same preferences; Reset expands it. The theme control (System, Light, Dark) is stored
separately under `agentwatch.theme` and defaults to the operating-system setting.
Connection details describe the actual history source or local fallback. Check connection
requests current history again and reports failures without discarding loaded evidence.

## Local validation

```sh
pnpm --filter @agentwatch/web lint
pnpm --filter @agentwatch/web typecheck
pnpm --filter @agentwatch/web test
pnpm --filter @agentwatch/web build
```

The schema and aggregation tests cover invalid measurements, missing versus zero values,
tool-call retention, ordered span timestamps, mixed tool/inference totals, recorded-time scopes,
and the shared comparison compatibility check. Browser checks exercise page navigation, run
filtering, span inspection, recorded samples, preference persistence, mobile layout, policy
controls (`e2e/concurrency-controls.spec.ts`), a run's policy (`e2e/run-policy.spec.ts`) and
Compare (`e2e/compare-runs.spec.ts`, against synthetic saved runs from the fixture API).
