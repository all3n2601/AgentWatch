# AgentWatch workspace dashboard

The dashboard uses Tailwind, shadcn components, and Recharts. Every navigation item
opens a distinct route; run selections and comparisons are represented in the URL.
All displayed measurements come from completed recorded experiments.

## Pages

| Route | What works |
|---|---|
| `/` or `/overview` | Workspace run/step/request totals, actual execution/queue chart, signal coverage, recent activity |
| `/runs` | Run/model search, workload/status/time filters, source evidence export, older history loading when available |
| `/traces?run=UUID` | Searchable run navigator, timestamp-aligned worker waterfall, span search, contention filter, evidence drawer |
| `/agents?run=UUID` | Agent-session totals, model/tool timeline, request/server timings, token counts, exact conversation/tool feedback |
| `/resources?run=UUID` | Fetches the run's raw sampler artifact; sample coverage, PSI chart when present, evaluation coverage, acceptance checks, exports |
| `/compare?run=UUID&baseline=UUID` | Baseline/candidate selection, matching-workload check, measured differences and percent changes |
| `/settings` | Persistent table density and automatic completed-history refresh preferences, observed source/connection details |

Use the run navigator rather than an unlabeled run selector. Selected rows show the
model or workload, run ID, queue total, date, and verification state. Direct links
load a requested saved run even when it is outside the first history page. Missing
or invalid requested IDs produce a visible warning. Unknown page routes return 404.

## Measurement definitions

The overview summarizes the **loaded history**, filtered by recorded completion time.
“All recorded” includes all loaded runs; “Last 24h” and “Last 7 days” restrict that set.
If the API returns more history, use the Runs page's older-history control.

- Profiled tool steps: all baseline and evaluation steps in the loaded experiments.
- Worker waiting/execution: sums of directly measured queue/execution timings for those steps.
- Model requests and inference wall time: preserved agent requests; tracked separately
  from worker timings so inference is not counted as tool execution.
- Token totals: server-reported counts; missing values remain unavailable rather than zero.
- Comparison: candidate minus baseline. Matching output, workload iterations, and worker
  capacity are checked. A mismatch is visibly flagged and does not establish a regression.
- PSI: raw Linux CPU pressure avg10 readings, not queue fractions. Unsupported local Mac
  readings are shown as unavailable, while recorded sample and valid-reading counts are shown.

GPU and retrieval are explicitly marked uninstrumented. No synthetic saturation graphs,
learned-model confidence, team accounts, or user identities are presented.

## Settings

Density changes the run/span table padding. Refresh preferences poll completed history
manually or every 15, 30, or 60 seconds; they do not stream unfinished runs. Both settings
persist in this browser and apply across pages. Reset returns to comfortable/manual.
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
and matching comparison workloads. Browser checks exercise page navigation, run filtering,
span inspection, recorded samples, preference persistence, and mobile layout.
