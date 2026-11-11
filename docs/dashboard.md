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
| `/compare?run=UUID&baseline=UUID` | Exactly the baseline and candidate UUIDs in the URL, their recorded settings, measurements, quality, coverage and limitations; differences only when every setting is recorded and equal |
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
- Comparison: candidate minus baseline, for the two saved runs named by `?baseline=` and
  `?run=` only. A UUID that cannot be loaded is reported and never replaced by another run,
  and no run is preselected. Kind, workload, checksum, iterations, competing iterations,
  worker capacity, tests per step, baseline repeats, agent model and concurrency policy are
  listed as same, differs or unknown. Differences and relative changes appear only when all
  of them are recorded and equal; otherwise the comparison is blocked and each run's values
  are shown side by side. Saved runs record no policy version, so policy is unknown and
  every saved-run comparison is currently blocked. Throughput and p95 task latency are not
  recorded for single saved runs. Matching settings show observed differences on two runs,
  not a cause.
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
and comparability of saved runs (`tests/run-comparison.test.ts`). Browser checks exercise
page navigation, run filtering, span inspection, recorded samples, preference persistence,
mobile layout, and the selected UUIDs, missing runs, blocked comparisons and keyboard
selection on Compare (`e2e/compare-runs.spec.ts`).
