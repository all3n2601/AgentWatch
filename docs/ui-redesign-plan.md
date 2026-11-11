# AgentWatch UI redesign plan

Status: proposal, written 2026-11-10. Opinions to argue with, not decisions; the owner and the
affected contributors decide.

Implemented 2026-11-10 in the web app only (see `docs/dashboard.md`): a dense command-center Overview led by time lost against the clean baseline (queue delay is near zero in the runs that slow down most, so it is no longer the headline), the sidebar groups, the
unified Runs list, the Run page with tabs and a Diagnosis tab, Experiments, Data quality, route
redirects and scope labels. Not implemented: everything in section 5 that needs a backend change
(`source_run_id`, a unified run read model with counts, measurement-state fields, an attribution
endpoint, experiments in the database). Until those exist the Runs list reads two APIs and its
live rows cannot show steps or waiting share.

## 1. The verdict

The dashboard is a set of well-built pages without an organizing idea. It is organized around
how the code was written (live runs, saved runs, traces, resources, agents) instead of around
the question the product exists to answer.

The product's own objective (`docs/problem-statement-and-plan.md` section 2) is three questions
for every agent step:

1. How much of this step's time was real work, and how much was waiting?
2. If it waited, which resource was saturated: GPU inference, CPU sandboxes or retrieval?
3. How confident is that answer, and how accurate has the method been?

A platform engineer should be able to get from the home page to an answer to all three, with
the evidence open beside it, in three clicks. Today they cannot, for four reasons:

- A run has two homes. Live runs and saved runs have different lists, different detail pages and
  different capabilities, and Compare, Trace, Agents and Resources only see saved runs.
- Question 2 and question 3 are not answered anywhere. The UI shows queue and execution time
  (question 1) and shows labels in the dataset summary, but no run page says "this step waited
  on X, by this method, with this uncertainty".
- Scope is never stated. A tile can count saved runs while the chart beside it counts dataset
  steps.
- Pages are named after implementation (Resources, Agents, Trace explorer), so a new user cannot
  predict what is behind them.

## 2. Who uses it and for what

| Person | Job | Needs from the UI |
|---|---|---|
| Platform engineer (primary) | Decide whether to add GPU, CPU or retrieval capacity | Which resource waits most, how sure we are, evidence to show others |
| AI engineer | Remove waste from an agent's design | Which steps waste time and why, run-to-run comparison |
| Head of platform | Approve spend | One page: where time goes, trend, how trustworthy the numbers are |
| Team member (us) | Validate the dataset and the method | Labels, splits, rule baseline results, telemetry gaps |

The first three need a diagnosis. The fourth needs an audit. Both read the same objects (run,
step, sample), so they should share pages, not separate "dataset" and "runs" worlds.

## 3. Information architecture

One noun per page, ordered by the diagnosis workflow.

```
Investigate
  Overview        where is time going, and is it getting worse
  Runs            every run, live or saved, one list
    Run page      Diagnosis | Timeline | Agent | Resources | Evidence
  Experiments     baseline versus injected contention, replay by replay
  Compare         two runs or two conditions of one workload
Trust
  Data quality    telemetry coverage, labels, splits, rule-baseline result
Operate
  Controls        concurrency policy
  Settings        display, data source
```

What changes against today:

| Today | Proposed | Why |
|---|---|---|
| Live runs + Run history | Runs (one list, source and status filters) | One noun, one list |
| Trace explorer + Agents + Resources | Tabs of the Run page | They are views of one run, each repeating a run picker |
| Dataset summary under Overview | Data quality | It answers "can I trust this", question 3 |
| (missing) | Experiments | The replay protocol's baseline-versus-contention structure is the evidence behind every label |
| Compare runs | Compare | Reached from a selection in Runs, not from a blank page |

## 4. How the main pages should look

Design language, applied everywhere:

- **One colour per resource class**, reused in every chart, chip and legend: CPU sandbox amber,
  inference violet, retrieval sky, mixed rose, no excess slate. Queue and execution keep their
  own colours, distinct from resource classes, because queue delay is not the same claim as
  resource attribution.
- **Every measurement carries a state:** measured, estimated, unavailable, not instrumented. An
  unavailable reading renders as a dash with its state, never as zero.
- **Every number names its scope** ("56 saved runs", "598 replay runs, release steps-v1.3").
- **Charts answer one of the three questions.** A chart that answers none is cut.
- **A detail drawer for evidence** on every table row, with the raw record one click away.

### Overview: the diagnosis summary

```
+--------------------------------------------------------------------------+
| Overview                      [scope: all runs v] [range v]     refresh  |
+--------------------------------------------------------------------------+
| Time waiting      Runs diagnosed     Telemetry coverage   Method           |
| 38% of step time  412 of 654         13% of steps         rules-v1         |
| queue / exec      with a resource    have host samples    not yet evaluated|
+----------------------------------+---------------------------------------+
| Where waiting comes from         | What changed                          |
| bar per resource class,          | runs whose waiting rose vs the        |
| share of waiting time, with      | previous window, each linked to       |
| "not measured" as its own bar    | its Diagnosis tab                     |
+----------------------------------+---------------------------------------+
| Worst runs (by waiting)          | Experiments in progress / last result |
+----------------------------------+---------------------------------------+
```

The home page is for deciding where to look, so it is a list of things to open, with a few
numbers, not a wall of charts.

### Runs: one list

Columns: run, workload, source (live, saved), state, started, duration, waiting share, top
resource, telemetry quality, checks. Filters: source, state, workload, resource, time. Selecting
two rows enables Compare. 650+ runs means server-side paging and counts; the list must not
load everything.

### Run page, Diagnosis tab: the answer first

```
+--------------------------------------------------------------------------+
| fix-window-count-r1  20409226   completed   source: live   Oct 25 04:14   |
| [Diagnosis] Timeline  Agent  Resources  Evidence                          |
+--------------------------------------------------------------------------+
| Slowest steps, ranked by waiting                                          |
|  model-call-4   712 ms   waited 31%   inference (suggested)   telemetry: ok|
|  tool-2-rejected  1 ms   failed       n/a                                  |
|  ...                                                                       |
+---------------------------------------+----------------------------------+
| Step timeline (queue | execution)     | Evidence for the selected step   |
| shared clock if timestamps exist      | queue, execution, CPU, samples,  |
|                                       | method + version, limits         |
+---------------------------------------+----------------------------------+
```

"Suggested" is deliberate. Until the rule baseline is evaluated (it is not today) and a model
passes its promotion gates, the page shows the method, its version and "not yet evaluated",
never a percentage called confidence. Raw scores are not calibrated confidence.

### Experiments

One row per replay experiment: task, protocol, baseline count and variability (the protocol's
own gate), then one row per condition (CPU, inference, retrieval, mixed) with its observed
difference against baseline and its clean control. It states "observed, not causal" because the
comparison is not randomized. This is the page that makes the labels believable.

### Data quality

Telemetry coverage by source and metric, label counts and basis, split composition, held-out
checks, quality-gate result, and the rule baseline's evaluation status. Today's dataset summary
becomes this page's content.

## 5. How well the data must be defined

The UI can only be as clear as the objects behind it. Presentation can be forgiving, but the
boundaries must be strict. The canonical model I would hold the whole system to:

| Object | Must have | Today |
|---|---|---|
| **Run** | UUID, source (live, saved, imported), task and workload, state, start and end, clock domain, configuration digest, model and version, experiment reference, replay role and condition | Two shapes (live registration, saved result); no experiment reference; no source field |
| **Step** | UUID, run, parent, type, resource class, submitted, started and finished ticks with clock domain, queue, execution, duration, process CPU (nullable) | Complete in the live API; saved runs use a different structure |
| **Sample** | resource, class, service, metric, value or unavailable reason, tick, clock domain | Complete in the live API |
| **Attribution** | step, label, method and version, excess fraction, evidence references, telemetry quality, abstained flag, uncertainty (only when calibrated) | Offline only. No endpoint serves per-step attribution |
| **Experiment** | id, protocol version, plan digest, conditions, replays, baseline variability, acceptance checks | Files in the evidence store, not in the database |
| **Dataset link** | release, label version, split, and the run UUID the row came from | Broken: imported runs have new UUIDs and nothing stores the original |

Gaps I would close before building more screens, most important first:

1. **Run identity.** The 598 imported replay runs have new UUIDs (the live API assigns them),
   so a dataset row cannot be tied to its live run. Add a stored `source_run_id` and
   `source_system`. Without it the Data quality and Run pages cannot link, and every join is a
   guess.
2. **One read model for runs.** Either a database view or an API (`GET /v1/runs?source=...`)
   that returns the same fields for live and saved runs, with total counts and cursors. The UI
   should never branch on where a run came from, except to disable a tab.
3. **Measurement states as data.** `measured | estimated | unavailable | not_instrumented`
   needs to be a field, not an inference from null. The checklist already insists missing is not
   zero; the API should say so too.
4. **An attribution endpoint** returning the label, method, version and telemetry quality per
   step, with "not evaluated" as a first-class answer. This is the only way question 2 and 3
   get answered on the Run page.
5. **Experiments in the database,** or at least a read-only index, so the Experiments page does
   not parse evidence folders in the web server.
6. **Units and clocks written down once.** Seconds (float) for durations, nanosecond ticks for
   clock readings, UTC for wall time, and which clock each field uses. Several are stated in
   separate docs today.
7. **Honest timestamps.** Separate `recorded_at` (when it happened) from `created_at` (when it
   entered this database). Imported runs show import time as creation time today.

How strict is enough: every API response validated against a schema shared with the web app
(the web side already validates dataset files and live responses with zod), contract tests on
both sides, and no UI field that has no definition in `docs/shared-event-contracts.md`.

## 6. Changes plan

Estimates are focused work for one engineer; confirm with whoever owns each area.

| Phase | Work | Size | Depends on |
|---|---|---|---|
| 0 | Fix vocabulary and scope labels on every page; move Controls and Settings into a separate "Operate" group; consistent empty, loading and error states | 1 day | None |
| 1 | Data contracts: `source_run_id` migration, unified run read model with paging and counts, measurement-state field, `recorded_at` versus `created_at`; contract tests | 3 to 4 days | Owner approval of schema changes (migration, API) |
| 2 | Runs list, Run page shell with tabs, Diagnosis tab (ranked steps, evidence drawer), retire the separate Trace, Agents and Resources routes with redirects | 4 to 5 days | Phase 1 read model |
| 3 | Overview rebuilt as the diagnosis summary; Compare reached from Runs selection | 2 to 3 days | Phase 2 |
| 4 | Experiments page and index; Data quality page from the existing summary | 3 to 4 days | Experiments index (phase 1 extension) |
| 5 | Attribution endpoint and confidence display: method, version, calibrated uncertainty, abstention | 3 days plus the model work | PR 61 (calibration, abstention) and PR 62 (mixed scoring) merged, rule baseline evaluated |

Rules for the work, from `AGENTS.md`: schema and API changes update `api/openapi/`, the SDKs, tests
and docs together; migrations are ordered and tested for upgrade; the PRs stay small and in the
order above; nothing in the UI claims a capability the backend does not have yet.

## 7. What I would not do

- **No confidence percentages** until a method is calibrated and has passed its promotion gates.
  Show the method and "not yet evaluated". A wrong confident number is worse than none.
- **No pipeline-map graph yet.** The concept image is good, and live runs do carry parent steps,
  but a graph invites reading causation into a timeline. Build it after the Diagnosis tab proves
  the data.
- **No more charts for their own sake.** The home page should fit on one screen and point at
  things to open.
- **No separate "dataset" destination.** Dataset facts belong in Data quality (audit) and
  Experiments (provenance), where they have a job.
- **No UI built on guessed joins.** Phase 1 comes first even though it is invisible.

## 8. Decisions for the owner and team

1. Is `source_run_id` an acceptable schema addition (migration 00xx), and should it also cover
   saved runs?
2. Should Experiments be a database table, or a read-only index built from the evidence store?
3. Who owns the unified runs API: the worker service (my suggestion) or a new service?
4. Are "Investigate, Trust, Operate" the right sidebar groups for the team and the demo?
5. Does the head-of-platform view need its own page, or is Overview enough?

## 9. How to know it worked

- From Overview, a new user opens the slowest run's Diagnosis tab in two clicks and can state
  which step waited, what the evidence is, and what is not yet known.
- A live run and a saved run open the same page; only unavailable tabs are disabled.
- Every number on every page states its scope and, where relevant, its measurement state.
- Joining a dataset row to its run in the database needs no lookup file.
- The first three questions are each answered by a named element on the Run page, or the page
  says plainly why it cannot yet answer.
