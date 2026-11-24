# AgentWatch optimization contract

Status: proposed `opt-contract-v2`, pending acceptance by all three teams. Thresholds live in
`pipelines/model/optimization-contract.toml`. O02 stores and serves versioned manual policy
([concurrency policy](concurrency-policy.md)), and agent runner processes enforce it for their own
work ([ADR 0010](adr/0010-runner-process-admission-limits.md), O03). No optimization has been
evaluated; O06 runs the evaluation.

This contract fixes the policy semantics, preregistration and numerical gates for manual
concurrency limits before any evaluation, so no gate is tuned to results. It turns the
proposals in [the optimization plan](optimization-plan.md) into a versioned contract. Each
value's basis is **Plan** (from the plan), **Plan, metric chosen here** (the plan sets an aim
and this contract picks the exact rule), or **Proposed** (no supporting data; must be accepted
or changed before first use).

## Policy semantics

| Rule | Basis |
|---|---|
| Two independent positive-integer limits: model (inference) requests and test jobs, for the designated workload only | Plan |
| Enforcement is per runner process; no limit is claimed across runners | Plan |
| Changing a limit affects future admissions only; lowering it never cancels in-flight work | Plan |
| Each admission queue has a maximum length and a maximum wait, declared in the experiment manifest before the run | Plan, metric chosen here |
| Permits are released on every terminal path: completion, failure, timeout and cancellation | Plan |
| The baseline is no admission limit, as the runner behaves today; the manifest records the baseline limits used, and disable restores them | Plan, metric chosen here |
| A policy version encodes "no limit" for a control by omitting that control's limit; a limit that is present must be an integer of at least 1, and zero, negative, null or sentinel values are rejected | Proposed |

Policy versions are immutable. Each records the effective limits, the operator and reason,
resource and service identity, and run/step correlation. Admission wait, inference-service
queue, execution wall time, process CPU time and predicted excess latency stay separate.

## Preregistration

Before evaluation, the experiment manifest records the workload, task and model versions,
hardware and power conditions, the arrival schedule, test and retrieval state, the contended
condition's injector set and intensity, candidate limits, repetitions, warm-up, randomization
seed, queue bounds and the frozen evaluation tasks. Anything not preregistered is not gated.

| Requirement | Value | Basis |
|---|---|---|
| Unit | A batch: tasks submitted together to one runner; batch time runs from first submission to last terminal task | Plan, metric chosen here |
| Conditions | Clean, and contended with a preregistered injector set and intensity | Plan |
| Candidate limits | 1, 2 and 4, against the baseline; a candidate above measured capacity is not run | Plan |
| Controls changed per candidate | One; factorial runs that change both are reported as not evaluated in this version | Plan, metric chosen here |
| Pairs | One baseline and one candidate batch on the same tasks, inputs, hardware and power state, back to back, order randomized with a recorded seed | Plan, metric chosen here |
| Repetitions | At least 5 pairs and 20 completed tasks per configuration | Plan |
| Warm-up and cold requests | Declared warm-up; cold requests reported separately and excluded from warm comparisons | Plan |
| Tasks | Settings chosen on development tasks; frozen evaluation tasks are disjoint and used once per experiment: never to choose settings, and an evaluation is not rerun to replace its result. A later preregistered experiment may freeze the same tasks only to replicate one with unchanged candidates (on another host, for example), and its report says they were evaluated before. Candidates chosen or changed after a failed or inconclusive evaluation need new frozen tasks, so held-out tasks never select settings | Plan, metric chosen here |
| Minimum frozen evaluation tasks | 4: every batch runs every frozen task, so 4 tasks over the minimum 5 pairs give the 20 completed tasks per configuration; the agent's six tasks leave two for development | Plan, metric chosen here |

## Gates

| Gate | Threshold | Conditions | Basis |
|---|---|---|---|
| Improvement | Median batch completion time at least 10% lower than baseline | Contended | Plan |
| Improvement uncertainty | The 95% percentile paired bootstrap interval (10,000 resamples over pairs, recorded seed) for the median relative reduction excludes zero; otherwise the gate is inconclusive | Contended | Proposed |
| Throughput | Completed tasks per second at least 95% of baseline | Clean and contended | Plan |
| Tail latency | p95 task latency at most 105% of baseline | Clean and contended | Plan |
| Failures and timeouts | No new failures or timeouts | Clean and contended | Plan |
| Quality | No frozen task whose acceptance tests pass at baseline fails with the candidate | Clean and contended | Plan, metric chosen here |
| Starvation and leaks | No admission waits longer than the declared maximum; no permit still held after its batch ends | Clean and contended | Plan, metric chosen here |
| Admission overhead | Median wall-time increase at most 5% with unchanged limits, measured separately from instrumentation | Clean | Plan |
| Instrumentation overhead | Median wall-time increase at most 5%, shared with [the evaluation contract](evaluation-contract.md) | Clean | Proposed, then checked on hardware ([below](#overhead-evidence)) |
| Baseline stability (`max_baseline_root_step_cv`) | Coefficient of variation of the root step's total duration across baseline batches at most 15%, the existing gate in `services/agent`; otherwise every comparison on that baseline is inconclusive | Clean and contended | Plan |

With 5 pairs, the bootstrap interval excludes zero only when every pair shows a reduction, so at
the minimum the uncertainty gate is in effect a sign test on the pairs: the 10,000 resamples make
the interval reproducible, not more precise, and a small or noisy sample stays inconclusive. No
multiplicity correction is applied across candidates in v1 (see below). Reports include paired effect estimates and intervals
for median batch time, median and p95 task latency, throughput, admission wait, service queue
when observable, failures, timeouts and variability. Passing does not establish a general
causal improvement beyond the declared workload and hardware.

### Overhead evidence

The 5% instrumentation budget was proposed without data and has since been checked on one host,
an RTX 5070 laptop under WSL2 (A07, [project checklist](project-checklist.md)): GPU vLLM
`within_budget` and GPU Ollama `inconclusive` (baseline variability). CPU-container attempts
are still open, so the value rests on one host and one shared GPU. The O06 admission-overhead
measurement on the same host was `within_budget` against the 5% admission budget.

**Shared gate.** The instrumentation budget is the same value as the evaluation contract's
`max_median_wall_time_increase`, and the contract test requires the two to be equal. Accepting
this contract does not accept the evaluation contract, which has its own acceptance; changing
either value needs a new version of both contracts, so they cannot diverge.

**Recorded deviations.** An overhead verdict whose only recorded protocol deviation is
operating-system display processes sharing the measured GPU, on a host where they cannot be
removed, may support a promotion only when the evaluation ran on the same host under the same
condition; the report lists it as a limitation. A verdict with any other deviation, such as
another workload on the measured GPU, never supports a promotion. This rule was fixed before any
evaluation result existed.

**One value, two uses.** `max_baseline_root_step_cv` (15%) gates the root step's duration
across baseline batches here, and the [instrumentation overhead
protocol](instrumentation-overhead-protocol.md) borrows the same value for per-repetition
uninstrumented wall time. Accepting this contract fixes both uses together, so a later change
to the value moves both.

## Outcomes and no-action conditions

Each gate is pass, fail, inconclusive or not evaluated. Each candidate is gated independently
against the baseline. A candidate is promoted only when every gate passes, and at most one is
promoted: the passing candidate with the lowest median batch time, or the smaller limit on a
tie. Gating three candidates at 95% each raises the chance that one passes by luck; reports
state how many candidates were tried. Otherwise the baseline limits stay, and every attempt is
preserved. The baseline also stays when:

- evidence is missing, stale, unpaired, or mixes hardware, model, task or contract versions;
- fewer than the minimum pairs or completed tasks were collected;
- the baseline fails the stability gate;
- a candidate exceeded measured capacity or the declared queue bounds.

## Handoffs

| Consumer | Uses |
|---|---|
| Policy controls and runner admission (Allen, Akshayaa) | Policy semantics, enforcement scope, queue bounds, permit release, baseline and disable/restore |
| Experiment evidence (Akshay) | Manifest fields above, pair identity, contract version and hash, rejection of incompatible comparisons, preservation of every attempt |
| Hardware evaluation (Benoy) | Preregistration, pairs, repetitions, conditions and every gate |
| Controls UI (Meghana) | Positive-integer validation, rejected limits and policy conflicts shown as errors, disable/restore, and missing or stale evidence shown as such |

## Acceptance

`status` moves to `accepted` only when every team has accepted this version. Teams accept on the
pull request; once all three have, one commit fills every row below and sets `status` to
`accepted` together. The contract test enforces that: it fails if `status` is `accepted` while
any row lacks a name and an ISO date, or if a row is filled while it is still `proposed`.
Repository merge still requires owner approval as usual.

Acceptance of `opt-contract-v2` was requested on 2026-11-24 in #109.

| Team | Members | Accepted by | Date |
|---|---|---|---|
| Agent & experiments | Benoy, Akshayaa | | |
| Frontend & ML | Rohit, Meghana | | |
| Data & backend | Allen, Akshay | | |

## Change policy

Change thresholds only before an evaluation uses them, and bump `version` with every change.
Reports record the version and the first 8 hex digits of the SHA-256 of the parsed TOML as
canonical JSON. Earlier versions and their reports are kept.
`tests/contract/test_optimization_contract.py` validates the TOML's types and ranges and checks
that its thresholds match this page's table rows, the optimization plan, the evaluation contract
and the agent's existing stability gate, so a threshold edit cannot drift silently.

| Version | Change |
|---|---|
| `opt-contract-v1` (2026-10-26) | First proposal; never used by an evaluation, so it has no reports. Its file is in git history |
| `opt-contract-v2` (2026-11-20) | Minimum frozen evaluation tasks 2 → 4, replacing v1's placeholder now that the agent has six tasks. No other value changed. The page also states that frozen tasks are used once per experiment, and that only a replication with unchanged candidates may reuse them |

## Known limitations

- The agent has six tasks, so an evaluation that freezes four keeps two for development.
  Adding tasks would let a later version freeze more.
- "Used once" is per experiment, so a replication with unchanged candidates (a second host)
  reuses the same four frozen tasks. Its results are not independent of the earlier
  evaluation's, and its report must say so. New candidates after a failed or inconclusive
  evaluation need new frozen tasks; with six tasks in all, more tasks must be written first.
- Earlier use is not recorded in the manifest (no field or registry exists), so the rule is
  enforced by review and by the report's statement, not by the manifest check.
- Only one GPU laptop is confirmed; its results apply to that model, GPU and WSL setup.
- Per-process enforcement does not limit concurrency across several runners.
