# AgentWatch optimization contract

Status: proposed `opt-contract-v1`, pending acceptance by all three teams. Thresholds live in
`pipelines/model/optimization-contract.toml`. No concurrency control exists and no optimization
has been evaluated; O02 and O03 implement the controls, and O06 runs the evaluation.

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
| Controls changed per candidate | One; factorial runs that change both are reported as not evaluated in v1 | Plan, metric chosen here |
| Pairs | One baseline and one candidate batch on the same tasks, inputs, hardware and power state, back to back, order randomized with a recorded seed | Plan, metric chosen here |
| Repetitions | At least 5 pairs and 20 completed tasks per configuration | Plan |
| Warm-up and cold requests | Declared warm-up; cold requests reported separately and excluded from warm comparisons | Plan |
| Tasks | Settings chosen on development tasks; frozen evaluation tasks are disjoint and used once | Plan |
| Minimum frozen evaluation tasks | 2, a placeholder that must be revisited before first use: with the agent's two fixed tasks, disjoint development and evaluation tasks cannot both exist yet | Proposed |

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
| Instrumentation overhead | Median wall-time increase at most 5%, shared with [the evaluation contract](evaluation-contract.md) | Clean | Proposed |
| Baseline stability (`max_baseline_root_step_cv`) | Coefficient of variation of the root step's total duration across baseline batches at most 15%, the existing gate in `services/agent`; otherwise every comparison on that baseline is inconclusive | Clean and contended | Plan |

With 5 pairs, the bootstrap interval excludes zero only when every pair shows a reduction, so at
the minimum the uncertainty gate is in effect a sign test on the pairs: the 10,000 resamples make
the interval reproducible, not more precise, and a small or noisy sample stays inconclusive. No
multiplicity correction is applied across candidates in v1 (see below). Reports include paired effect estimates and intervals
for median batch time, median and p95 task latency, throughput, admission wait, service queue
when observable, failures, timeouts and variability. Passing does not establish a general
causal improvement beyond the declared workload and hardware.

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

`status` moves to `accepted` only when every team has accepted this version; the contract test
fails if `status` is `accepted` while any row below lacks a name and an ISO date, or if a row is
filled while it is still `proposed`. Repository merge still requires owner approval as usual.

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

## Known limitations

- The agent currently has two fixed tasks, so development and frozen evaluation tasks cannot
  yet be disjoint; until more tasks exist, the gates are not evaluated.
- Only one GPU laptop is confirmed; its results apply to that model, GPU and WSL setup.
- Per-process enforcement does not limit concurrency across several runners.
