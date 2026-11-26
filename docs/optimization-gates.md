# Optimization gates

Status: O06 gate evaluation, implemented 2026-11-22 on branch `feature/optimization-gates`
and verified on synthetic experiments only. `opt-contract-v2` was accepted on 2026-11-25 (#109),
so an evaluation can now promote a candidate; no hardware experiment has been judged yet.
Nothing here is an optimization result.

`make optimization-gates EXPERIMENT=<dir> [OVERHEAD_VERDICT=<A07 overhead-verdict.json>]
[ADMISSION_VERDICT=<admission overhead-verdict.json>]`
recomputes the experiment's [comparison record](optimization-evidence.md) and applies every
gate of [the optimization contract](optimization-contract.md). It writes
`.data/reports/optimization/<first 12 hex of the manifest hash>/gates.json`, next to
`comparison.json`, and exits 1 without writing when the manifest, the contract or the overhead
verdict cannot be used. The module is `agentwatch_data.optimization_gates`.

Thresholds are read from the contract's `[gates]` table of the same file the comparison
loaded, checked by its hash. Nothing is tuned to the results.

## Gates

Each gate is `pass`, `fail`, `inconclusive` (the evidence cannot decide it) or
`not_evaluated` (no measurement exists for it).

| Gate | Conditions | Rule |
|---|---|---|
| `improvement` | Contended | Median batch time reduction at least `min_median_batch_time_reduction` |
| `improvement_uncertainty` | Contended | The paired bootstrap interval lies above zero (excludes it on the side of improvement). An interval that includes zero is inconclusive; one wholly below zero, a confident slowdown, fails |
| `throughput` | Both | Candidate/baseline completed tasks per second at least `min_throughput_ratio` |
| `tail_latency` | Both | Candidate/baseline p95 warm task latency at most `max_p95_task_latency_ratio` |
| `starvation_and_leaks` | Both | No starved admission or leaked permit in candidate batches, and no wait above the declared maximum; unobserved counts are inconclusive |
| `failures_and_timeouts` | Both | No new failure or timeout, counted over every pair with two accepted records |
| `quality` | Both | No task passing at baseline fails with the candidate; unknown acceptance is inconclusive |
| `baseline_stability` | Both | For each frozen task, the coefficient of variation of its root step's total duration over the baseline batches of complete pairs (completed, warm tasks, from each run's own records in its journal, by `run_id`) is at most `max_baseline_root_step_cv`. A missing journal or a task with fewer than two durations is inconclusive |
| `admission_overhead` | Experiment | The [admission overhead](admission-overhead.md) verdict's `admission-overhead-v2` group for this experiment's model (a v1 group judges neither overhead gate), measured on the manifest's host (`host_label` equal to `hardware.host_id`): `within_budget` passes, `over_budget` fails, otherwise inconclusive; not evaluated without a verdict or with no single matching group |
| `instrumentation_overhead` | Experiment | The same, from the A07 overhead verdict's group of any other protocol for this model and host. A verdict file given for the wrong gate, or measured on another host, has no matching group, so it is never used. A verdict judged under another contract (its `contract.sha256`), or against a budget other than this gate's threshold (the group's `budget`, else the report's), is `not_evaluated` too. Each overhead gate records `source`: the verdict file's canonical SHA-256, its contract, and the matched group's settings, task and budget |

## Candidate outcome and promotion

In this order:

1. `fail` when a zero-tolerance gate fails (`failures_and_timeouts`, `quality`,
   `starvation_and_leaks`). One observed occurrence decides these on any sample, so a
   candidate whose failures also leave too few completed tasks is still reported as failing.
2. `inconclusive` when a comparison is insufficient (missing, rejected or unpaired evidence,
   too few pairs or completed tasks, a development experiment), or a baseline is not stable.
   The contract keeps the baseline in either case.
3. `fail` when any other gate fails.
4. `pass` when every gate passes; otherwise `inconclusive`.

At most one candidate is promoted: the passing one with the lowest contended median batch
time, the smaller limit on a tie. Nothing is promoted while the contract is not `accepted`.
The report states how many candidates were tried, because gating several raises the chance
that one passes by luck.

## Known limitations

- Synthetic experiments only (`pipelines/data/tests/test_optimization_gates.py`).
- Without an admission overhead verdict (`make agent-admission-overhead`, then
  `make overhead-verdict`), no candidate can reach `pass`.
- Both overhead gates trust the verdict files they are given; they match a group by protocol,
  model and host, not by sampling or admission settings.
- Root step durations are read from each run's journal (`live-journal.jsonl`, the root step
  `agent-run`), a format the data pipeline declares separately from the agent.
- A pass describes the declared workload, model and host only, not a general causal
  improvement.
