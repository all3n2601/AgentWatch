# AgentWatch optimization layer

Status: planned extension, October 13, 2026. No controller, concurrency policy API, or
optimization result is implemented by this plan. O01–O07 are coordination labels used in
task titles; the Team Planner assigns its own card IDs.

AgentWatch will first measure an agent workload, then let an operator limit concurrent
model requests and test jobs, and measure the effect. Reliable telemetry remains the
prerequisite. The first release uses explicit manual limits; a bounded adaptive controller
is a stretch follow-up after controlled evaluation. LightGBM attribution is not an action
policy and is not required to implement manual controls.

## Scope and architecture

```text
Live run/step events + correlated samples -> AgentWatch evidence
Operator-selected versioned policy -> runner admission limits -> model/test execution
Policy changes + admission timings + outcomes -> saved-run evidence -> comparison UI
Accepted evaluation -> optional shadow controller -> opt-in bounded adaptation
```

Use the existing ingestion service, worker/runner responsibilities, shared contracts,
and dashboard. Do not create a second scheduler or put scheduling logic in route handlers
or UI components. Inspect implementation boundaries before selecting the execution hook.
Limit admissions to the designated workload's inference service and test capacity;
do not change unrelated workloads or the Windows Ollama application.

Initial actions are separate positive integer concurrency limits for model requests and
test jobs, plus disable/restore controls. Changing a limit affects future admissions;
lowering it never kills in-flight work. Bound queue length and waiting time, preserve
fairness, propagate failures/cancellation, and release permits on every terminal path.
Define enforcement scope explicitly: a per-process semaphore cannot claim a global limit
across multiple runners. Reserve tests and controller experiments for designated capacity.

Record immutable policy versions, effective limits, operator/decision reason, resource and
service identity, and run/step correlation. Keep admission wait, inference-service queue,
execution wall time, process CPU, and predicted excess latency separate. Use compatible
monotonic clocks for elapsed timings and timezone-aware timestamps for events. Update
applicable API/schema/SDK/consumer definitions together; use ordered transactional migrations
and isolated-schema PostgreSQL integration tests if storage changes.

No automatic model switching, retrieval caching, cloud autoscaling, reinforcement learning,
or new training model is in this first release. Those actions have additional correctness,
cost, and validation requirements and need separate scopes.

## Owners and dependency order

Each row is one primary contributor and a separate task branch/PR. Partner reviewers below
do not replace the repository's required contributor approvals. All tasks begin waiting;
dependency completion includes explicit acceptance, not merely a merged proposal.

| Task | Primary owner | Partner reviewer | Board stage | Prerequisites | Deliverable |
| --- | --- | --- | --- | --- | --- |
| O01 Define optimization contract and gates | Rohit | Meghana | M3 | C01, A04, M01 | Proposed then accepted policy/evaluation contract, numerical gates, workload and telemetry requirements; Allen reviews shared interfaces |
| O02 Add versioned manual concurrency controls | Allen | Akshay | M4 | O01, B01, B02 | Validated policy API/storage and audit evidence; disabled/baseline default and restore path |
| O03 Enforce limits in the agent runner | Akshayaa | Benoy | M4 | O02, A02, A03 | Bounded fair admission for inference and tests, queue timings, cancellation and permit cleanup |
| O04 Version optimization experiment evidence | Akshay | Allen | M4 | O01, D02 | Immutable paired-run manifests, policy/configuration hashes and compatible comparison validation |
| O05 Show optimization controls and comparisons | Meghana | Rohit | M5 | O02, F02 | Explicit manual controls, active policy and saved UUID-specific baseline/treatment comparison |
| O06 Evaluate fixed limits on real hardware | Benoy | Akshayaa | M6 | O03, O04, O05, A05, A07 | Controlled repeated CPU/inference evaluation, quality/throughput/latency/overhead evidence and restore verification |
| O07 Add gated adaptive concurrency controller | Rohit | Meghana | M6 stretch | O06 | Rule-based shadow recommendations, opt-in bounded changes, no-action conditions and rollback; Allen/Akshayaa integrate reviewed interfaces |

O01 can proceed once the telemetry and contracts are accepted; it does not wait for ML
training. O02 and O04 can then run independently. O03 and O05 follow their respective
interfaces. O06 waits for the runner, evidence pipeline, UI, bounded experiments and
instrumentation overhead verification. O07 must not delay the core demonstration.

Suggested branches: `docs/optimization-contract`, `feature/concurrency-policy`,
`feature/runner-concurrency-limits`, `feature/optimization-evidence`,
`feature/optimization-controls`, `test/optimization-evaluation`, and
`feature/adaptive-concurrency`. Coordinate shared sampler, event, and runner edits with
their existing owners; do not duplicate A02/A04/B01/B02 work.

Benoy has the team's only confirmed GPU laptop. Benoy executes GPU/inference experiments;
Akshayaa supports CPU/retrieval and runner tests. Rohit performs analysis and LightGBM
training on CPU. Allen, Akshay, and Meghana use redacted evidence and supported endpoints.
Laptop findings apply to its actual model, GPU and WSL configuration; L4/GCP readiness is
a separate requirement. Unknown or unsupported readings stay distinct from measured zero.

## Acceptance and experiment gates

O01 must accept and freeze a versioned contract before any optimization evaluation. That
contract is [the optimization contract](optimization-contract.md) (`opt-contract-v1`); it is
authoritative once accepted. The following numerical gates are proposals, not measured
performance or accepted team policy:

- Use at least five paired repeated batches and at least 20 completed tasks per configuration
  for tail-latency reporting. Fix inputs, task/model versions, arrivals, test/retrieval state,
  hardware, power conditions and declared warm-up. Alternate or randomize configuration
  order using a recorded seed. Preserve cold requests separately from warm measurements.
- Compare current baseline limits with preregistered fixed candidates, initially 1, 2 and 4
  slots where measured hardware capacity permits. Separate inference and test limits;
  changing both requires a declared experiment. Choose settings on development workloads,
  then freeze settings and separate evaluation tasks before the final comparison.
- Primary improvement gate: at least 10% lower median batch completion time on the declared
  contended workload. Report per-task median/p95 latency, throughput (completed tasks/s),
  admission wait, service queue if observable, failures/timeouts and variability alongside it.
- Guardrails: throughput at least 95% of baseline and p95 task latency at most 105% of baseline
  for each declared clean and contended evaluation condition. Preserve task acceptance-test
  results, with no new failures/timeouts, no leaks or starvation, and no quality regression.
- Measure controller/admission overhead independently from existing instrumentation overhead;
  proposed median wall-time increase at most 5% for unchanged limits. Preserve the existing
  baseline stability gate, a root-step coefficient of variation of at most 15%; noisy baselines
  make comparisons inconclusive.
- Report paired effect estimates and uncertainty; small or noisy samples do not establish
  general causal improvement. Fail, inconclusive, and not-evaluated gates block optimization
  promotion. Keep the current baseline setting and preserve every attempted result.

O02 tests invalid limits, unsupported resources, policy revision conflicts, retry identity,
restoration, and selected-run policy history. O03 tests simultaneous admissions, queue bounds,
deadline expiry, cancellation, exceptions, reduced limits while work is active, and scope
across runners. Do not assert unsupported global enforcement. O04 rejects cross-hardware,
cross-model, stale identity or unpaired comparisons and preserves failure evidence. O05
verifies selected UUIDs, explicit missing data, stale state, keyboard controls, and API errors.
O06 uses real GPU evidence, runs the accepted gates and proves disable/restore behavior.

O07 is deterministic rules first: recommend adjustments from fresh, correlated queue and
completion evidence with accepted minimum coverage/window size. Missing/stale/unsupported
telemetry, insufficient evidence, resource mismatch or attribution uncertainty means no
change. GPU utilization alone is not a queue measurement or sufficient trigger. Validate
min/max limits, cooldown, hysteresis, rate of change, fairness and rollback before actuation.
Begin in shadow mode; only an explicit operator opt-in and passing controlled evaluation
enable changes on designated capacity. Preserve all decisions and rejected actions.

## Demo and handoff

X01 keeps its existing prerequisites and also requires O06. Show the same declared workload
at baseline and the evaluated manual setting with the corresponding saved-run evidence.
If gates fail or remain inconclusive, demonstrate controls and restore behavior while
clearly showing the result; do not claim a speedup or enable an unvalidated controller.
X02 includes policy/evaluation contracts, manifests, runbooks, failure evidence, and hardware
limitations. O07 stays optional and is not a dependency of X01 or X02.

The original eight-week schedule is historical planning, not a commitment that the extra
scope fits without tradeoffs. Reassess remaining time after the O01 contract and A04 evidence.
