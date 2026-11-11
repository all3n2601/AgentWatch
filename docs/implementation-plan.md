# AgentWatch — Implementation Plan

Status: Draft after project pivot, 2026-10-01.

Implementation update, 2026-10-02: the local CPU queue experiment, coding test tool, resource
sampler, completed-run ingestion API, PostgreSQL history, and Next.js dashboard now work.
Use [the project and testing checklist](project-checklist.md) for current implementation and
verification status. The eight-week sequence below is the original target plan, not elapsed progress.

Planning horizon: Eight weeks; relative weeks, not committed calendar dates.

Planning addition, 2026-10-13: the [optimization layer plan](optimization-plan.md) adds
manual inference/test concurrency controls and controlled evaluation after reliable telemetry.
It assigns O01–O07 across all six contributors, reserves GPU execution for Benoy, and keeps
adaptive control as a gated stretch task. This is new scope, not implemented behavior or an
extension of the original schedule with committed dates.

Planning addition, 2026-11-11: [deeper inference and GPU diagnosis](#7-deeper-inference-and-gpu-diagnosis)
adds request-phase explanations, evidence-backed hypotheses and controlled confirmation.
This is planned scope, not implemented diagnosis or a commitment to additional calendar weeks.

## 1. Architecture

AgentWatch is a resource-bottleneck profiler with an MLOps lifecycle for its attribution model.
The supplied [problem statement](problem-statement-and-plan.md) is the planning basis.

```text
Coding agent -> OpenTelemetry step spans -> FastAPI correlation service
Node sampler -> Prometheus measurements -> FastAPI correlation service
                                             -> MLflow-registered model
                                             -> Postgres attributions -> Grafana / alerts

Clean replay + contention injectors -> labeled Parquet dataset on GCS
                                    -> DVC + validation
                                    -> rule baseline + LightGBM training
                                    -> held-out evaluation + CI gate -> MLflow registry
Canary evaluation + drift checks -> Airflow retraining pipeline
Accepted telemetry + manual concurrency policy -> runner admission limits
                                               -> paired evaluation + dashboard comparison
```

| Layer | Planned tools |
|---|---|
| Agent workload | Python ReAct agent, vLLM with a 7B coding model on an NVIDIA L4, Docker sandboxes, Qdrant |
| Telemetry | OpenTelemetry, custom sampler, Prometheus, CPU PSI/cgroups, DCGM and vLLM metrics |
| Data | Parquet on GCS, DVC, TFDV |
| Models | Rule thresholds; LightGBM classification and regression; MLflow tracking/registry |
| Serving | FastAPI, Postgres, Grafana, Alertmanager |
| Automation | Airflow, GitHub Actions, Helm, Artifact Registry |
| Platform | GCP, GKE, Docker |
| Optimization | Versioned manual concurrency limits; optional bounded rule controller after evaluation |

These are proposed dependencies from the supplied plan, not a claim that they are installed or
that this repository currently deploys them. Confirm runtime compatibility during implementation.

## 2. Data and labeling contract

Each step record needs a run ID, step ID, workload/task reference, resource class, start/end times,
duration, relevant size features, and time-window saturation aggregates. Track telemetry coverage
and model/dataset versions alongside predictions.

Generate clean runs over 30–50 fixed coding tasks with temperature 0. Replay steps without
contention to estimate baseline duration and noise. Fix inputs, model configuration, and relevant
sandbox/retrieval state; temperature 0 alone does not guarantee repeatable timing.

Inject CPU contention with stress-ng, inference contention with competing vLLM requests, and
retrieval contention with Qdrant query floods. Declare a slowdown threshold within the proposed
2–3 baseline standard-deviation range before generating labels. A step that does not cross it is
labeled `none` even when an injector was active.

The proposed regression target is `(contended duration - baseline duration) / contended duration`.
It estimates excess latency attributable to the experiment; it is not direct measurement of all
queue time. Define treatment of negative/noisy values and invalid timings before dataset creation.

For mixed contention, replay each injector separately and choose the resource producing the
largest isolated slowdown as the primary label. Preserve secondary effects and ties in experiment
metadata; a single primary label cannot fully describe simultaneous bottlenecks.

Keep injection logs and label-generation metadata out of model features. Group related replays
and baselines together during splitting to prevent leakage. Freeze held-out sets before tuning.

## 3. Evaluation and serving contract

- Compare both models on identical splits using resource accuracy/macro F1 and waiting-fraction MAE.
- Include unseen tasks, an entirely held-out injection method, and mixed-contention cases.
- Check natural contention from concurrent agents; compare GPU predictions with vLLM queue-time
  measurements where available. This does not establish ground truth for CPU or retrieval.
- Validate confidence on held-out data and choose an abstention threshold; raw classifier
  probabilities alone do not establish calibrated confidence.
- Derive displayed waiting time from predicted fraction times span duration. Label the split as
  an estimate and include model version and telemetry quality.
- Track serving latency and instrumentation overhead independently of prediction quality.
- Define promotion thresholds for the primary metric, clean-run false positives, and regression
  error before running the CI release gate.

## 4. Eight-week sequence

| Week | Deliverable | Verification checkpoint |
|---|---|---|
| 1 | GKE/GPU access, agent, vLLM, Qdrant, telemetry, Grafana | A full run produces aligned spans and resource measurements; use a mock LLM during quota delays |
| 2 | Step recorder, repeatable replay, resource injectors | Clean replay timing varies less than the proposed 10–15% target; define the statistic and repeats |
| 3 | Versioned dataset v1, validation, rule baseline | Labels reproduce; injection metadata is excluded from features; splits prevent replay leakage |
| 4 | LightGBM v1, mixed contention, held-out evaluation | Compare classification/regression against rules and verify uncertainty behavior |
| 5 | Live FastAPI service, MLflow loading, Postgres results, CI/CD gate | Attribution appears within about 15 seconds; a regressing candidate cannot be promoted |
| 6 | Canary checks, drift checks, Airflow retraining | Known-label canary failure triggers the quality workflow; retraining still passes the release gate |
| 7 | Cost/overhead analysis, trace redaction, demo controls | Measure overhead and cloud spend; avoid sensitive prompts/code in exported telemetry |
| 8 | Documentation, clean-environment demo, backup recordings | Reviewer-selected contention is identified and the run can be reproduced |

Canary injections must run in a bounded, isolated test workload or designated capacity so evaluation
does not disrupt unrelated agent runs. Record their coverage rather than treating canary accuracy
as comprehensive production accuracy.

## 5. Implementation backlog

The Python worker now implements local CPU queue profiling and validated completed-run ingestion
with PostgreSQL history. The Next.js application provides the product dashboard requested during
implementation. The Go API/CLI and SDK/Protobuf files remain scaffolds. Grafana remains a proposed
operational integration. Current completed work, remaining features, and test gaps are tracked in
[the project and testing checklist](project-checklist.md).

- Generalize completed-run ingestion into live run/step lifecycle and automatic dashboard updates.
- Validate hardware measurements on Linux and add inference/retrieval telemetry and injectors.
- Connect one instrumented coding agent through the generalized telemetry client.
- Define dataset, labeling, prediction, and evaluation contracts before model training.
- Add reproducible data generation, validation, strong rule evaluation, and learned attribution.
- Add model serving, MLflow tracking/registration, confidence calibration, and release gates.
- Add monitoring, isolated canary evaluation, retraining, deployment, and operational runbooks.
- Add the [optimization layer](optimization-plan.md): manual limits, runner admission evidence,
  controlled evaluation and restore controls; gate any adaptive follow-up on measured results.
- Add deeper inference and GPU diagnosis through section 7, starting with request phases and
  matched baselines before optional kernel profiling.

## 6. Main risks and fallback

- Replay noise: use the Week 2 checkpoint; fall back to resource classification if waiting-fraction
  labels remain unreliable, and disclose the reduced scope.
- GPU quota: request access early and use a mock server for development; GPU validation still
  requires real inference infrastructure.
- Injection artifacts: hold out an injection method and test natural overload.
- Weak gain over rules: report results honestly; mixed-contention data is central to the comparison.
- Cost: budget alerts and scheduled capacity reduction; evaluate spot capacity for batch generation.

## 7. Deeper inference and GPU diagnosis

Goal: explain why an inference step slowed, beyond naming GPU/inference as the suspected
resource. Show measured phase differences, supporting evidence, alternative explanations,
missing evidence and a suggested confirmation experiment. Keep observations, hypotheses and
experimentally supported conclusions distinct. High utilization alone proves neither saturation
nor the cause of a particular request's slowdown.

### Existing evidence and dependencies

Reuse request identity joins and timing definitions from
[ADR 0007](adr/0007-vllm-request-traces-and-server-metrics.md), the Windows-side GPM path in
[ADR 0008](adr/0008-windows-gpm-exporter-for-wsl2.md), and the
[GPU measurement record](gpu-measurement-access.md). vLLM request spans provide queue,
prefill, decode and token counts in local agent evidence after run closure; live StepEvents
currently do not carry that breakdown. Server-wide metrics cannot substitute for a request's
own wait. Ollama queue time remains unavailable.

Build on the [natural-overload capacity-sweep PR #65](https://github.com/all3n2601/AgentWatch/pull/65)
for compatible concurrency experiments once its protocol and artifacts are accepted. Coordinate
shared runner and experiment changes with that task instead of implementing another sweep.
Assign one primary contributor and task branch per milestone before implementation. GPU
execution stays on Benoy's designated test capacity; no new owner assignments or dates are
committed here.

### Sequenced milestones

1. **G01 — Persist and expose request-phase evidence.** Define versioned API/SDK/storage
   contracts for queue, prefill, decode, token counts, model loading where measured, and runner
   stream timing, keyed to the selected run/step. Preserve existing local evidence. Verify
   exact identity joins, late traces, missing/duplicate/mismatched spans, retries, rollback and
   historical UUID selection. Mark evidence pending until received; post-run traces do not
   establish live phase diagnosis. Unsupported Ollama phases stay unavailable.
2. **G02 — Compare matched baselines.** Record backend, model/weights, quantization, hardware,
   token counts, concurrency, cache state and warm/cold state. Declare matching rules, repeat
   count, variability statistic and slowdown thresholds before evaluation. Compare phase
   durations and token throughput only where compatible; expose configuration differences and
   abstain without a suitable baseline. Keep clock domains separate and do not sum overlapping
   phases or subtract incompatible clocks.
3. **G03 — Explain likely causes with rules.** Distinguish increased queueing, longer prompt
   processing, more generated output and slower generation. Correlate with running/waiting
   requests, KV-cache use, preemptions and GPU compute/DRAM activity. Add temperature, clocks,
   power limits and throttle reasons only where support is verified. Require source records,
   units, baseline, method version, coverage, alternatives and limitations for each explanation.
   Test clean, mixed and ambiguous cases. Device-wide metrics are context, memory allocation
   does not prove bandwidth pressure, and low clocks alone do not prove thermal throttling.
4. **G04 — Confirm through controlled experiments.** Reuse bounded paired experiment controls
   to change one factor, such as concurrency, prompt length or cache state. Preregister order,
   repetitions and acceptance criteria; preserve failures and inconclusive evidence. Validate
   against measured request phases on designated capacity. Uncontrolled comparisons report
   observed differences without causal claims.
5. **G05 — Present diagnosis in the app.** Show phase deltas, supporting signals, alternatives,
   evidence quality and a suggested next experiment for the selected UUID. Browser acceptance
   covers late evidence, missing metrics, insufficient short-step coverage, unmatched baselines
   and historical selection. Distinguish provisional live observations from completed analysis;
   raw rule/classifier scores are not calibrated confidence.
6. **G06 — Optional targeted profiling.** For unresolved cases, assess a compatible profiler
   for short kernel and CPU–GPU transfer captures. Verify Windows/WSL/device/driver support,
   bound capture duration, redact exported artifacts and clean up on failure. Measure profiler
   overhead against an unprofiled baseline before adoption; unsupported capture does not block
   G01–G05.

Merge order: G01 → G02 → G03 → G04 → G05; G06 is a gated follow-up. Start with vLLM phases
and reuse existing feature joins, attribution modules and experiment controls. Keep diagnosis
logic outside routes/rendering. Update API definitions, SDKs, consumers and ordered PostgreSQL
migrations together when needed, preserving history and idempotency.

### Verification and rollout gates

- Declare diagnosis quality, clean-run false-positive, abstention, latency and overhead gates
  before evaluation. Compare any learned follow-up against rules on identical frozen splits;
  exclude injection metadata from features and keep related replays in the same split.
- Cover invalid timings, zero output tokens, counter resets, stale samples, partial coverage,
  conflicting identities and unavailable telemetry. Missing readings never become measured zero.
- Measure actual sampling intervals and coverage. Whole-device GPM windows cannot explain every
  short step or attribute activity to one process; disclose these limits.
- For ingestion/storage changes, run dedicated PostgreSQL integration tests with isolated schemas
  and migration upgrade tests. Run affected language checks and browser flows before shipping.
- Gate each explanation on evidence availability and retain existing views when analysis abstains.
  Record completed milestones, verification and real artifact references in
  [the project checklist](project-checklist.md); keep unverified work open.
