# AgentWatch — Implementation Plan

Status: Draft after project pivot, 2026-10-01.

Implementation update, 2026-10-02: the local CPU queue experiment, coding test tool, resource
sampler, completed-run ingestion API, PostgreSQL history, and Next.js dashboard now work.
Use [the project and testing checklist](project-checklist.md) for current implementation and
verification status. The eight-week sequence below is the original target plan, not elapsed progress.

Planning horizon: Eight weeks; relative weeks, not committed calendar dates.

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

## 6. Main risks and fallback

- Replay noise: use the Week 2 checkpoint; fall back to resource classification if waiting-fraction
  labels remain unreliable, and disclose the reduced scope.
- GPU quota: request access early and use a mock server for development; GPU validation still
  requires real inference infrastructure.
- Injection artifacts: hold out an injection method and test natural overload.
- Weak gain over rules: report results honestly; mixed-contention data is central to the comparison.
- Cost: budget alerts and scheduled capacity reduction; evaluate spot capacity for batch generation.
