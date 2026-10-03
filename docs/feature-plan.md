# AgentWatch — Feature Plan

Status: Local CPU profiling, completed-run ingestion, PostgreSQL history, and the custom dashboard
are implemented. General agent intake, three-resource attribution, and the ML lifecycle remain.

See [the project and testing checklist](project-checklist.md) for current completion status,
verification evidence, testing gaps, and remaining milestones. The features below describe the
full target rather than a claim that every feature is already implemented.

Planning basis: [Problem statement and project plan](problem-statement-and-plan.md), supplied for the project pivot on 2026-10-01.

Target: An eight-week DADS 7305 MLOps final project.

## 1. Project in one sentence

AgentWatch profiles agent steps to estimate how much time was spent working versus waiting,
identify the saturated resource, and show confidence in that attribution.

## 2. Problem and users

An agent mixes model inference, sandbox execution, retrieval, and orchestration. A slow step
can reflect expensive work or contention. Teams need to distinguish these cases before deciding
which resource to scale.

Primary users are platform and infrastructure engineers operating self-hosted agents. AI engineers
use the results to investigate overhead; platform leads use them to guide capacity decisions.

## 3. Core workflow

1. Record each agent step with OpenTelemetry and a resource-class tag.
2. Sample CPU, GPU/inference, and retrieval saturation during execution.
3. Join the step span with its resource measurements.
4. Predict the waiting resource and waiting fraction using a versioned model.
5. Store the attribution and display estimated working/waiting time, confidence, and evidence.
6. Check model quality with controlled canaries and monitor feature drift.
7. Retrain when needed and promote only candidates that pass evaluation gates.

## 4. MVP features

| Feature | Capability | Demonstration or verification |
|---|---|---|
| Step telemetry | Step IDs, run IDs, timestamps, resource tags, and workload size | Reconstruct one complete coding-agent run |
| Resource sampler | CPU PSI/throttling, GPU utilization, vLLM queues, retrieval latency at a target 250 ms interval | Align samples with completed spans and flag missing readings |
| Replay and injectors | Clean repeated runs, isolated CPU/GPU/retrieval contention, and mixed contention | Compare a step with its clean baseline |
| Labeled dataset | Baseline duration/noise, slowdown labels, and waiting-fraction targets | Reproduce labels from experiment records without hand labeling |
| Rule baseline | Versioned resource thresholds | Evaluate on the same held-out data as the learned model |
| Learned attribution | LightGBM classification and regression | Report resource macro F1/accuracy and waiting-fraction MAE |
| Live correlation service | FastAPI joins spans and measurements, loads the MLflow model, and writes Postgres results | Attribution available about 15 seconds after step completion |
| Grafana views and alerts | Estimated working/waiting split, resource, confidence, uncertainty, and repeated-wait alerts | Locate the resource chosen for the demo injection |
| Quality monitoring | Controlled canary checks and input drift checks | Detect a degraded model using known-label evaluation |
| Retraining and release | Airflow pipeline, tracked datasets/models, and GitHub Actions promotion gate | Reject a candidate that regresses against the incumbent |

## 5. Scope and acceptance targets

- One coding-agent workload, one GKE cluster, and three resource classes: GPU inference,
  CPU sandboxes, and retrieval. Predictions also include `none`; uncertain results are displayed
  explicitly when confidence is insufficient.
- Aim for a 10+ percentage-point improvement over the rule baseline on mixed-contention cases;
  specify the primary metric before evaluating.
- Label at least 95% of clean evaluation steps `none`.
- Return attributions within about 15 seconds of completion.
- Measure instrumentation overhead against uninstrumented runs; set a numerical budget before
  declaring overhead negligible.
- Demonstrate attribution with a resource selected by a reviewer and preserve a backup recording.

Automatic remediation, predictive scaling, GPU kernel profiling, answer-quality evaluation,
and enterprise governance workflows are outside this MVP.

## 6. Delivery sequence

Follow the eight-week sequence and verification checkpoints in the
[implementation plan](implementation-plan.md).
