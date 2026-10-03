# AgentWatch — Resource-Class Bottleneck Profiler for Agentic Systems

*Problem statement and project plan, DADS 7305 (MLOps) final project*

---

## 1. Problem statement

### Context

AI agents aren't single model calls. One coding-agent task mixes several kinds of work, each using a different resource:

- GPU inference for the language model
- CPU-heavy code execution in sandboxes
- Memory-heavy retrieval from a vector database
- Overhead such as tool schemas and helper calls

Research shows that much of an agent's latency often comes from outside the model. In tool-heavy workloads, CPU-side tool processing has been measured at up to 88% of end-to-end latency [1]. The bottleneck also shifts from request to request as load changes [2].

### The gap

Today's tracing tools record how long each step took. The better ones also show infrastructure metrics on the same screen. But they can't tell whether a slow step was slow by nature or stuck waiting for a crowded resource. An engineer has to look at both and guess.

### Why it matters

Those two cases need opposite fixes. If sandboxes are CPU-starved, buying another GPU costs a lot and changes nothing. If the GPU queue is the problem, adding CPU doesn't help. Platform teams make these scaling decisions regularly, mostly by intuition, and wrong guesses waste money.

### Who has this problem

- **Platform and infrastructure engineers** at companies running agents on their own infrastructure, especially coding and research agents with self-hosted models and sandboxes. They are the primary users.
- **AI engineers**, who can use it to find wasted overhead in agent design.
- **The head of platform**, who usually signs off. The pitch to them is cost: stop buying GPUs when the bottleneck is CPU.

---

## 2. Objective

For every agent step, the system will answer three questions automatically:

1. How much of this step's time was real work, and how much was waiting?
2. If it waited, which resource was saturated: GPU inference, CPU sandboxes, or retrieval?
3. How confident is that answer, and how accurate has the model been recently?

---

## 3. What we plan to build

### 3.1 Telemetry

- A lightweight **OpenTelemetry plugin** in the agent records each step's start and end time and tags it with its resource class.
- A **node sampler** on each machine records saturation signals every 250 ms: CPU pressure (PSI) and cgroup throttling, GPU queue length and utilization (DCGM), vLLM queue metrics, and vector database latency.

### 3.2 Labeled dataset

No public dataset exists, so our team generates one with automatic labels. No rows are labeled by hand: the labels come from how each experiment is set up.

1. Record clean agent runs on a fixed set of 30–50 coding tasks, with the LLM frozen at temperature 0.
2. Replay each step several times with no contention to measure its normal (baseline) duration and natural noise.
3. Replay steps again while deliberately jamming one resource, or two at once:
   - **CPU:** stress-ng on the sandbox nodes
   - **GPU:** a competing request flood on vLLM
   - **Retrieval:** a query flood on Qdrant
4. Scripts assign two labels to each step:
   - **Which resource it waited on:** the injected resource, but only if the step slowed significantly (at least 2–3 standard deviations above its baseline). Otherwise the label is `none`.
   - **Waiting fraction:** `(contended duration − baseline duration) ÷ contended duration`.

For mixed contention, each step is also replayed under each injector separately. The resource whose injector alone caused the largest slowdown is the primary label.

The injection log is used only for labeling and is **never** given to the model as a feature.

**Example row**

| Field | Example |
|---|---|
| Step | Run tests (CPU sandbox step) |
| Duration | 4.2 s (baseline 1.3 s) |
| Features | CPU pressure 78%, throttled 60%, GPU queue 2, Qdrant latency normal, 3 agents running |
| Label: resource | CPU sandbox |
| Label: waiting fraction | 0.69 |

### 3.3 Model

- A **LightGBM** model trained on per-step features: step type, duration, size, and the saturation readings during the step. It makes two predictions:
  - the waiting resource (classification)
  - the waiting fraction (regression)
- A **rule-based threshold system** serves as the baseline. The model must beat it, especially when several resources are under pressure at once, because that's where rules break down.

**Evaluation**

- Accuracy and macro F1 for naming the resource
- Mean absolute error for the waiting fraction
- Tests on held-out data: one injection method held out entirely, unseen tasks, and mixed-contention cases
- A natural-contention check: overload the system with concurrent agents and compare GPU predictions against vLLM's own queue-time metrics

### 3.4 Deployment

A **correlation service** (FastAPI on GKE) runs live:

1. It receives each step's span.
2. It pulls the saturation readings for that time window and builds the feature row.
3. It scores the row with the model registered in MLflow.
4. It writes the result to Postgres.

**Grafana** shows each step split into working and waiting, with the named resource and a confidence score. Low-confidence results are shown as "uncertain" rather than guessed. Alerts fire when a resource repeatedly causes waiting.

### 3.5 Monitoring and retraining

- **Canary injections** periodically create small, known slowdowns in production and check whether the model names them correctly. This gives a live accuracy measurement against ground truth.
- **Drift checks** watch the input features.
- When accuracy drops or the data drifts, an **Airflow** pipeline retrains the model.
- **GitHub Actions** blocks any new model that performs worse than the current one.

---

## 4. Success criteria

- The model beats the rule baseline at naming the resource, by about 10+ points on mixed-contention cases.
- Clean runs are labeled `none` at least 95% of the time.
- Attributions appear within about 15 seconds of a step finishing.
- The profiler adds negligible overhead to the agents it monitors.
- In the live demo, it correctly names a resource that a judge chose to jam.

---

## 5. Scope

**In scope**

- One coding-agent workload
- Three resource classes: GPU inference, CPU sandboxes, and retrieval
- One GKE cluster

**Out of scope**

- Automatically fixing bottlenecks
- Predictive scaling
- GPU kernel profiling
- Judging the quality of agent answers

---

## 6. Tech stack

| Layer | Tools |
|---|---|
| Workload | Python ReAct agent, vLLM with a 7B coding model on an NVIDIA L4, Docker sandboxes, Qdrant |
| Telemetry | OpenTelemetry, custom sampler (PSI, cgroups, DCGM, vLLM metrics), Prometheus |
| Data | Parquet on GCS, DVC, TFDV |
| Model | LightGBM, MLflow tracking and registry |
| Serving | FastAPI on GKE, Postgres, Grafana, Alertmanager |
| Pipelines and CI/CD | Airflow, GitHub Actions, Helm, Artifact Registry |
| Platform | GCP, GKE, Docker |

---

## 7. Plan (8 weeks)

| Week | Work |
|---|---|
| 1 | Cluster, GPU quota, agent and vLLM running, telemetry reaching Grafana |
| 2 | Step recorder, replay, injectors. **Checkpoint:** replay timing must vary less than 10–15% |
| 3 | Labeled dataset v1, TFDV validation, rule baseline |
| 4 | LightGBM v1, mixed-contention data, held-out tests |
| 5 | Model in live service, CI/CD gate |
| 6 | Canaries, drift alerts, Airflow retraining |
| 7 | Cost analysis, trace redaction, demo controls |
| 8 | Rehearsals, backup recordings, documentation |

---

## 8. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| Noisy replay labels | Week-2 checkpoint. If labels are too noisy, fall back to predicting only which resource was contended. |
| GPU quota delays | Request quota immediately; develop against a mock LLM server meanwhile. |
| The model doesn't beat the rules | Make mixed-contention cases a large part of the test set. |
| The model learns the injection patterns rather than real contention | Hold out one injection method for testing; validate on natural overload. |
| Cloud costs | Scale node pools to zero overnight; use spot GPUs for bulk data generation; set budget alerts. |

---

## 9. One line

We find the crowded station in the AI's kitchen, so teams fix the right thing.

---

## References

1. "A CPU-Centric Perspective on Agentic AI," Georgia Tech and Intel, arXiv:2511.00739.
2. Chang et al., "From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems" (AgentSysBench), arXiv:2608.15127.
