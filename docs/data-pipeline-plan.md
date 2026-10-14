# AgentWatch data pipeline plan

Status: 2026-10-13. Owner: data pipeline workstream. The column-level contract is
[the data card](data-card.md); model gates and split settings are in the proposed evaluation
contract ([`docs/evaluation-contract.md`](evaluation-contract.md), #11). This plan says what is built, what comes
next, and what the pipeline needs from other workstreams.

## What the pipeline does

```text
recorded runs (local results.json + samples.jsonl, or GET /v1/runs)
  -> extract   validate strictly, deduplicate across sources, record telemetry status
  -> flatten   one row per step, resource readings per step window, run lineage
  -> label     experiment-derived labels against a frozen, fingerprinted rule file
  -> build     schema validation, content-addressed version, steps.parquet + manifest.json
  -> gate      TFDV statistics against the frozen schema (pipelines/validation)
  -> split     70/15/15 grouped by task_id, measured-only test, leakage check
```

Every stage rejects untrustworthy runs individually with a reason, including runs whose rows
break the dataset schema. Only an unreadable source, invalid label rules, an empty result, or a
defect in the builder itself stops a build, and a failed build
writes nothing. Each stage is in `pipelines/data/src/agentwatch_data/`, tested at 100% line and
branch coverage, and exercised against real recorded runs and the live ingestion API.

## Roadmap

| Step | Delivers | Status |
|---|---|---|
| Schema and data card | `steps-v1.2` columns, roles, leakage guards, row validation | Done (#1, #12, #13) |
| Extract | Strict validation, retries, cursor and row checks, duplicate detection | Done (#2, #10, #13) |
| Flatten | Step rows, telemetry windows, coverage against the declared interval | Done (#3, #12, #13) |
| Label | `labels-v1.1`, `injection_method`, per-run rejection | Done (#5, #12, #13) |
| 5. Build | `make dataset`: versioned Parquet release, manifest, and per-build report | Done (#14) |
| 6. Quality gate | `make validate-dataset`: TFDV statistics against a frozen, reviewed schema; anomalies fail, drift is reported. Isolated in `pipelines/validation` because TFDV brings TensorFlow (~1.7 GB) | Done (#16) |
| **7. Splits** | `make split-dataset`: 70/15/15 grouped by `task_id` from the evaluation contract, measured-only test split, a deterministic split file with per-split row hashes, and the contract's leakage check | This PR |
| 8. Versioning | DVC with a GCS remote; a rebuild from the same evidence reproduces the same version | Next |
| Hardening pass | Request batching for large API histories; sweeping staging left by killed builds; anything left from reviews | Before Airflow |
| 9. More resources | Prometheus reader; `steps-v2` inference and retrieval features; `labels-v2` with mixed contention. Split follow-ups: force at least one injection method wholly into test, and check that mixed-contention cases reach test | Waits on B |
| 10. Real collection | Scheduled runs over a task suite on a Linux GCP VM, where PSI and cgroup readings exist. Split follow-up: rebuild and re-split once step events carry a real `task_id`, so the test split holds unseen tasks and training is no longer empty | Waits on A, B, F |
| 11. Airflow DAG | collect → extract → validate → build → split → version, with retries and alerts | After 6–8 |
| 12. Reports | Quality, slice, label-sanity, and drift statistics per release | After 11 |
| 13. Shared features | The live attribution service reuses `window_readings` and related code | With E |
| 14. Demo | Corrupted evidence rejected live; lineage from model to dataset to raw run | Last |

Split items owned elsewhere: the minimum test counts are judged by evaluation from the counts
each split file records, and freezing a new test split after 5 promotion evaluations is the
promotion owner's call (it means a new `SPLIT_METHOD`).

## What the pipeline needs from other workstreams

| From | Need | Why |
|---|---|---|
| A, agent | A real `task_id` (and `step_type`) in step events | Today's `task_id` is a placeholder (`kind:workload:iterations`), so grouped splits cannot guarantee unseen tasks |
| B, contention | Prometheus metric names for inference and retrieval; at least a second injection method; Qdrant in its own CPU-limited container | `steps-v2` features, the contract's held-out injection method, and separable retrieval labels |
| Worker owner | Faster or per-step sampling (250 ms leaves short steps without coverage); the `CpuSampler.summarize` reset fix; monotonic durations; strict contract bounds; `demo.py --output` defaulting by workload; agent evidence written when the LLM loop fails | Measurement quality at the source |
| F, platform | A pinned CI Postgres image (`public.ecr.aws/docker/library/postgres:17-alpine@sha256:b0f9560a…`, verified identical to Docker Hub's) | Docker Hub's anonymous pull limit has failed CI before tests ran |
| D, model | Consume releases by `dataset_version`; record `label_version` and the contract hash in reports | Reproducible evaluation |

## Known limits

The data card's [known limitations](data-card.md#known-limitations) apply to every release. The
most important today: on CPU-only data the label equals the rule's prediction on every row, so
no model evaluation is informative until inference, retrieval, and mixed contention exist.
