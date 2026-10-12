# AgentWatch data pipeline plan

Status: 2026-10-11. Owner: data pipeline workstream. The column-level contract is
[the data card](data-card.md); model gates are in the proposed evaluation contract (#11,
`docs/evaluation-contract.md` once merged). This plan says what is built, what comes
next, and what the pipeline needs from other workstreams.

## What the pipeline does

```text
recorded runs (local results.json + samples.jsonl, or GET /v1/runs)
  -> extract   validate strictly, deduplicate across sources, record telemetry status
  -> flatten   one row per step, resource readings per step window, run lineage
  -> label     experiment-derived labels against a frozen, fingerprinted rule file
  -> build     schema validation, content-addressed version, steps.parquet + manifest.json
```

Every stage rejects untrustworthy runs individually with a reason. Only an unreadable source,
invalid label rules, a schema violation, or an empty result stops a build, and a failed build
writes nothing. Each stage is in `pipelines/data/src/agentwatch_data/`, tested at 100% line and
branch coverage, and exercised against real recorded runs and the live ingestion API.

## Roadmap

| Step | Delivers | Status |
|---|---|---|
| Schema and data card | `steps-v1.2` columns, roles, leakage guards, row validation | Done (#1, #12, #13) |
| Extract | Strict validation, retries, cursor and row checks, duplicate detection | Done (#2, #10, #13) |
| Flatten | Step rows, telemetry windows, coverage against the declared interval | Done (#3, #12, #13) |
| Label | `labels-v1.1`, `injection_method`, per-run rejection | Done (#5, #12, #13) |
| **5. Build** | `make dataset`: versioned Parquet release with manifest | This PR |
| 6. Quality gate | TFDV statistics and a frozen schema; anomalies fail the build. TFDV 1.21 runs on Python 3.13 (verified) but pulls in TensorFlow (~1.7 GB), so it is an optional extra used by the pipeline job, not a core dependency | Next |
| 7. Splits | 70/15/15 grouped by `task_id`, frozen to a split file; the contract's leakage check | Next |
| 8. Versioning | DVC with a GCS remote; a rebuild from the same evidence reproduces the same version | Next |
| Hardening pass | Request batching for large API histories; anything left from reviews | Before Airflow |
| 9. More resources | Prometheus reader; `steps-v2` inference and retrieval features; `labels-v2` with mixed contention | Waits on B |
| 10. Real collection | Scheduled runs over a task suite on a Linux GCP VM, where PSI and cgroup readings exist | Waits on A, B, F |
| 11. Airflow DAG | collect → extract → validate → build → split → version, with retries and alerts | After 6–8 |
| 12. Reports | Quality, slice, label-sanity, and drift statistics per release | After 11 |
| 13. Shared features | The live attribution service reuses `window_readings` and related code | With E |
| 14. Demo | Corrupted evidence rejected live; lineage from model to dataset to raw run | Last |

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
