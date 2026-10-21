# Shared dataset snapshots

Archives committed here are interim team copies of a gated dataset release, for pipeline work
(such as the Airflow DAG) until the DVC remote exists. They are not the versioned stores:
`data/evidence/` and `data/datasets/` remain DVC-managed and untracked by Git
(see `docs/data-card.md`, Versioning). Each archive mirrors the repository layout, so unzip it
at the repository root; its own `README.md` describes the collection and its limitations, and
`SHA256SUMS` lists every file.

| Archive | Release | Contents | SHA-256 |
|---|---|---|---|
| `agentwatch-dataset-steps-v1.2-5954f97ef8b2.zip` | `steps-v1.2+5954f97ef8b2` | 350 rows from 25 CPU and 25 coding runs recorded 2026-10-21 UTC (TFDV gate passed; split frozen but blocked by the leakage check), plus 5 GPU-sampled agent runs as raw evidence | `ba58a5f057da22ceee33e58a680d02db878912d8807b2c826730b5224e1098da` |

This release can train or evaluate nothing yet: it has two tasks and two classes, and its
features separate the classes almost perfectly. Use it to build and test pipelines.

Once the team bucket exists, version the stores with `make version-dataset` and
`make push-dataset` instead of adding archives here, and remove this directory.
