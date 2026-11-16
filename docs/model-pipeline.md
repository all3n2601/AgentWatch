# AgentWatch model pipeline image and registry

One image runs the whole model pipeline with the repository's own make targets, and every run
ends by pushing the evaluated model to a model registry. Until the team's GCP model registry
and its credentials exist, MLflow's model registry on the local tracking database stands in for
it.

```sh
make model-pipeline-image   # docker build -f pipelines/model/Dockerfile -t agentwatch-model-pipeline .
make model-pipeline-run     # outputs, figures and the MLflow database in .data/model-pipeline/
```

## What the image runs

[`pipelines/model/run_pipeline.sh`](../pipelines/model/run_pipeline.sh), stopping at the first
failure:

1. `make pull-dataset TARGET=data/datasets`: dataset v1 from the public DVC remote (no login).
2. Rebuild `split-v3` from the release and check its SHA-256 against the evaluated split
   (`32488a94…`); a different file stops the run.
3. `make train-candidates`, `make calibrate`, `make bias-check`, `make model-analysis` on that
   split, with the packaged settings (`training.toml`, `calibration.toml`, `bias.toml`).
4. `make register-model`: push the model to the registry.

The image is `ghcr.io/astral-sh/uv:python3.13-bookworm` pinned by digest, with `libgomp1` for
LightGBM and `make`; the locked environments are installed at build time (`uv sync --frozen`),
and it runs as a non-root user. Its build context leaves out local data, results, environments
and the git directory (`pipelines/model/Dockerfile.dockerignore`), so DVC runs without git
(`core.no_scm`). Mount a folder at `/agentwatch/.data` to keep the results; the run writes
nothing outside it and the container.

## Registry rule and rollback

`make register-model` ([`register_model.py`](../pipelines/model/src/agentwatch_model/register_model.py))
follows [`registry.toml`](../pipelines/model/src/agentwatch_model/registry.toml) (`registry-v2`):

- It refuses a candidates folder that is not an evaluated test report, a classifier that does
  not match its report's hash, an unfitted calibration, or a calibration of another training
  version. It needs the bias report (`make bias-check`); `NO_BIAS_REPORT=1` (`--no-bias-report`)
  pushes without one, tagged `bias_test_outcome = unavailable`.
- A model is registered once per **identity**: training and calibration versions, split
  SHA-256, dataset version and classifier SHA-256. A rerun of the same model returns the
  existing version and changes no alias, so reruns cannot erase the rollback target.
- Each new version is one run in the `agentwatch-registry` experiment (model files, serving
  contract, calibration, reports), tagged with its identity, test macro F1, `blocks_promotion`,
  eligibility and bias outcome; `candidate` moves to it.
- `champion` moves to it only when it is **eligible** (no failed contract or calibration gate,
  no failed bias check on test, and not blocking promotion), was scored on the **same split and
  dataset** as the current champion, and its test macro F1 is at least the champion's
  (`max_macro_f1_drop = 0`). Otherwise the champion stays (the automatic rollback), with the
  reason in `.data/registry/registration.json`. `make rollback-model` points `champion` back at
  the version it replaced.

While every model blocks promotion (four contract gates are `not_evaluated`), no version is
`champion`; `candidate` is the newest evaluated model. Serving must refuse versions tagged
`blocks_promotion = true` ([serving contract](model-serving-contract.md)).

A registered version's `source` is where its artifacts were logged: inside the image that is
`/agentwatch/.data/mlflow/artifacts/...`, which `make model-pipeline-run` mounts from
`.data/model-pipeline/` on the host, so open that tracking database from the same layout. On a
Linux host the bind mount is written by the image's user (uid 10001); give that uid write access
to `.data/model-pipeline/` first (Docker Desktop on macOS needs nothing).

## Run of 2026-11-15

`make model-pipeline-image` then `make model-pipeline-run` on a laptop (Docker, arm64): the image
built in about 40 s and the run took about 20 s. The rebuilt `split-v3` matched (`OK`), and the
model it trained is byte-identical to the local `training-v3+71c4dd58` run (same `classifier.txt`
and `predictions.jsonl`; test macro F1 0.790). It was registered as version 1 of
`agentwatch-attribution` under `candidate`, tagged `blocks_promotion = true`; `champion` was not
set, because it blocks promotion and the bias check fails on test ([model bias](model-bias.md)).

## CI, alerts and the GCP registry

[Model CI/CD](model-ci.md) runs this image on GitHub Actions, alerts Slack and pushes eligible
models to GCP Artifact Registry. It needs the one-time setup in that document.

## Still needed

- Run the workflow once on GitHub with the GCP variables and the Slack secret set, and record
  the result (the workflow and message are verified locally only).
- A persisted registry or champion pointer, so the champion comparison and rollback work across
  CI runs (the MLflow database is rebuilt in each run).

Follow-ups that change shared CI or are optimisations: building the image and running hadolint
in CI, pinning the apt packages, and copying the lockfiles before the source so a code change
does not reinstall every dependency layer.
