# 0013: Model CI pushes to GCP Artifact Registry through two generic packages

Date: November 15, 2026. Status: proposed in the `feature/model-ci-cd` PR.

The model pipeline image ([model pipeline](../model-pipeline.md)) registers each evaluated model
in a local MLflow registry. The course guidelines also ask for CI that trains and validates on
every model change, blocks or alerts on failed validation or bias checks, pushes a passing model
to a GCP registry, and notifies on failures.

**Decision.**

- GitHub Actions (`.github/workflows/model.yaml`) runs the existing image. Pull requests run
  everything except the push and the alert; `main` and manual runs also push and alert.
- Artifact Registry's **generic** format holds the model files. Two packages mirror the
  registry rule: `attribution-candidate` takes every evaluated model, `attribution-champion`
  takes only an eligible one (`registry.toml`). Versions are the model's identity hash prefix, so
  a rerun is the same immutable version and is not uploaded again.
- Authentication is Workload Identity Federation limited to this repository's `main` branch and
  a service account with writer access to one repository; no key file is stored. The Slack
  webhook is the only secret, under the dataset DAG's existing name.
- The decision logic is Python (`ci_report.py`, tested without network); the workflow only
  moves files and calls `gcloud` and the Slack action.

**Consequences.** MLflow's database is rebuilt in each CI run, so the champion-versus-previous
comparison is not applied across runs and a rollback is choosing an earlier version in the
champion package. A persisted registry or a champion pointer file is open work. Docker
(Artifact Registry's other format) was not chosen because the artifact is a set of model
files, not an image. The workflow is not a required check, so it does not block merges.
