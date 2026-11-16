# Model CI/CD, Slack alerts and GCP Artifact Registry

[`.github/workflows/model.yaml`](../.github/workflows/model.yaml) runs the
[model pipeline image](model-pipeline.md) whenever code that affects the model changes
(`pipelines/model`, `pipelines/data`, `pipelines/versioning`, `data/*.dvc`, `Makefile`,
`uv.lock`, the workflow itself), and on demand (Actions → Model pipeline → Run workflow).

| Event | Runs the pipeline | Keeps reports | Pushes to GCP | Alerts Slack |
|---|---|---|---|---|
| Pull request | yes | yes | no | no (the check and job summary report it) |
| Push to `main` | yes | yes | yes, if configured | yes, if configured |
| Manual run on `main` | yes | yes | yes, if configured | yes, if configured |

Steps, stopping at the first failure: build the image → pull dataset v1 and rebuild `split-v3`
(hash-checked) → train, calibrate, bias-check, draw figures → register in MLflow → decide what
may be pushed ([`ci_report.py`](../pipelines/model/src/agentwatch_model/ci_report.py)) → push →
Slack. The figures, reports and `registration.json` are kept as the `model-pipeline-<run id>`
workflow artifact for 30 days.

## What gets pushed

A generic-format Artifact Registry repository holds two packages. The version is
`m-<first 12 hex of the model's identity hash>`, so a rerun of the same model names the same
version, and the workflow skips an upload that already exists.

| Package | Receives | Rule |
|---|---|---|
| `attribution-candidate` | every evaluated model | mirrors MLflow's `candidate` alias |
| `attribution-champion` | only an **eligible** model | no failed gate, no failed bias check on test, not blocking promotion ([`registry.toml`](../pipelines/model/src/agentwatch_model/registry.toml)) |

Each version holds `candidates/` (classifier, regressor, preprocessing, report), `calibration/`
and `registration.json` (the identity, gates and bias outcome). Serving must refuse a version
whose `registration.json` has `blocks_promotion: true` ([serving
contract](model-serving-contract.md)).

**Today no model is eligible** (contract gates are `not_evaluated` and the bias check fails on
test), so only `attribution-candidate` fills and Slack reports "not eligible" with the reasons.

**Rollback.** MLflow's database does not outlive a CI run, so the registry's "keep the champion
unless the new model is at least as good" comparison is not applied across runs. The champion
package holds every eligible model; roll back by serving an earlier version
(`gcloud artifacts versions list --package=attribution-champion ...`). Open: a persisted
registry or a champion pointer in the repository.

## One-time GCP setup (you run these; nothing here ran against a real project)

Replace the three placeholders. Authentication is Workload Identity Federation, so no key file
exists to leak.

```sh
gcloud auth login
PROJECT_ID=<your-project-id>
LOCATION=us-central1
REPO=agentwatch-models
GITHUB_REPO=all3n2601/AgentWatch
SA=agentwatch-model-ci@$PROJECT_ID.iam.gserviceaccount.com
gcloud config set project $PROJECT_ID
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

gcloud services enable artifactregistry.googleapis.com iamcredentials.googleapis.com

# 1. The repository (generic format holds arbitrary model files)
gcloud artifacts repositories create $REPO --repository-format=generic --location=$LOCATION \
  --description="AgentWatch attribution models"

# 2. A service account that may only write to that repository
gcloud iam service-accounts create agentwatch-model-ci
gcloud artifacts repositories add-iam-policy-binding $REPO --location=$LOCATION \
  --member="serviceAccount:$SA" --role=roles/artifactregistry.writer

# 3. Let GitHub Actions runs of this repository on main act as that account
gcloud iam workload-identity-pools create github --location=global
gcloud iam workload-identity-pools providers create-oidc github --location=global \
  --workload-identity-pool=github --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='$GITHUB_REPO' && assertion.ref=='refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding $SA --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$GITHUB_REPO"

# 4. Repository variables (not secrets: none of these grants access by itself)
gh variable set GCP_PROJECT_ID --body "$PROJECT_ID"
gh variable set GCP_LOCATION --body "$LOCATION"
gh variable set GCP_ARTIFACT_REPOSITORY --body "$REPO"
gh variable set GCP_SERVICE_ACCOUNT --body "$SA"
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER \
  --body "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github"
```

The push step runs only when `GCP_WORKLOAD_IDENTITY_PROVIDER` is set and the run is on `main`.
Check a push with:

```sh
gcloud artifacts packages list --repository=$REPO --location=$LOCATION
gcloud artifacts versions list --package=attribution-candidate --repository=$REPO --location=$LOCATION
```

## Slack alerts

Create a Slack incoming webhook for the channel, then store it as the repository secret
(the same name the dataset DAG reads from `.env`):

```sh
gh secret set AGENTWATCH_ALERT_SLACK_WEBHOOK_URL
```

Alerts on `main` and manual runs:

- the verdict after every run that reached it: eligible or not, failed gates, bias result on
  test, where the model went, and the run link. A clean channel therefore means no run happened.
- a failure alert when training, validation or a gate stops the pipeline before a verdict.
- a registry push failure, reported in the verdict message.

The webhook URL is a secret; the message carries gate names and reasons, never data or
credentials.

## Verification status

- `ci_report.py`: 22 unit tests (`pipelines/model/tests/test_ci_report.py`), plus `stage` and
  `message` run against real output of `make model-pipeline-run` on 2026-11-15.
- The workflow passes `actionlint`; it has **not** run on GitHub, and the GCP push and Slack
  delivery have **not** been exercised against a real project or webhook.
- The repository's required checks (`javascript`, `python`, `go`, `infrastructure`) are
  unchanged; this workflow is not required, so a failure here does not block merging.
