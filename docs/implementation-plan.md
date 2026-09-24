# Model Passport — Implementation Plan

Status: Draft v0.1  
Date: 2026-09-23  
Planning horizon: 12 weeks to a design-partner MVP

## 1. Decision summary

Build Model Passport as a **modular control plane with isolated execution workers**.

| Layer | Choice | Why |
|---|---|---|
| Web application | **Next.js 16 (App Router), React, TypeScript** | Best fit for the dense enterprise dashboard in the visual reference; strong routing, server rendering, component model, and production tooling. |
| Design system | Tailwind CSS + Radix UI primitives | Enables the reference design without locking core behavior into a large component framework; Radix supplies accessible interaction primitives. |
| Control-plane API | **Go 1.27 modular monolith**, `net/http` + `chi` | Strong concurrency and predictable operations while keeping the first release simpler than microservices. |
| API contract | OpenAPI 3.1, generated TypeScript/Python clients | The public API remains conventional REST while clients and tests stay contract-driven. |
| Worker protocol | Protobuf over Connect/gRPC | Type-safe communication across Go and Python, with browser and gRPC compatibility available where useful. |
| Security/ML worker | Python 3.13, FastAPI only for local worker administration | Keeps PyTorch/Transformers/security tooling in its natural ecosystem without making Python the system of record. |
| Durable workflows | Temporal | Native support for retries, timeouts, cancellation, checkpointing, and auditable long-running scans. |
| Primary data | PostgreSQL + row-level security | Transactional source of truth for tenants, assets, Passports, findings, policies, lineage edges, and audit events. |
| Evidence blobs | S3-compatible object storage with versioning/object lock | Separates large evidence bundles from metadata and supports immutable retention. |
| Policy engine | OPA/Rego | Decouples policy decisions from enforcement and works across API, CI/CD, and Kubernetes gates. |
| Local platform | Docker Compose | Fast developer setup for PostgreSQL, Temporal, OPA, and object storage. |
| Production platform | Kubernetes + Helm + Terraform | Matches the customer-worker deployment and isolation requirements. |
| Observability | OpenTelemetry + Prometheus/Grafana/Loki/Tempo | One telemetry contract across the web app, API, workflow engine, and workers. |

### Why this is the best framework choice

The screenshot is primarily a data-heavy application shell, so Next.js is the strongest framework for the product surface. Model Passport is also a security system with long-running, privileged, customer-resident jobs. Using Next.js as the whole backend would blur those trust and execution boundaries. The split above gives the team a fast UI framework, a durable authoritative control plane, and a deliberately isolated ML/security runtime.

Do not begin with microservices. Keep identity, inventory, Passports, findings, evidence metadata, policy, approvals, and reporting as modules in one Go deployable. Run the customer worker and sandbox as separate deployables from day one because they have different trust and compute profiles.

## 2. Product slice for the first release

The first sellable vertical slice is:

1. A user creates an organization and workspace.
2. A model version is registered from a file, S3 URI, or MLflow reference.
3. A customer-resident worker hashes the artifact and collects non-sensitive metadata.
4. The control plane creates an immutable Passport draft tied to the exact digest.
5. The worker runs three initial checks: PII exposure, provenance completeness, and canary detection.
6. Every result creates a reproducible finding and signed evidence reference.
7. OPA evaluates a production policy and returns `allow`, `deny`, or `approval_required`.
8. The web UI renders the model overview shown in the reference, with every score linking to evidence.
9. A CLI command checks the gate and returns stable CI exit codes.
10. Any artifact, evidence, policy, or approval change creates a new Passport version and append-only audit event.

### MVP success criteria

- A new design partner reaches a first Passport in less than 60 minutes.
- The same artifact and test inputs reproduce the same deterministic results.
- No raw dataset rows, prompts, model weights, secrets, or unredacted outputs leave the worker by default.
- Cross-tenant access tests fail at both the application and PostgreSQL RLS layers.
- A denied policy produces CLI exit code `1`; infrastructure failure produces `2`.
- Every status, risk label, and gate decision resolves to a stored evidence record.
- A full demonstration can register, scan, review, approve or deny, and gate one model version end to end.

## 3. Architecture boundaries

```text
Browser
  -> Next.js web/BFF (session and presentation only)
      -> Go control-plane API
          -> PostgreSQL (authoritative metadata + RLS)
          -> S3-compatible store (signed evidence bundles)
          -> Temporal (workflow state)
          -> OPA (policy decisions)

Customer environment
  -> Enrolled worker
      -> artifact/model/dataset connectors
      -> isolated scan and test jobs
      -> local redaction + evidence signing
      -> outbound-only mTLS connection to control plane
```

Rules:

- The browser never calls workers directly.
- Next.js never becomes the authoritative data layer; it uses the Go API.
- The control plane never executes customer model code.
- Workers receive short-lived, tenant-bound credentials and cannot select another tenant.
- Evidence metadata is mutable only while a run is open; finalized bundles are content-addressed and immutable.
- Risk scores are derived views, never evidence substitutes.

## 4. Initial domain modules

| Module | First-release responsibility |
|---|---|
| Identity | Users, organizations, workspaces, service identities, roles, permissions |
| Inventory | Assets, models, model versions, datasets, training runs, deployments |
| Passport | Versioned Passport aggregate, state transitions, freshness, revocation |
| Assessment | Campaigns, test cases, runs, retry/cancel state |
| Findings | Findings, severity, confidence, reproduction, remediation state |
| Evidence | Manifests, hashes, signatures, storage references, retention |
| Lineage | Relationship edges and AI-BOM components in PostgreSQL |
| Policy | OPA bundles, versioned decisions, approvals, exceptions, expiry |
| Workers | Enrollment, heartbeat, capabilities, job leasing, credential rotation |
| Audit | Append-only security and business events |

## 5. Repository shape

```text
model-passport/
├── apps/
│   ├── web/                 # Next.js
│   ├── control-plane/       # Go API modular monolith
│   └── cli/                 # Go CLI
├── services/
│   ├── worker/              # Python customer-resident worker
│   └── sandbox/             # isolated test execution
├── api/
│   ├── openapi/             # public REST contract
│   └── proto/               # worker/internal contracts
├── packages/
│   ├── ui/                  # shared UI and tokens
│   ├── sdk-typescript/
│   └── sdk-python/
├── policy/                  # Rego policy bundles and tests
├── db/
│   ├── migrations/
│   └── queries/
├── deploy/
│   ├── compose/
│   ├── helm/
│   └── terraform/
├── docs/
│   ├── adr/
│   ├── threat-model/
│   └── runbooks/
└── tests/
    ├── contract/
    ├── end-to-end/
    └── tenant-isolation/
```

Use a monorepo, but do not force Go and Python builds through JavaScript tooling. A root task runner may coordinate commands; each language keeps its native dependency and test tooling.

## 6. Twelve-week implementation sequence

### Phase 0 — Security and contracts (Week 1)

- Write ADRs for tenancy, evidence immutability, worker trust, API style, and workflow orchestration.
- Produce a product threat model covering tenant escape, malicious models, poisoned evidence, evaluator manipulation, and worker compromise.
- Define Passport, Finding, Evidence Manifest, Policy Decision, and Audit Event v1 schemas.
- Define the Passport state machine: `draft -> assessing -> review_required -> verified | denied -> expired | revoked`.
- Specify zero-sensitive-data fields and explicit prohibited payloads.

Exit gate: security review accepts the trust boundaries and schema invariants before feature work begins.

### Phase 1 — Platform skeleton (Weeks 2–3)

- Scaffold the monorepo, local Compose environment, CI checks, dependency scanning, and secret scanning.
- Create the Go API with request IDs, structured errors, idempotency keys, audit hooks, and OpenTelemetry.
- Add PostgreSQL migrations, tenant context, RLS, and isolation tests.
- Add organization/workspace/user/service-identity models and development authentication.
- Scaffold the Next.js application shell and responsive navigation from the screenshot.

Exit gate: two seeded tenants cannot read, mutate, infer, or cache each other's records.

### Phase 2 — Passport core (Weeks 4–5)

- Implement assets, models, versions, artifact digests, Passports, findings, evidence manifests, and audit events.
- Implement append-only Passport versions and canonical serialization for signing.
- Add model registration and Passport read APIs.
- Build the Overview UI using real API data: overall status, risk summary, test results, findings, provenance, and lineage.

Exit gate: registering the same artifact is idempotent; registering changed bytes creates a distinct version and Passport.

### Phase 3 — Secure worker and workflow (Weeks 6–7)

- Implement one-time enrollment followed by short-lived worker identity.
- Add capability advertisement, heartbeat, job lease, cancellation, timeout, and retry semantics.
- Create Temporal workflows for registration and assessment.
- Hash artifacts and collect approved metadata inside the worker.
- Store signed, content-addressed evidence bundles in object storage.

Exit gate: kill a worker mid-assessment, restart it, and verify safe recovery without duplicate finalized evidence.

### Phase 4 — Initial assurance engines (Weeks 8–9)

- Implement deterministic provenance completeness checks.
- Implement baseline PII detection with versioned detector configuration.
- Implement synthetic canary insertion/detection in a controlled test fixture.
- Convert test outputs into normalized findings and evidence.
- Add evaluator fixtures, golden results, false-positive review, resource limits, and sandbox restrictions.

Exit gate: every finding reproduces from a fixture and links to its exact inputs, engine version, configuration, and evidence hash.

### Phase 5 — Policy gate and delivery path (Weeks 10–11)

- Add versioned OPA policy bundles and unit tests.
- Implement `allow`, `deny`, and `approval_required` decisions with explanations.
- Add approvals, exceptions, expiry, and revocation audit events.
- Build the Go CLI: `login`, `model register`, `assess`, `verify`, `report`, and `gate`.
- Add one CI integration using the documented exit-code contract.

Exit gate: a policy blocks a known-bad fixture in CI, an authorized exception expires, and the next gate is denied again.

### Phase 6 — Design-partner hardening (Week 12)

- Complete report export, share controls, loading/empty/error states, and accessibility checks.
- Add backup/restore, key rotation, credential revocation, worker upgrade, and incident runbooks.
- Run performance, tenant-isolation, failure-recovery, and zero-sensitive-data tests.
- Record the end-to-end demo and package Helm/Terraform deployment instructions.

Exit gate: production-readiness review passes the engineering criteria in the blueprint, and the design-partner demo runs from a clean environment.

## 7. First implementation backlog

These are the first 12 pull-request-sized deliverables, in dependency order.

- [ ] **1. Record architecture decisions**
  - Build: ADRs for the framework, modular-monolith boundary, tenancy, worker identity, and evidence immutability.
  - Accept: Each decision includes alternatives, consequences, owner, and review date.
  - Verify: Architecture and security review sign-off.

- [ ] **2. Define v1 contracts**
  - Build: JSON Schema/OpenAPI definitions for Passport, Finding, Evidence Manifest, Policy Decision, and errors; Protobuf worker envelope.
  - Accept: Contracts include version fields, tenant scope, artifact digest, timestamps, and provenance.
  - Verify: Schema examples validate in CI and breaking-change detection is enabled.

- [ ] **3. Create the threat model**
  - Build: Data-flow diagram, assets, trust boundaries, abuse cases, and mitigations.
  - Accept: Covers malicious artifacts, cross-tenant access, forged evidence, secret leakage, and compromised workers.
  - Verify: Every high-risk threat has a linked control and test owner.

- [ ] **4. Scaffold local and CI environments**
  - Build: Web, API, worker, PostgreSQL, object storage, Temporal, and OPA development services.
  - Accept: One documented command starts a clean local system; CI runs format, lint, unit, contract, and security checks.
  - Verify: Fresh-clone smoke test.

- [ ] **5. Implement tenancy and audit foundation**
  - Build: Organizations, workspaces, actors, tenant context, RLS policies, and append-only audit events.
  - Accept: Every domain write emits an actor- and tenant-bound audit event in the same transaction or via an outbox.
  - Verify: Automated cross-tenant negative test suite.

- [ ] **6. Implement model registration**
  - Build: Asset/model/version persistence, idempotency, digest capture, and registration API.
  - Accept: Same request and digest return the same result; changed digest creates a new immutable version.
  - Verify: API contract and integration tests.

- [ ] **7. Implement Passport lifecycle**
  - Build: State machine, versioning, freshness, denial, expiry, and revocation.
  - Accept: Invalid transitions fail closed and all valid transitions are audited.
  - Verify: Table-driven state-transition tests.

- [ ] **8. Build the dashboard vertical slice**
  - Build: Responsive shell and model Overview page matching the supplied visual hierarchy.
  - Accept: Real API data renders status, risks, tests, findings, provenance, and lineage; every risk links to evidence.
  - Verify: Component tests, accessibility scan, and desktop/tablet/mobile visual snapshots.

- [ ] **9. Enroll and operate a worker**
  - Build: Enrollment, credential exchange, heartbeats, capabilities, job leasing, cancellation, and rotation.
  - Accept: Bootstrap material cannot be reused; credentials are short-lived and tenant-bound.
  - Verify: Expiry, replay, wrong-tenant, restart, and cancellation tests.

- [ ] **10. Persist immutable evidence**
  - Build: Canonical manifest, hashing, signing, object write, metadata transaction, and verification endpoint.
  - Accept: Mutated content, metadata, or signature fails verification.
  - Verify: Tamper-test suite plus object-retention configuration check.

- [ ] **11. Run one real assessment end to end**
  - Build: Temporal workflow, provenance checker, normalized finding generation, and UI progress state.
  - Accept: Retries do not duplicate findings; cancellation and timeout are visible and audited.
  - Verify: End-to-end test from registration through finalized Passport.

- [ ] **12. Enforce one production policy in CI**
  - Build: OPA bundle, policy decision record, CLI `gate`, and CI example.
  - Accept: Good fixture exits `0`, denied fixture exits `1`, infrastructure failure exits `2`.
  - Verify: CI matrix asserts all three outcomes.

## 8. UI implementation notes from the supplied reference

- Preserve the left navigation + model-level tab structure; it scales naturally to the broader control plane.
- Treat the Overview as a composition of domain-backed cards, not one dashboard-specific response blob.
- Keep status language separate from scores: `Verified with warnings` is a policy state; `68` is a measured risk value.
- Make every count and chart segment interactive and traceable to filtered evidence or findings.
- Use semantic color plus icon/text. Red, amber, and green alone must never carry meaning.
- Build skeleton, empty, partial-data, stale-Passport, permission-denied, and worker-offline states alongside the happy path.
- The first polished route is `/models/:modelId/versions/:versionId`; all displayed data belongs to that immutable version.

## 9. Explicitly deferred

- Full SIEM, GRC, IAM, DLP, CSPM, model registry, or observability replacement.
- Real-time production traffic inspection and automated incident containment.
- Dedicated graph database, vector database, OpenSearch, or ClickHouse until measured scale requires one.
- Multiple cloud connectors before the S3 + MLflow path is reliable.
- Multi-region active-active, fully self-hosted packaging, marketplace, and vendor-risk portal.
- Generative attack planning before deterministic checks, evidence integrity, and evaluator calibration are trustworthy.

## 10. Decision checkpoints

1. **End of Week 1:** approve trust boundaries, state machine, and evidence schema.
2. **End of Week 3:** approve tenancy isolation and application shell.
3. **End of Week 5:** approve the real-data Passport Overview vertical slice.
4. **End of Week 7:** approve worker recovery and evidence integrity demonstration.
5. **End of Week 9:** approve assessment precision on the calibration fixtures.
6. **End of Week 11:** approve policy semantics and CI gate behavior.
7. **End of Week 12:** approve design-partner deployment and demo.

## 11. Open decisions before coding beyond the skeleton

- Identity provider for the first design partner: managed OIDC provider versus self-hosted Keycloak.
- First cloud target: AWS is recommended unless a committed design partner requires another cloud.
- Artifact scope for MVP: files and S3/MLflow references are recommended; Hugging Face and cloud registries can follow.
- Evidence signing root: cloud KMS key for the control plane plus worker-scoped keys, with Sigstore integration evaluated after the basic envelope is stable.
- First design partner's concrete gate policy and the single “wow moment” for the demo.

## 12. Current-source checks

The framework decision was checked against current official documentation on 2026-09-23:

- Next.js App Router and framework overview: https://nextjs.org/docs
- Next.js release and security channel: https://nextjs.org/blog
- Go release history: https://go.dev/doc/devel/release
- Temporal Go SDK: https://docs.temporal.io/develop/go
- Connect for Go: https://connectrpc.com/docs/go/getting-started/
- Open Policy Agent: https://www.openpolicyagent.org/docs

