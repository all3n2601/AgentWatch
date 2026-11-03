# 0007: Label mixed contention with every confirmed resource and freeze the test hold-out

Date: November 2, 2026. Status: proposed for D02 review.

The bounded injectors and replay runner (ADR 0006) record agent tasks replayed under `cpu`,
`inference` and `retrieval` contention alone and under `mixed` (`cpu` plus `inference`)
contention. `labels-v1.1` labels one injected resource per run and fails on mixed contention.
Before any model is trained or evaluated, the repository needs a label rule for replayed
steps, a regression target, and a test split that is declared rather than discovered from
results.

**Every confirmed resource is listed.** A step slowed by two contentions is labeled with both
resources in `label_resources` (a sorted, comma-separated list; `""` for none), and
`label_resource` is the class, `none`, or `mixed`. The evidence has one total duration per
step, so splitting it between resources would fabricate attribution; no resource is ranked and
there are no ties. Each resource in a mixed label needs its own evidence: the same step,
replayed under that resource's contention alone, must also have exceeded noise (the same
`max(0.01 s, 3.0 × stdev)` threshold as `labels-v1.1`, strictly exceeded). The labeler
requires exactly one labelable solo replay per mixed member class. Steps that slowed only when
both contentions ran are labeled with both members under the distinct basis `mixed_joint_only`
so evaluation can report them separately; the alternative was rejecting them, which would drop
data because the rule's answer was inconvenient. Steps are rejected only for missing or
invalid evidence, each with its own recorded reason. The target `label_excess_fraction` is the
whole step's excess against its per-step baseline, never divided by resource.

**Step type is not filtered.** The label names the contention that was applied, an operational
proxy. The injectors' spillover (CPU load slows the model server; the retrieval flood slows
every runner thread) is recorded in the data card, and `step_type` stays on each row.

**Rules are fingerprinted.** `labels-v2.toml` holds the rule values; `label_version` is
`labels-v2` plus a fingerprint of the values and the labeling code, pinned in a test. It is
separate from `labels.toml`, so the `labels-v1.1` pin and current releases are untouched.

**The test hold-out is frozen first.** `split-assignment.toml` (`split-v2`) holds out the
injection method `worker_queue_blocker` and the task group `fix-window-count`. Holding out
`cpu-spin-processes-v1` would send every mixed row to test, and the only inference or
retrieval method cannot be held out without removing that class from training. With two agent
tasks, hashing cannot place one in test reliably, so the test task is declared by a rule fixed
before any label was seen (highest `split-v1:<name>` hash position). Task groups drop the
`-r<revision>` suffix so revisions of a task never cross splits, and independence of
`task_group`, `run_id` and `plan_sha256` across splits is checked by
`split_assignment.independence` (via `partition_v2`). D03 wires it into `make_split` and
`load_split` so each split write and read recomputes it.
Changing the file requires `split-v3` and a new test split.

**Scope.** D02 changes `pipelines/data` and documentation only. `make dataset` keeps building
`labels-v1.1` releases with `split-v1`, and `eval-contract-v1` is unchanged. Ingesting replay
experiments into releases (`steps-v1.3`, evidence-store entries, the frozen TFDV schema) is
D03. The replay label columns are declared non-feature in `REPLAY_LABEL_COLUMNS`, outside the
release schema. [The `eval-contract-v2` proposal](../evaluation-contract-v2-proposal.md) asks
Rohit to accept the labels and split; thresholds are unchanged.

Consequences: no replay experiment is in the evidence store, so `labels-v2` is verified on
synthetic test fixtures only and real-replay acceptance is open. With two tasks the
validation split is empty and the test split is one task, so held-out results will have wide
uncertainty. Replay timings under contention remain features, so D03 must check they do not
encode the condition. D03 must also allow the empty `label_resources` string in the release
value checks. Labels are operational proxies and still need validation against natural
overload.
