# 0009: Build one `steps-v1.3` release with two row kinds and split it with `split-v2`

Date: November 6, 2026. Status: proposed for D03 review. Builds on
[ADR 0006](0006-bounded-injectors-and-agent-replay.md) (replay experiments) and
[ADR 0007](0007-multi-resource-labels-and-frozen-hold-out.md) (`labels-v2`, `split-v2`).

D02 left `labels-v2` and the frozen `split-v2` assignment as library code with no release using
them. On top of his four A05 attempts from 2026-10-25, Benoy then recorded 50 more replay
experiments (54 in total, on two agent tasks), and Rohit accepted
`eval-contract-v2` on 2026-11-05 (told to the owner directly; his approval of the D03 pull
request is the written confirmation). M02 needs one immutable, rebuildable dataset release that
holds the real replays beside the existing CPU and coding runs, with statistics that expose its
defects.

**One release, two row kinds and two rule sets.** `steps-v1.3` keeps every `steps-v1.2` column
name, type and role and adds agent replay rows (`kind = agent_replay`). CPU and coding run rows
keep `labels-v1.1` and the `steps-v1.2` feature definitions; replay rows use `labels-v2` and
`step-features-v1`, joined on `(run_id, step_id)` from each replay's own journal. Each row records
its `label_version` and `feature_version`, so the shared feature columns are labeled by
definition rather than presented as one, and the manifest counts both. The alternative, a
separate replay-only release, would have kept the CPU and coding rows out of the `split-v2`
hold-out that the accepted contract defines, and given M02 two datasets to reconcile. Readings
the replay evidence lacks (workload size, worker capacity, test count, rule output, and process
CPU time for model calls and retrievals) are nullable and stay null, never zero. A step whose
feature timings differ from its label's timings by more than 1e-6 s is rejected, because its
label and features would describe different windows.

**Replay evidence is curated like runs.** `make add-evidence` accepts an experiment directory,
reads it strictly before copying, and stores it byte for byte under a name derived from its own
content (`replay-<task>-<started_at>-<6 hex of experiment.json's SHA-256>`), not its delivery
folder. Experiments that failed protocol checks are kept and rejected with a reason at labeling;
integrity rebuilds replay rows from the store before anything is versioned.

**`split-v2` is wired through the contract, and v1 files are kept.**
`pipelines/model/evaluation-contract.toml` becomes `eval-contract-v2` (`method = "split-v2"`,
`group_key = "task_group"`, the assignment file path, both label rules; every threshold
unchanged). `make_split` and `load_split` call `split_assignment.partition_v2` and record the
assignment's hash and the D02 independence, held-out and mixed-in-test checks; `load_split`
rebuilds and compares on every read, as for `split-v1`. `eval-contract-v1` is kept unchanged as
`evaluation-contract-v1.toml` and `steps.pbtxt` stays beside the new
`steps-v1.3.pbtxt`, so the `steps-v1.2` split and the #45/#46 results still rebuild when those
paths are passed. Editing v1 in place was rejected: a contract identifier hashes content, and a
changed v1 would silently orphan existing splits.

**Leakage counts membership; the mixed minimum counts `mixed` labels.** The one-vs-rest leakage
check treats a step as a positive for every class in its `label_resources`. Counting a
`cpu_sandbox,inference` step as a negative for its own resources would add label noise and lower
the AUCs, making a leak harder to see; for CPU and coding rows membership equals
`label_resource`, so `split-v1` results do not change. The contract's mixed test minimum (30)
counts steps labeled `mixed` (confirmed slowed by two or more resources), the conservative
reading; rows under the `mixed` condition are recorded beside it. A report-only design-encoding
check compares contended and clean-control replay steps labeled `none`; it is not a gate,
because live telemetry legitimately sees injected load on steps that were not slowed.

**`mixed` stays unscored.** Item 2 of the proposal (the multi-resource metric and how a
four-class prediction is scored on a `mixed` step) is Rohit's M02 decision and is not encoded.
Until then the rule baseline and training refuse a split holding `mixed` labels with a reason
naming it, instead of scoring them under an invented rule.

Consequences: release `steps-v1.3+aafae73eb211` (5,953 rows: 5,603 replay, 350 CPU and coding)
passed the gate and split `split-v2+febe942a2b59` met class balance and passed leakage, at an
observed maximum AUC of 0.974242 against the 0.98 bound, so timing features nearly separate some
classes. The frozen assignment leaves validation empty and training one task group, so
calibration and validation-dependent gates are inconclusive, and no rule or model result exists
on v1 until M02 defines `mixed` scoring. Replay telemetry is mostly null (427 of 5,603 replay
rows have readings), and `step-features-v1` reads no GPU, inference-server or retrieval metric;
that is a new feature version. The gate compares drift only against a release validated with
byte-identical frozen-schema bytes, so drift against `steps-v1.2` is unavailable in v1, although
the design proposed it; cross-schema drift is follow-up work. The label fingerprints moved
(`labels-v1.1+2c495a72`, `labels-v2+d803a9c6`) because hashed source changed, while the rule
versions stayed because no produced label or eligibility changed.

### D03 acceptance and evaluation merge conditions (2026-11-10)

PR #59 must not merge until Rohit (M02/promotion-check owner) supplies written
confirmation of `eval-contract-v2` and the final revision has the required two
non-author approvals. The owner's report of verbal acceptance is not that record.
The frozen contract's `status = "accepted"` and hash `9b55506c` are preserved to
avoid silently changing published split identities; this metadata does not prove
that the written merge condition has been satisfied. That condition remains open.

The test split crosses feature definitions: 350 CPU/coding rows use `steps-v1.2`,
while replay rows use `step-features-v1`. Model and rule evaluation now refuse a
combined population with differing feature versions (including present versus
missing identity), even after M02 defines mixed-label scoring. A future versioned
split/report design must evaluate compatible populations separately. Model and
serving metadata retain the full label-version set beside the legacy singular field.
