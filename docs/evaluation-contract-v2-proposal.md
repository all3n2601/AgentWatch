# AgentWatch evaluation contract v2 proposal

Status: verbal acceptance reported on 2026-11-05; written confirmation pending.
Owner: Rohit (M02). His approval confirming the contract is an explicit merge condition
for PR #59. D03 encoded items 1, 3 and 4 as `eval-contract-v2` in
`pipelines/model/evaluation-contract.toml` (identifier `9b55506c`; see
[the evaluation contract](evaluation-contract.md) and
[ADR 0009](adr/0009-combined-dataset-release-and-split-v2.md)). Item 2 is Rohit's M02 decision,
recorded when he [approved PR #59](https://github.com/all3n2601/AgentWatch/pull/59#pullrequestreview-5846813656) on 2026-11-10 and implemented in code as
`mixed-scoring-v1`: naming any confirmed resource is correct, `mixed_joint_only` steps are
reported separately, and per-resource F1 is reported beside macro F1. `eval-contract-v1` is kept
unchanged as `pipelines/model/evaluation-contract-v1.toml`.

This proposal comes from D02 ([data card](data-card.md),
[ADR 0007](adr/0007-multi-resource-labels-and-frozen-hold-out.md)). The text below is the
proposal as reviewed; the notes after it record what D03 did.

## Proposed changes

1. **Accept `labels-v2`.** Add `mixed` to the label values evaluated. `mixed` is a label value
   only; it is not added to the four prediction classes `none`, `cpu_sandbox`, `inference` and
   `retrieval`.
2. **Add a multi-resource metric.** Report per-resource F1 on `label_resources` next to macro
   F1, and define how a four-class prediction is scored on a `mixed` step (for example,
   whether naming one confirmed member is correct). Rohit decides the scoring rule before any
   evaluation. Steps with basis `mixed_joint_only` are reported separately.
3. **Select `split-v2`.** This is a schema change, not a value edit: add `method` and
   `assignment` keys to `[splits]`, and allow `task_group` as a group key (today `splits.py`
   accepts only `task_id`; a D03 code change). Then set `method = "split-v2"`,
   `group_key = "task_group"` and
   `assignment = "pipelines/data/src/agentwatch_data/split-assignment.toml"`. The file
   holds out injection method `worker_queue_blocker` and task group `fix-window-count`; changing
   it requires `split-v3`. `worker_queue_blocker` is the `injection_method` of every current `labels-v1.1` CPU and
   coding row, so once a release carries both row kinds (D03), `split-v2` sends every CPU and coding
   task group to test, and training contains only the non-held-out replay task rows (today
   `fix-mean-empty`). This changes the rows #45 and #46 train on today, which is why `split-v1`
   stays in force until Rohit accepts this contract.
4. **Leave every threshold unchanged.**

## Dependencies and open items

- Done in D03 (2026-11-06): replay experiments are in the evidence store and in release
  `steps-v1.3+aafae73eb211` with these labels, and `split-v2` is wired into `make_split` and
  `load_split` (split `split-v2+febe942a2b59`). The assignment path is written relative to the
  contract file (`../data/src/agentwatch_data/split-assignment.toml`), which resolves to the
  file named in item 3.
- Done, M02 (Rohit): item 2 as `mixed-scoring-v1`; see [the rule baseline](rule-baseline.md).
- With two agent tasks, validation is empty and tuning data is unavailable; calibration and
  every validation-dependent gate are `inconclusive` rather than passed. The test split meets
  the count minimums (data card, Dataset v1).

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
