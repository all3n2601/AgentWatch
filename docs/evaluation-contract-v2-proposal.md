# AgentWatch evaluation contract v2 proposal

Status: proposed, not accepted. Owner: Rohit, who accepts, changes or rejects it.

This proposal comes from D02 ([data card](data-card.md),
[ADR 0007](adr/0007-multi-resource-labels-and-frozen-hold-out.md)). Nothing in
`eval-contract-v1` ([evaluation contract](evaluation-contract.md)) changes until it is
accepted. `pipelines/model/evaluation-contract.toml` is not edited by D02, `make dataset`
keeps building `labels-v1.1` releases with `split-v1`, and every threshold is unchanged.

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

- D03 ingests replay experiments into releases (`steps-v1.3`) with these labels and wires
  `split-v2`; it follows acceptance.
- No recorded replay experiment is in the evidence store, so none of this is verified on real
  replays. Benoy records at least one per task, per the A05 board notes.
- With two agent tasks, validation is empty and tuning data is unavailable; the contract's
  count minimums make the dependent gates `inconclusive` rather than passed.
