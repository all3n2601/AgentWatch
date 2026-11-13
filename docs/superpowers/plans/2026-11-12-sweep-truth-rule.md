# sweep-truth-v1 Label Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record each held-out capacity-sweep step's measurements in `services/agent`, and judge
them with the versioned rule `sweep-truth-v1` in `pipelines/data`.

**Architecture:** `capacity-sweep-steps` (agent) reads a frozen sweep folder and writes
`steps.jsonl` and `steps.json` with no judgement. `agentwatch_data.sweep_truth` validates that
input, applies the frozen `sweep-truth-v1.toml`, and writes `truth.jsonl` and `truth.json` with a
fingerprinted `label_version`. A contract test in `tests/contract` runs both on the frozen A06
sweeps when they are pulled.

**Tech Stack:** Python 3.13, uv workspace, pytest, Ruff, `tomllib`.

**Spec:** [docs/superpowers/specs/2026-11-12-sweep-truth-rule-design.md](../specs/2026-11-12-sweep-truth-rule-design.md)

## Global Constraints

- Package code in `pipelines/data` never imports `agentwatch_agent`, and `services/agent` never
  imports `agentwatch_data`. Only `tests/contract` may import both.
- Nothing is written under any `data/held-out/` folder; both commands refuse such an output.
- Missing readings stay null with a reason; nothing is recorded as a measured zero.
- Every test fixture is synthetic and its module docstring says so, except the frozen-sweep
  contract test, which reads `data/held-out/a06-20261111/` and skips when it is absent.
- Accepted protocol hashes, in full: cfg-default
  `22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9`, cfg-hardware
  `1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd`.
- Noise threshold max(0.01 s, 3.0 × baseline stdev); minimum baseline runs 3.
- Commit titles `<type>: <specific change>`; no co-author trailer or tool footer.
- Run tests in WSL: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash /mnt/c/Users/benoy/AppData/Local/Temp/claude/e25ce4ecbc9d655e/7bd33e96-ab25-4ed3-a459-6ae75dfe51ef/scratchpad/wsl-uv.sh pytest <paths> -q`
  (Windows-native runs fail unrelated tests with PermissionError).
- Ruff: `... wsl-uv.sh ruff check <paths>` and `ruff format --check <paths>`.

## Review Focus

1. A `steps.jsonl` with Windows line ends or a trailing blank line: read as the same records
   (Task 2, `test_crlf_and_trailing_newline_read_the_same`).
2. A `steps.json` whose `steps` count is right but a line is repeated: refused as a repeat, not
   accepted twice (Task 2, `test_repeated_run_and_step_is_refused`).
3. A baseline with `stdev_seconds` null (count 1 would not reach the recorder, but a hand-made
   file can): threshold falls back to the floor (Task 3, `test_null_stdev_uses_the_floor`).
4. A root step that is also `model_call` type (hand-made input): `root_step` wins, so no root is
   ever scored (Task 3, `test_root_step_wins_over_model_call_type`).
5. An `--output` given as a relative path that resolves into `data/held-out/` through `..`:
   refused (Task 1 and Task 4, `test_output_resolving_into_held_out_is_refused`).

## Ruling recorded with this plan (2026-11-12)

The spec says the scoring branch's refactor commits `83be357` and `807285c` come across. They
are not needed: the recorder reads runs through `sweep_labels.read_run` (today's `_read_run`,
renamed public) and `baselines`, which exist on `main`. Leaving them out keeps `sweep_labels`'
judgement code untouched in this PR. The spec gets a dated note saying so (Task 1, Step 1).

## File map

| File | Task | Responsibility |
|---|---|---|
| `services/agent/src/agentwatch_agent/sweep_steps.py` | 1 | Recorder: read a held-out sweep, write `steps.jsonl`/`steps.json` |
| `services/agent/src/agentwatch_agent/sweep_labels.py` | 1 | Rename `_read_run` → `read_run` (3 call sites) |
| `services/agent/src/agentwatch_agent/cli.py` | 1 | `capacity-sweep-steps` subcommand |
| `services/agent/tests/test_sweep_steps.py` | 1 | Recorder tests |
| `pipelines/data/src/agentwatch_data/sweep_steps_input.py` | 2 | Read and validate recorder output |
| `pipelines/data/tests/test_sweep_steps_input.py` | 2 | Input validation tests |
| `pipelines/data/src/agentwatch_data/sweep-truth-v1.toml` | 3 | Frozen rule values |
| `pipelines/data/src/agentwatch_data/sweep_truth.py` | 3, 4 | Rule config, fingerprint, decisions, writer, CLI |
| `pipelines/data/tests/test_sweep_truth.py` | 3, 4 | Rule and writer tests |
| `tests/contract/test_sweep_truth_frozen.py` | 5 | Recorder → rule on the frozen sweeps |
| `Makefile`, `services/agent/README.md`, `docs/data-card.md`, `docs/project-checklist.md`, `services/agent/src/agentwatch_agent/sweep_protocol.py` (comment) | 1, 4, 5 | Targets and docs |

---

### Task 1: Recorder `capacity-sweep-steps`

**Files:**
- Create: `services/agent/src/agentwatch_agent/sweep_steps.py`, `services/agent/tests/test_sweep_steps.py`
- Modify: `services/agent/src/agentwatch_agent/sweep_labels.py` (rename `_read_run`),
  `services/agent/src/agentwatch_agent/cli.py`, `Makefile`, `services/agent/README.md`,
  `services/agent/src/agentwatch_agent/sweep_protocol.py` (comment beside
  `LABEL_THRESHOLD_FLOOR_SECONDS`/`LABEL_THRESHOLD_STDEV_MULTIPLIER`: "must equal
  `pipelines/data/src/agentwatch_data/sweep-truth-v1.toml` `[noise]`"),
  `docs/superpowers/specs/2026-11-12-sweep-truth-rule-design.md` (dated note from the ruling)

**Interfaces:**
- Consumes (on `main`): `sweep_labels.read_run(run_dir) -> (plan | None, timings, spans)`,
  `sweep_labels.baselines(n1_runs) -> Baselines` (values carry `count`, `mean_seconds`,
  `stdev_seconds`), `sweep_labels.ramp_run_dirs`, `sweep_labels.BASELINE_LEVEL`,
  `sweep_analysis._resolve_run_dir`, `sweep_analysis.split_ramp_entries`,
  `sweep_offline.SweepFolderError`, `_read_record`, `_is_int`, `sweep_protocol.PROTOCOL_VERSION`,
  `protocol_sha256`, `coverage.read_journal`, `runner.JOURNAL_FILE`, `sweep.SWEEP_FILE`.
- Produces: in `sweep_steps.py`, `STEPS_FORMAT = "sweep-steps-v1"`, `STEPS_FILE = "steps.jsonl"`,
  `SUMMARY_FILE = "steps.json"`, `ACCEPTED_PROTOCOL_SHA256: dict[str, str]`,
  `RECORD_KEYS: frozenset[str]`, `SPAN_STATUSES: tuple[str, ...]`,
  `sweep_steps(sweep_dir: Path) -> tuple[dict, list[dict]]`,
  `write_steps(sweep_dir: Path, output_dir: Path) -> Path`; CLI
  `python -m agentwatch_agent capacity-sweep-steps <sweep_dir> [--output DIR]`, default output
  `.data/a06-scoring/steps/<sweep folder name>`; Make `agent-capacity-sweep-steps SWEEP=… [OUTPUT=…]`.

Port the scoring branch's `sweep_truth.py`
(`git show feature/a06-held-out-scoring:services/agent/src/agentwatch_agent/sweep_truth.py`) and
its tests (`…:services/agent/tests/test_sweep_truth.py`, which reuse `test_sweep.Harness` and
`test_sweep_labels.span_call`/`write_run`): keep `_check_protocol`, `_scored_entry_problem`,
`_scored_entries`, `_inside_held_out`, `_write_lf` and the read-everything-then-write order;
drop `_truths_of`, `_tally`, `StepTruth` and every threshold. Records are built from
`read_run` + `baselines` directly.

`RECORD_KEYS` = `run_id, run_folder, step_id, step_type, parent_step_id, plan_sha256, task_id,
configuration, n, phase, batch_sequence, batch_id, state, duration_seconds, queue_seconds,
execution_seconds, span_status, vllm_queue_seconds, vllm_service_seconds, baseline`.
`SPAN_STATUSES` = `("measured", "unavailable", "malformed", "incomplete", "not_recorded")`.
`parent_step_id` comes from the step's latest `step_event` in the journal (`read_journal`), not
from `step_timings`, whose output is recorded by replay experiments and must not change.
`baseline` is `{"count", "mean_seconds", "stdev_seconds"}` from `baselines(...)[(plan, step_id)]`
or null. `vllm_*_seconds` are null unless `span_status == "measured"`. Summary keys: `format`,
`sweep_folder`, `sweep_id`, `configuration`, `protocol_version`, `protocol_sha256`,
`sweep_json_sha256`, `held_out`, `steps`, `excluded_batches`, `runs_without_evidence`,
`runs_without_plan`.

- [ ] **Step 1: Add the ruling note to the spec** under "Recorder": "Changed 2026-11-12 (plan):
  the refactor commits are not carried; the recorder reads runs through `read_run` and
  `baselines` on `main`."
- [ ] **Step 2: Write the failing tests** in `test_sweep_steps.py` (docstring: synthetic sweeps
  from the stub driver, not recorded evidence):
  - `test_lines_carry_exactly_the_record_keys`: every line's keys == `RECORD_KEYS`.
  - `test_root_step_has_null_parent_and_children_name_it`: the `agent-run` line has
    `parent_step_id is None`; the model step's is `"agent-run"`.
  - `test_measured_span_keeps_vllm_seconds`: `span_status == "measured"`,
    `vllm_queue_seconds == 2.0` for a call written with `queue=2.0`.
  - `test_unmeasured_span_has_null_vllm_seconds`: a call with `status="unavailable"` →
    `span_status == "unavailable"`, both vLLM fields None; a tool step → `"not_recorded"`.
  - `test_baseline_is_count_mean_stdev_or_null`: N = 1 durations (0.9, 1.0, 1.1) give
    `{"count": 3, "mean_seconds": 1.0, "stdev_seconds": pytest.approx(0.1)}`; a step id absent
    from the baseline runs gives `None`.
  - `test_no_threshold_or_truth_fields`: no key contains `threshold`, `truth`, `slower`, `queued`.
  - `test_warmup_is_skipped_and_live_runs_count_as_without_plan`.
  - `test_summary_fields`: keys as listed, `format == "sweep-steps-v1"`, `held_out is True`,
    `steps == number of lines`, `sweep_json_sha256` == SHA-256 of the file bytes.
  - Refusals, each asserting `SweepFolderError` and that the output folder does not exist after
    `write_steps`: other protocol version, unknown hash, recorded hash not matching the protocol,
    `held_out` false, malformed scored batch entry, batch folder with `..`, unreadable run.
  - `test_output_inside_held_out_is_refused` and
    `test_output_resolving_into_held_out_is_refused` (`tmp/x/../data/held-out/out`).
  - `test_cli_writes_default_folder_and_exits_2_on_refusal` (in `test_cli.py` style).
- [ ] **Step 3: Run** `… wsl-uv.sh pytest services/agent/tests/test_sweep_steps.py -q` — FAIL
  (module missing).
- [ ] **Step 4: Implement** the rename, `sweep_steps.py`, the CLI subcommand (mirroring the
  scoring branch's `capacity_sweep_truth_command`: exit 2 with `error: …` on
  `SweepFolderError`/`OSError`, print the two written paths on success), the Make target with a
  usage line, the README section, and the `sweep_protocol.py` comment.
- [ ] **Step 5: Run** `… wsl-uv.sh pytest services/agent/tests -q` — all pass, including the
  unmodified `test_sweep_labels.py`; `ruff check` and `ruff format --check` on touched files clean.
- [ ] **Step 6: Commit** `feature: record held-out capacity-sweep step measurements for labeling`.

### Task 2: Read and validate recorder output (`pipelines/data`)

**Files:**
- Create: `pipelines/data/src/agentwatch_data/sweep_steps_input.py`,
  `pipelines/data/tests/test_sweep_steps_input.py`

**Interfaces:**
- Consumes: `agentwatch_data.labels.LabelingError`, `agentwatch_data.files.file_sha256`.
- Produces: `STEPS_FORMAT = "sweep-steps-v1"`, `STEPS_FILE = "steps.jsonl"`,
  `SUMMARY_FILE = "steps.json"`, `RECORD_KEYS: frozenset[str]` and
  `SPAN_STATUSES: tuple[str, ...]` (re-declared, equal to Task 1's; comment names the agent
  module), `PHASES = ("ramp", "live", "injected", "control")`,
  `class SweepStepsError(LabelingError)`,
  `@dataclass(frozen=True) class SweepSteps: summary: dict; records: tuple[dict, ...];
  steps_sha256: str; summary_sha256: str`,
  `read_steps(directory: Path, accepted_protocols: dict[str, str]) -> SweepSteps`.

`read_steps` checks everything in the spec's "Validation" list except the output folder and
raises `SweepStepsError` naming the file and, for a line, its 1-based number and field. Lines are
read with `splitlines()` after UTF-8 decoding; empty lines are ignored. Seconds fields:
`duration_seconds`, `queue_seconds`, `execution_seconds`, `vllm_queue_seconds`,
`vllm_service_seconds`, and the baseline's `mean_seconds` and `stdev_seconds` (`bool` is not a
number). `baseline.count` is an integer ≥ 1. `parent_step_id`, `plan_sha256`, `task_id` are a
string or null; `step_id`, `step_type`, `state`, `run_folder`, `batch_id`, `configuration` are
non-empty strings; `run_folder` is relative with no `..`.

- [ ] **Step 1: Write the failing tests** (docstring: synthetic recorder output):
  a `valid_line(**changes)` helper and a `write_input(tmp_path, lines, **summary_changes)` helper;
  `test_valid_input_reads_back` (records equal, hashes equal `file_sha256`);
  `test_crlf_and_trailing_newline_read_the_same`; one parametrized
  `test_summary_is_refused` over: wrong `format`, `held_out` false or missing, protocol version
  other than `natural-overload-sweep-v1`, hash not in `accepted_protocols`, hash accepted for a
  different `configuration`, `steps` ≠ line count; one parametrized `test_line_is_refused` over:
  missing key, extra key, `run_id` not a UUID, `n` 0, `n` True, `batch_sequence` −1, phase
  `warmup`, `duration_seconds` −0.1, NaN, `"1.0"`, `span_status` `"reported"`, measured
  values with `span_status` `"unavailable"`, `baseline` without `stdev_seconds`, `baseline.count`
  0, `run_folder` `"../x"`; `test_repeated_run_and_step_is_refused`;
  `test_missing_summary_is_refused`; each refusal asserts the message names the line number
  and field where one applies.
- [ ] **Step 2: Run** `… pytest pipelines/data/tests/test_sweep_steps_input.py -q` — FAIL.
- [ ] **Step 3: Implement** `sweep_steps_input.py`.
- [ ] **Step 4: Run** the same — PASS; Ruff clean.
- [ ] **Step 5: Commit** `feature: validate capacity-sweep step records before labeling`.

### Task 3: Rule values, identifier and decisions

**Files:**
- Create: `pipelines/data/src/agentwatch_data/sweep-truth-v1.toml`,
  `pipelines/data/src/agentwatch_data/sweep_truth.py`, `pipelines/data/tests/test_sweep_truth.py`

**Interfaces:**
- Consumes: `canonical_sha256` (`agentwatch_data.extract`), `require_keys`, `require_number`,
  `require_text`, `LabelingError` (`agentwatch_data.labels`).
- Produces: `TRUTH_NAMES = ("queue", "slower", "protocol_clean")`,
  `@dataclass(frozen=True) class SweepTruthRule` with `version: str`,
  `stdev_multiplier: float`, `min_seconds: float`, `min_baseline_runs: int`,
  `scored_step_types: tuple[str, ...]`, `clean_phases: tuple[str, ...]`, `clean_level: int`,
  `baseline_phase: str`, `baseline_level: int`, `protocols: dict[str, str]`,
  `fingerprint: str`, property `label_version -> f"{version}+{fingerprint[:8]}"`;
  `load_truth_rule(path: Path | None = None) -> SweepTruthRule`;
  `noise_threshold(baseline: dict, rule: SweepTruthRule) -> float`;
  `judge(record: dict, rule: SweepTruthRule) -> dict` returning `{"truths": {...},
  "truth_reasons": {...}, "noise_threshold_seconds": float | None, "excess_seconds": float | None}`.

The `.toml` holds exactly the spec's table (`version`, `[noise]`, `[baseline] min_runs`,
`[steps] scored_types`, `[protocol_clean] phases, level`, `[slower] baseline_phase,
baseline_level`, `[protocols] cfg-default = …, cfg-hardware = …`) with a header comment matching
`labels-v2.toml`'s ("frozen before any evaluation; label_version fingerprints these values and the
rule's code") and the `[noise]` comment that `sweep_protocol.py` must agree. The fingerprint is
`canonical_sha256({"rules": validated_values, "logic": {name: source}})` over
`sweep_truth.py` and `sweep_steps_input.py`, read with `files("agentwatch_data")` and
`\r\n` → `\n`, as `replay_labels.logic_source` does. `judge` follows the spec's truth table and
null-reason order exactly; `noise_threshold_seconds`/`excess_seconds` are set whenever a usable
baseline exists for a scored step, else None.

- [ ] **Step 1: Write the failing tests** (docstring: synthetic records):
  - `test_packaged_rule_values`: every field equals the spec's table;
    `label_version` matches `^sweep-truth-v1\+[0-9a-f]{8}$`.
  - `test_comment_does_not_change_the_identifier` and
    `test_a_value_changes_the_identifier` (`min_seconds = 0.02`), on copies in `tmp_path`.
  - `test_malformed_rule_is_refused` parametrized: missing `[noise]`, `stdev_multiplier = 0`,
    `scored_types = []`, a protocol hash that is not 64 hex.
  - Truth table: `test_model_call_queue_above_and_at_threshold` (queue 2.0 → `inference`;
    queue exactly at threshold 0.3 → `none`; baseline stdev 0.1);
    `test_model_call_slower_above_and_below`; `test_slower_unset_on_ramp_n1` (reason
    `not_applicable`, `queue` still judged); `test_slower_judged_on_control_n1`;
    `test_protocol_clean_only_in_clean_batches` (ramp N=1 and control N=1 → `none`; injected
    N=1, ramp N=2, live → null `not_applicable`); `test_protocol_clean_ignores_state`
    (state `failed` in a clean batch → `none`).
  - Exclusions: `test_root_step_is_root_step_everywhere`;
    `test_tool_retrieval_other_are_no_queue_reference`;
    `test_root_step_wins_over_model_call_type`.
  - Reasons in order: `test_not_completed_before_malformed_before_no_baseline`;
    `test_completed_without_duration_is_not_completed`; `test_baseline_below_min_runs_is_no_baseline`
    (count 2); `test_unmeasured_span_leaves_queue_unset_but_judges_slower`;
    `test_null_stdev_uses_the_floor` (threshold == 0.01).
  - `test_a_truth_is_a_class_or_a_reason_never_both` over every case above.
- [ ] **Step 2: Run** `… pytest pipelines/data/tests/test_sweep_truth.py -q` — FAIL.
- [ ] **Step 3: Implement** the `.toml`, `load_truth_rule`, `noise_threshold`, `judge`.
- [ ] **Step 4: Run** — PASS; Ruff clean.
- [ ] **Step 5: Commit** `feature: add the sweep-truth-v1 label rule for held-out capacity sweeps`.

### Task 4: Writer, CLI and Make target

**Files:**
- Modify: `pipelines/data/src/agentwatch_data/sweep_truth.py`, `pipelines/data/tests/test_sweep_truth.py`, `Makefile`

**Interfaces:**
- Consumes: Task 2 `read_steps`, `SweepSteps`; Task 3 `load_truth_rule`, `judge`, `TRUTH_NAMES`.
- Produces: `TRUTH_FILE = "truth.jsonl"`, `TRUTH_SUMMARY_FILE = "truth.json"`,
  `sweep_truths(steps: SweepSteps, rule: SweepTruthRule) -> tuple[dict, list[dict]]`,
  `write_truths(steps_dir: Path, output_dir: Path, rule: SweepTruthRule | None = None) -> Path`,
  `main(argv: list[str] | None = None) -> int` (`python -m agentwatch_data.sweep_truth <steps_dir>
  [--output DIR] [--rules TOML]`, default output `.data/a06-scoring/truth/<steps dir name>`; exit 0,
  or 2 with `error: …` on `LabelingError`/`OSError`); Make `sweep-truth STEPS=… [OUTPUT=…]` via
  `$(DATA).sweep_truth`.

Each `truth.jsonl` record: the input record minus `span_status`, `vllm_service_seconds` and
`baseline`, plus `baseline_mean_seconds`, `baseline_stdev_seconds`, `baseline_count` and
`judge`'s four keys. `truth.json`: `label_version`, the input summary's `sweep_folder`,
`sweep_id`, `configuration`, `protocol_version`, `protocol_sha256`, `sweep_json_sha256`,
`excluded_batches`, `runs_without_evidence`, `runs_without_plan`, plus `steps_sha256`,
`steps_summary_sha256`, `held_out: true`, `steps`, and `counts` = per truth
`{"classes": {...}, "reasons": {...}}`. JSON with `sort_keys=True`, `allow_nan=False`, LF, via a
temporary file and `os.replace`; summary last. The `data/held-out` check resolves the path first
(same parts test as Task 1).

- [ ] **Step 1: Write the failing tests:** `test_records_and_summary_fields`;
  `test_counts_match_records`; `test_same_input_gives_identical_bytes` (two runs, compare bytes);
  `test_refused_input_writes_nothing` (bad summary → output folder absent);
  `test_output_inside_held_out_is_refused`; `test_output_resolving_into_held_out_is_refused`;
  `test_main_exit_codes` (0 on success; 2 on a refused input, on a malformed `--rules`, and on a
  held-out output).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** the writer, `main`, `if __name__ == "__main__"`, and the Make target
  (usage line on missing `STEPS`, exit 2; add `sweep-truth` to `.PHONY`).
- [ ] **Step 4: Run** `… pytest pipelines/data/tests -q` — PASS; Ruff clean.
- [ ] **Step 5: Commit** `feature: write sweep-truth-v1 truths for recorded sweep steps`.

### Task 5: Frozen-sweep contract test, real run and docs

**Files:**
- Create: `tests/contract/test_sweep_truth_frozen.py`
- Modify: `docs/data-card.md`, `docs/project-checklist.md`

**Interfaces:**
- Consumes: Task 1 `write_steps`, `RECORD_KEYS`, `SPAN_STATUSES`, `ACCEPTED_PROTOCOL_SHA256`;
  Task 2's re-declared constants; Task 3 `load_truth_rule`; Task 4 `write_truths`.

- [ ] **Step 1: Write the tests:**
  - `test_recorder_and_rule_agree_on_the_format` (always runs): Task 1's and Task 2's
    `RECORD_KEYS`, `SPAN_STATUSES` and format name are equal, and
    `load_truth_rule().protocols == ACCEPTED_PROTOCOL_SHA256`.
  - `test_frozen_sweeps_keep_their_model_call_truths`, parametrized over the two sweep folders,
    skipped when `data/held-out/a06-20261111/capacity-sweeps/<folder>/sweep.json` is absent:
    record into `tmp_path`, judge into `tmp_path`, then assert counts. Model calls (by
    `step_type == "model_call"`) and all steps:

    | | cfg-default-20261111T211237Z-2d75ea | cfg-hardware-20261111T213039Z-b4ad4a |
    |---|---|---|
    | steps / model calls | 1,896 / 825 | 2,104 / 915 |
    | `queue` inference / none | 686 / 139 | 8 / 907 |
    | `slower` inference / none / null `not_applicable` | 715 / 80 / 30 | 663 / 222 / 30 |
    | `protocol_clean` none / null `not_applicable` | 45 / 780 | 30 / 885 |
    | `root_step` (each truth) | 165 | 183 |
    | `no_queue_reference` (each truth) | 906 | 1,006 |

    The model-call rows are attempt `20261112T025531Z-220363`'s; the last two are the spec's
    change (root = the `other` steps; tool + retrieval).
- [ ] **Step 2: Run** `… pytest tests/contract/test_sweep_truth_frozen.py -q` with the data
  present — PASS (2 frozen cases run, not skipped). A mismatch is a defect to diagnose, never a
  number to update.
- [ ] **Step 3: Real run** from a clean tree: `make agent-capacity-sweep-steps SWEEP=…` then
  `make sweep-truth STEPS=…` for both sweeps (in WSL); note `label_version` and the counts per
  truth by class and reason from each `truth.json`.
- [ ] **Step 4: Docs.** `docs/data-card.md`: a `sweep-truth-v1` section (what it labels, the
  truth table, the exclusions and why, `label_version`, the two sweeps' counts, that it never
  feeds `make dataset` or a release, Akshay as reviewer). `docs/project-checklist.md`: a dated
  2026-11-12 entry under "Compare models and rules on the same held-out tasks and contention
  cases" naming this branch, the label version and the contract test; the item stays open.
- [ ] **Step 5: Full verification:** start Docker and the PostgreSQL test database, run
  `… wsl-uv.sh ruff check .` and `… wsl-uv.sh pytest -q`, stop Docker. Record pass/skip counts.
- [ ] **Step 6: Commit** `docs: record sweep-truth-v1 on the frozen A06 sweeps`.
