# Natural-overload capacity sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `capacity-sweep` command that grows concurrent agent replays on the GPU laptop under
a declared protocol, records throughput, variability, measured contention, failures and
cleanup, decides each vLLM configuration's capacity, validates the experiment labels against
vLLM's measured queue time, and records a held-out prefill-flood condition.

**Architecture:** `sweep_protocol.py` declares `natural-overload-sweep-v1` as frozen values
and a hash. `vllm_server.py` reads vLLM's state (`/metrics`, `/v1/models`) and checks it
against the declared configuration. `sweep.py` drives phases and batches through the existing
`AgentBatch`, `ReplayRunner` and `InjectionSet`, writing `sweep.json` after every batch.
`sweep_analysis.py` computes level metrics, the capacity decision and label validation from
recorded evidence only, so it reruns on any sweep folder without the GPU.

**Tech Stack:** Python 3.13, standard library, the agent and injector packages, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-10-natural-overload-capacity-sweep-design.md`
(sections "Declared protocol", "PR 2", "Analysis", "Errors and exit codes").

**Branch:** `feature/natural-overload-capacity-sweep`. Tasks 1–5 use only activity dicts
and a fake injector, so they can be written before the prefill-flood PR merges. Task 6 imports
`PrefillFloodInjector`: merge `origin/main` into this branch once that PR merges, before
Task 6.

## Global Constraints

- Protocol `natural-overload-sweep-v1`; any changed value is a new version.
- vLLM v0.32.0, `Qwen/Qwen3-1.7B@70d244cc86cc`, `max_model_len = 4096`, prefix caching on.
- `cfg-default`: `max_num_seqs = 2`, `gpu_memory_utilization = 0.6`.
  `cfg-hardware`: `max_num_seqs = 16`, `gpu_memory_utilization` in `(0.8, 0.75, 0.7)`, the
  first that starts, every attempt recorded.
- Ramp `(1, 2, 3, 4, 6, 8, 12, 16)`; 1 warm-up batch; 3 batches per level; N = 1 runs 3
  batches per source (6).
- Idle wait ≤ 30 s for 0 running and 0 waiting, then a fixed 10 s pause.
- Capacity criteria: throughput gain ≥ 0.10 over the previous level; pass rate ≥ N = 1 pass
  rate − 0.20; p90 run duration ≤ 0.5 × run limit (900 s → 450 s); no hard stop.
- Live confirmation: 3 batches at capacity N. Injected: N_inj = max(1, ⌊capacity / 2⌋), 3
  flood batches alternating with 3 clean controls, flood first; `PrefillFloodInjector`
  `concurrency=2`, `prompt_tokens=2048`, ramp 2 s, bound = run limit + 32 s.
- Label rule: contended when (duration − baseline mean) > max(0.01 s, 3 × baseline stdev);
  reference: measured vLLM queue > the same threshold. Verdict "validated" when precision ≥
  0.90 and recall ≥ 0.90.
- `--no-policy` for every batch. Ticks only from one clock per batch; vLLM span durations are
  never compared with runner ticks or `observed_at`.
- Missing, unavailable or malformed readings stay distinct from zero everywhere.
- Natural-overload and injected runs are held-out test evidence; nothing in this PR writes
  them to a dataset release.
- Commit titles `<type>: <specific change>`; no Claude co-author trailer or footer.

## Review Focus

- **vLLM stops answering `/metrics` between batches** (container crashed, port gone): a hard
  stop recorded as `server_state_unavailable`, never a crash and never a silent continue.
  Test: `test_unreadable_server_state_is_a_hard_stop` (Task 5).
- **Every run of a batch is `not_started`** (live API down): the sweep stops with
  `stopped_run_not_started`, and the batch is never counted as zero throughput. Test:
  `test_a_batch_whose_runs_never_started_stops_the_sweep` (Task 5).
- **A level with no passed run, or N = 1 with pass rate 0**: throughput is 0.0 (measured), the
  gain over a zero previous level is undefined and fails the criterion explicitly; no division
  error. Test: `test_zero_throughput_levels_fail_without_dividing_by_zero` (Task 3).
- **A batch folder whose `batch.json` or run evidence is missing or truncated** when the
  analysis reruns: the batch is reported `unreadable` and its level fails the decision; it is
  never read as zeros. Test: `test_unreadable_batches_are_reported_not_zeroed` (Task 3).
- **A model call whose vLLM span is `unavailable` or `malformed`**: excluded from label
  validation and counted by status; never queue 0. Test:
  `test_steps_without_a_measured_span_are_excluded_and_counted` (Task 4).

---

### Task 1: Prefill activity check in the experiment's `_sustained`

**Files:**
- Modify: `services/agent/src/agentwatch_agent/experiment.py` (`_sustained`)
- Test: `services/agent/tests/test_experiment.py`

**Interfaces:**
- Consumes: the prefill injector's activity keys (`report`, `stopped_by_backstop`,
  `requests_failed`, `requests_completed`).
- Produces: `injection_active(log)` returns True for a set whose prefill injection was
  stopped as `requested`, reported, not stopped by backstop, with 0 failures and ≥ 1
  completion.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_sustained_prefill_flood_is_active():
    log = {"injections": [{"injector": "prefill", "stop_reason": "requested", "activity": {
        "report": "received", "stopped_by_backstop": False,
        "requests_failed": 0, "requests_completed": 3}}]}
    assert injection_active(log) is True

@pytest.mark.parametrize("change", [{"requests_completed": 0}, {"requests_failed": 1},
    {"report": "missing"}, {"stopped_by_backstop": True}])
def test_a_prefill_flood_with_gaps_is_not_active(change): ...  # injection_active(...) is False
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest services/agent/tests/test_experiment.py -k prefill -q`
Expected: the first FAILS (unknown kind returns False).

- [ ] **Step 3: Add the `kind == "prefill"` branch** — completions only (abandoned in-flight
prefill requests do not count: with `max_tokens=1` a request that never finished never
proved load).

- [ ] **Step 4: Run the experiment tests** — `uv run pytest services/agent/tests/test_experiment.py -q`; all pass.

- [ ] **Step 5: Commit** — `feature: accept a sustained prefill flood as active contention`

### Task 2: Declared protocol `natural-overload-sweep-v1`

**Files:**
- Create: `services/agent/src/agentwatch_agent/sweep_protocol.py`
- Test: `services/agent/tests/test_sweep_protocol.py`

**Interfaces:**
- Produces:
  - `PROTOCOL_VERSION = "natural-overload-sweep-v1"`, `RAMP_LEVELS`, `BATCHES_PER_LEVEL = 3`,
    `BASELINE_BATCHES_PER_SOURCE = 3`, `WARMUP_BATCHES = 1`, `IDLE_WAIT_SECONDS = 30.0`,
    `PACING_SECONDS = 10.0`, `LIVE_CONFIRMATION_BATCHES = 3`, `INJECTED_PAIRS = 3`,
    and the criteria, injector and label values from Global Constraints as named constants.
  - `@dataclass(frozen=True) class VllmConfiguration(name: str, max_num_seqs: int,
    gpu_memory_utilization: tuple[float, ...], max_model_len: int = 4096,
    enable_prefix_caching: bool = True)`.
  - `CONFIGURATIONS: dict[str, VllmConfiguration]` with `cfg-default` and `cfg-hardware`.
  - `@dataclass(frozen=True) class Batch(phase: str, level: int, repeat: int,
    sources: tuple[int, ...], injected: bool)` — `phase` in `("warmup", "ramp", "live",
    "injected", "control")`; `sources[i]` is the source index for run i.
  - `ramp_batches(level: int) -> list[Batch]` — N = 1: six batches, sources
    `(0,), (1,), (0,), (1,), (0,), (1,)`; N ≥ 2: three batches where run i uses source
    `(i + repeat) % 2`.
  - `injected_level(capacity: int) -> int` = `max(1, capacity // 2)`.
  - `injected_batches(level: int) -> list[Batch]` — flood, control, flood, control, flood,
    control.
  - `protocol(configuration: VllmConfiguration, run_timeout_seconds: float) -> dict` and
    `protocol_sha256(document: dict) -> str` (canonical JSON, reuse
    `replay.canonical_sha256`).

- [ ] **Step 1: Write the failing tests**

```python
def test_one_agent_runs_three_batches_per_source():
    assert [b.sources for b in ramp_batches(1)] == [(0,), (1,), (0,), (1,), (0,), (1,)]

def test_sources_alternate_and_rotate_between_repeats():
    assert [b.sources for b in ramp_batches(3)] == [(0, 1, 0), (1, 0, 1), (0, 1, 0)]

@pytest.mark.parametrize("capacity,level", [(1, 1), (2, 1), (3, 1), (8, 4), (16, 8)])
def test_the_injected_level_is_half_capacity_at_least_one(capacity, level): ...

def test_injected_batches_alternate_flood_first():
    assert [b.phase for b in injected_batches(4)] == ["injected", "control"] * 3

def test_the_protocol_hash_changes_with_any_declared_value():
    a = protocol(CONFIGURATIONS["cfg-default"], 900.0)
    b = protocol(CONFIGURATIONS["cfg-hardware"], 900.0)
    assert protocol_sha256(a) != protocol_sha256(b)
    assert a["capacity_criteria"] == {"min_throughput_gain": 0.10, "max_pass_rate_drop": 0.20,
        "max_p90_fraction_of_run_limit": 0.5}
    assert a["ramp_levels"] == [1, 2, 3, 4, 6, 8, 12, 16]
```

- [ ] **Step 2: Run them to make sure they fail** — `uv run pytest services/agent/tests/test_sweep_protocol.py -q`; FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `sweep_protocol.py`.** The `protocol()` document lists every
constant, the configuration, the injector settings, label rule, verdict thresholds and
`held_out: true`.

- [ ] **Step 4: Run the tests** — all pass.

- [ ] **Step 5: Commit** — `feature: declare the natural-overload sweep protocol`

### Task 3: Level metrics and the capacity decision

**Files:**
- Create: `services/agent/src/agentwatch_agent/sweep_analysis.py`
- Test: `services/agent/tests/test_sweep_analysis.py` (fixtures built in `tmp_path` from
  small `batch.json` / step-timing dicts; labeled synthetic)

**Interfaces:**
- Consumes: `experiment.step_timings(run_dir)`, `experiment.variability(...)`,
  `runner.VERIFICATION_FAILED`, Task 2 constants.
- Produces:
  - `classify_run(entry: dict) -> str` — `"passed"` (state `completed`, `task_passed`
    true), `"task_failure"` (state `failed`, reason `VERIFICATION_FAILED`), else
    `"infrastructure_failure"`; `delivery_complete` not true is also
    `"infrastructure_failure"`.
  - `batch_metrics(batch_dir: Path, run_timeout_seconds: float) -> dict` with `status`
    (`"readable"` or `"unreadable"` with `error`), `runs`, `passed`, `task_failures`,
    `infrastructure_failures`, `wall_seconds`, `throughput_per_minute`
    (`passed / (wall_seconds / 60)`; `None` when `wall_seconds` ≤ 0),
    `run_durations` (agent-run `total_seconds`, missing ones counted in
    `run_durations_missing`), `reply_match_rate`, per-model-call span stats (queue, prefill,
    decode, first token, end to end; counts by span status), and `device` — from
    `coverage.coverage_report(run_dir)`'s `agent-run` step window, min/mean/max of the GPU
    gauges (utilization, memory used, SM clock, power), GPM SM occupancy when sampled, and
    CPU PSI `some avg10` and cgroup throttling, each with its availability status. Task 4
    adds label data.
  - `server_changes(before: ServerState-shaped dict, after: dict) -> dict` — counter
    increases for prefix-cache queries and hits (and their ratio), preemptions and request
    successes; `None` for a counter missing on either side, `"reset"` when it decreased.
  - `p90(values: list[float]) -> float | None` — sorted element at `round(0.9 * (n - 1))`.
  - `level_summary(batches: list[dict]) -> dict` — mean/stdev/CV of throughput, pass rate,
    p50/p90/max run duration over all runs, failure counts, `unreadable_batches`, span-stat
    and device distributions, summed `server_changes`, and `sm_clock_drift` (level mean SM
    clock minus the N = 1 mean, reported, never corrected).
  - `capacity_decision(levels: dict[int, dict], hard_stops: dict[int, list[str]],
    run_timeout_seconds: float) -> dict` → `{"capacity": int | None, "result":
    "measured" | "below_1" | "at_least_ceiling", "first_failing_level": int | None,
    "criteria": {level: {name: {"value", "limit", "passed"}}}}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_capacity_is_the_level_before_the_first_failing_one():
    # throughput 1.0, 1.8, 2.5, 2.6 at N=1,2,3,4; level 6 passes again at 3.1
    decision = capacity_decision(levels, {}, 900.0)
    assert decision["capacity"] == 3 and decision["first_failing_level"] == 4
    assert decision["criteria"][4]["throughput_gain"]["passed"] is False

def test_a_hard_stop_fails_its_level(): ...       # hard_stops={2: ["vllm_restart"]} -> capacity 1
def test_n1_failing_gives_below_1(): ...          # result "below_1", capacity None
def test_every_level_passing_reports_the_ceiling(): ...  # "at_least_ceiling", capacity 16
def test_pass_rate_may_drop_twenty_points_not_more(): ...  # 0.80 vs 1.00 passes; 0.79 fails
def test_p90_duration_limit_is_half_the_run_limit(): ...   # 450.0 passes; 450.1 fails
def test_p90_uses_the_rounded_index(): assert p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
def test_task_failures_are_not_infrastructure_failures(): ...  # VERIFICATION_FAILED reason
def test_zero_throughput_levels_fail_without_dividing_by_zero(): ...  # Review Focus
def test_unreadable_batches_are_reported_not_zeroed(tmp_path): ...    # Review Focus
```

- [ ] **Step 2: Run them to make sure they fail** — FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Levels are evaluated in `RAMP_LEVELS` order; a level with any
unreadable batch fails a `readable` criterion. The throughput gain is
`mean(N) / mean(previous) - 1`; a previous mean of 0 makes the gain `None` and the criterion
fails with `"reason": "previous level had zero throughput"`.

- [ ] **Step 4: Run the tests** — all pass.

- [ ] **Step 5: Commit** — `feature: compute capacity-sweep level metrics and capacity decision`

### Task 4: Label validation against measured vLLM queue time

**Files:**
- Modify: `services/agent/src/agentwatch_agent/sweep_analysis.py`
- Test: `services/agent/tests/test_sweep_analysis.py`

**Interfaces:**
- Consumes: each run's `agent-evidence.json` `model_calls[]` (`step_id`, `timing` with
  `status` and `queue_seconds`, `prefill_seconds`, `decode_seconds`), `step_timings`.
- Produces:
  - `baselines(n1_runs: list[Path]) -> dict[tuple[str, str], dict]` keyed by
    `(plan_sha256, step_id)` → `variability(...)` of `total_seconds`; a key with fewer than
    3 values is omitted and counted.
  - `noise_threshold(baseline: dict) -> float` = `max(0.01, 3.0 * stdev)`.
  - `label_validation(runs: list[Path], reference: dict) -> dict` per level and overall:
    `confusion` (`tp`, `fp`, `fn`, `tn`), `precision`, `recall` (`None` when undefined),
    `spearman` of excess vs queue (`None` below 3 pairs), `excess_explained`
    (`queue_seconds` sum and `prefill_seconds + decode_seconds` increase over the baseline's
    mean service time), `excluded` by span status, `tool_steps` label rate marked
    `"unvalidated"`, and `verdict` (`"validated"` / `"not_validated"` / `"inconclusive"` when
    precision or recall is `None`).
  - `max_running_check(runs: list[Path], max_num_seqs: int) -> dict` — highest sampled
    `inference.vllm.requests_running.v1`; `"contradicted"` when above `max_num_seqs`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_label_and_reference_use_the_same_noise_threshold(): ...
    # baseline mean 1.0, stdev 0.1 -> threshold 0.3; excess 0.31 & queue 0.31 -> tp;
    # excess 0.31 & queue 0.29 -> fp; excess 0.29 & queue 0.31 -> fn
def test_the_noise_floor_is_ten_milliseconds(): assert noise_threshold({"stdev_seconds": 0.001}) == 0.01
def test_the_verdict_needs_both_precision_and_recall_at_090(): ...  # 0.90/0.90 validated; 0.89 not
def test_undefined_precision_is_inconclusive(): ...               # no predicted positives
def test_steps_without_a_measured_span_are_excluded_and_counted(): ...  # Review Focus
def test_tool_steps_are_reported_unvalidated(): ...
def test_span_timestamps_are_never_used(): ...  # changing server_start_unix_nano changes nothing
def test_running_above_max_num_seqs_is_a_contradiction(): ...
```

- [ ] **Step 2: Run them to make sure they fail.**

- [ ] **Step 3: Implement.** Spearman: rank with ties averaged (standard library only).

- [ ] **Step 4: Run the tests** — all pass.

- [ ] **Step 5: Commit** — `feature: validate experiment labels against measured vLLM queue time`

### Task 5: vLLM server state and the sweep driver

**Files:**
- Create: `services/agent/src/agentwatch_agent/vllm_server.py`
- Create: `services/agent/src/agentwatch_agent/sweep.py`
- Test: `services/agent/tests/test_vllm_server.py`, `services/agent/tests/test_sweep.py`

**Interfaces:**
- Consumes: `prometheus_text.fetch_exposition`, `vllm.vllm_models`, `AgentBatch`,
  `ReplayRunner`, `InjectionSet`, `experiment.injection_active`, `write_json_atomic`,
  Tasks 2–4.
- Produces:
  - `@dataclass(frozen=True) class ServerState(start_time_seconds: float | None,
    cache_config: dict[str, str], counters: dict[str, float | None], running: float | None,
    waiting: float | None)`; `read_server_state(metrics_url: str, timeout_seconds: float =
    2.0) -> ServerState` (raises `ServerStateUnavailable` on fetch or parse failure).
    Counters: `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`,
    `vllm:num_preemptions_total`, `vllm:request_success_total`.
  - `verify_configuration(declared: VllmConfiguration, models: dict, state: ServerState) ->
    dict` → per setting `verified` / `contradicted` / `declared_not_verified`, plus
    `kv_cache_size_tokens`, `num_gpu_blocks`, `kv_cache_max_concurrency`;
    `gpu_memory_utilization` is verified when the reported value is one of the declared
    fallbacks.
  - `class CapacitySweep(*, configuration, sources: list[ReplayPlan], output_dir: Path,
    run_timeout_seconds: float, make_batch: Callable[[Batch, Path, MonotonicClock], AgentBatch],
    make_live_batch: Callable[[int, Path, MonotonicClock], AgentBatch],
    make_injectors: Callable[[MonotonicClock, float], list[BoundedInjector]],
    read_state: Callable[[], ServerState], models: Callable[[], dict],
    sleep: Callable[[float], None] = time.sleep, host: dict)` with `run() -> dict` and
    `request_cancel_from_signal(source, reason)`.
  - Sweep states as in the spec; hard-stop codes `infrastructure_failure`,
    `delivery_incomplete`, `server_busy_after_idle_wait`, `vllm_restart`,
    `server_state_unavailable`, `injector_cleanup_unconfirmed`.

- [ ] **Step 1: Write the failing `vllm_server` tests** using a fixture exposition copied
from the label set recorded on 2026-10-27 (`gpu_memory_utilization="0.6"`,
`enable_prefix_caching="True"`, `kv_cache_size_tokens="7376"`, `num_gpu_blocks="461"`,
`block_size="16"`), labeled synthetic:

```python
def test_reported_settings_are_verified_and_max_num_seqs_is_not(): ...
    # max_model_len 4096 verified; gpu_memory_utilization verified; max_num_seqs declared_not_verified
def test_a_contradicted_setting_is_reported(): ...      # gpu_memory_utilization "0.9"
def test_a_fallback_memory_value_is_accepted_for_cfg_hardware(): ...  # "0.75"
def test_an_unreachable_metrics_endpoint_raises_server_state_unavailable(): ...
```

- [ ] **Step 2: Run them to fail, implement `vllm_server.py`, run them to pass.**

- [ ] **Step 3: Write the failing driver tests** with fake `make_batch` (returns a stub
batch writing a fixed `batch.json`), a fake `read_state` sequence, a fake injector and a
recording `sleep`:

```python
def test_phases_run_in_declared_order(tmp_path): ...
    # warmup, ramp N=1 x6, ramp levels..., live x3 at capacity, injected/control x6
def test_each_hard_stop_ends_the_ramp(tmp_path, code): ...  # parametrized over the codes
def test_a_vllm_restart_stops_the_whole_sweep(tmp_path): ...
def test_unconfirmed_injector_cleanup_stops_the_sweep(tmp_path): ...
def test_unreadable_server_state_is_a_hard_stop(tmp_path): ...            # Review Focus
def test_a_batch_whose_runs_never_started_stops_the_sweep(tmp_path): ...  # Review Focus
def test_the_idle_wait_times_out_as_server_busy(tmp_path): ...
def test_pacing_pauses_ten_seconds_after_the_server_is_idle(tmp_path): ...
def test_sweep_json_is_rewritten_after_every_batch(tmp_path): ...
def test_a_contradicted_configuration_refuses_the_sweep(tmp_path): ...  # nothing runs
def test_cancellation_runs_nothing_further(tmp_path): ...
def test_capacity_below_one_skips_live_and_injected_phases(tmp_path): ...
def test_a_resend_needed_run_stops_the_sweep(tmp_path): ...  # state "stopped_resend_needed"
```

- [ ] **Step 4: Run them to fail, implement `sweep.py`, run them to pass.** Each batch gets
a new `MonotonicClock`; the injected phase wraps `AgentBatch.run()` in `InjectionSet(clock,
make_injectors(clock, bound), ramp_seconds=2.0, cancel=...)` and writes
`injections/<NN>.json`. Server state is read before and after every batch; counter changes are
stored per batch. After the ramp, `capacity_decision` runs on the recorded batches.

- [ ] **Step 5: Commit** — `feature: drive the natural-overload capacity sweep over AgentBatch`

### Task 6: `capacity-sweep` command, Make target and README

**Files:**
- Modify: `services/agent/src/agentwatch_agent/cli.py` (`parser`, `main`, new
  `capacity_sweep_command`), `Makefile` (`agent-capacity-sweep`, `.PHONY`),
  `services/agent/README.md` (a "Capacity sweep" section after "Replay experiments")
- Test: `services/agent/tests/test_cli.py`

**Interfaces:**
- Consumes: `CapacitySweep`, `CONFIGURATIONS`, `load_replay_plan`, `build_model`,
  `server_evidence`, `PrefillFloodInjector`.
- Produces: `capacity-sweep --configuration {cfg-default,cfg-hardware} --source DIR --source
  DIR [--output-dir .data/capacity-sweeps] [--run-timeout 900] [--start-attempt
  "UTIL:OUTCOME" ...] [model, server-evidence and --api-url arguments]`; the model backend
  is forced to vLLM. `--start-attempt` (repeatable, e.g. `"0.8:failed: CUDA out of memory"`,
  `"0.75:started"`) records the operator's vLLM start attempts in `sweep.json`; the last must
  be `started` with the utilization vLLM reports, or the sweep refuses (exit 2). Exit codes
  0/1/2/3 as the spec.
  Make: `make agent-capacity-sweep CONFIG=cfg-default SOURCES="dir1 dir2"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_capacity_sweep_needs_two_sources_of_different_tasks(tmp_path): ...  # exit 2
def test_capacity_sweep_refuses_an_unknown_configuration(): ...              # exit 2
def test_capacity_sweep_refuses_an_unreplayable_source(tmp_path): ...        # exit 2
def test_capacity_sweep_refuses_the_ollama_backend(): ...                    # exit 2
def test_start_attempts_must_end_with_the_reported_utilization(tmp_path): ... # exit 2
def test_capacity_sweep_exit_codes_follow_the_sweep_state(tmp_path, state, code): ...
```

- [ ] **Step 2: Run them to fail, implement, run them to pass.** Signals cancel the sweep as
`experiment_command` does. The injector factory refuses a non-loopback vLLM URL before
anything runs.

- [ ] **Step 3: Write the README section**: what the sweep measures, the declared protocol
(link the spec), invocation per configuration with the vLLM restart between them, output
layout, the capacity rule, label validation and its verdict, exit codes, that results
describe this laptop only, and the held-out status of the runs.

- [ ] **Step 4: Commit** — `feature: add the capacity-sweep command and its documentation`

### Task 7: Full local verification

- [ ] **Step 1:** In WSL Ubuntu, `uv run ruff check .` and the full `uv run pytest` with the
dedicated `agentwatch_test` database. Expected: clean, all pass, 0 skipped. Record counts.
- [ ] **Step 2:** Run `capacity-sweep` once end to end against the fake vLLM test server and
in-memory live server from a test (already covered in Task 5); fix anything found.

### Task 8: GPU sessions (operator: Benoy; serialized)

Not automatable. Each step records commands, start and end times and outcomes; failed and
stopped attempts are kept.

- [x] **Step 1: Record the two sources** under `cfg-default`: `make vllm-up`, the API in
Ubuntu, GPM exporter on Windows, then `make agent-run` for `fix-mean-empty` and
`fix-median-even` with `--model-backend vllm` and `--run-timeout 450`. A source must be
completed with passing verification; keep failed attempts, never re-record a passing source.
Done 2026-11-11 (`.data/a06-20261111/sources/`). `fix-window-count` failed 4 of 4 on vLLM and
was replaced by `fix-median-even` (see the spec, "Changes before the first protocol session").
- [ ] **Step 2: Sweep `cfg-default`**: `make agent-capacity-sweep CONFIG=cfg-default
SOURCES="..."` with `--gpm-file` and `--dcgm-url` if DCGM is up. Then
`python -m agentwatch_agent coverage` on one run per level for sampling quality.
- [ ] **Step 3: Restart vLLM for `cfg-hardware`**: `VLLM_MAX_NUM_SEQS=16
VLLM_GPU_MEMORY_UTILIZATION=0.8 make vllm-up`; on failure try 0.75, then 0.7, recording each
attempt's error. Sweep `cfg-hardware`, passing every attempt with `--start-attempt`.
- [ ] **Step 4: Stop Docker services and Postgres when done** (Benoy's preference).

### Task 9: Capacity record, checklist, data card and handoff

**Files:**
- Create: `docs/capacity-record.md`
- Modify: `docs/project-checklist.md`, `docs/data-card.md`

- [ ] **Step 1:** Write `docs/capacity-record.md` from `sweep.json` only: hardware, both
configurations as verified, each level's throughput/pass rate/p90/CV/queue/prefix-cache/
SM clock, hard stops, capacity per configuration with criteria, label-validation verdicts,
the injected-condition result, every failed or stopped attempt, limitations (this laptop
only; replays share prefixes; thermals; WSL2 PSI scope).
- [ ] **Step 2:** Checklist: A06 progress dated 2026-11-xx with sweep IDs; "Evaluate natural
overload" stays open until model and rule scoring (PR 3). Data card: natural-overload and
`prefill-flood-openai-chat-v1` runs are held-out test evidence.
- [ ] **Step 3:** Copy redacted bundles to `C:\Users\benoy\AgentWatch-evidence\a06-<date>\`;
draft the split-v3 hold-out note for Rohit and Akshay and the configuration table for Meghana
(Benoy sends them).
- [ ] **Step 4:** Commit — `docs: record measured capacity and natural-overload label validation`.

### Task 10: Sync, final checks and PR

- [ ] **Step 1:** `git fetch origin && git merge origin/main`; resolve by understanding both
sides; rerun Task 7's checks.
- [ ] **Step 2:** Review `git diff origin/main...HEAD` and `git log --oneline
origin/main..HEAD`; no Claude trailers.
- [ ] **Step 3:** Ask Benoy before pushing. PR title `feature: measure natural-overload
capacity on the GPU laptop and validate labels against vLLM queue time`; body lists the
dependency on the prefill-flood PR, checks and counts, measured capacities, label verdicts,
limitations and that PR 3 (rule and model scoring) follows separately.
