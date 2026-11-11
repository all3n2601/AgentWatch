# Prefill-flood injector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PrefillFloodInjector` (`prefill-flood-openai-chat-v1`), a bounded, prefill-heavy
vLLM injector that is the GPU held-out injection method for A06.

**Architecture:** The competing-chat child loop in `inference.py` becomes a shared request loop
driven by a picklable request profile (payload per request, measurement per reply, summary per
report). `CompetingInferenceInjector` keeps its exact behavior through a competing-chat profile;
the new `prefill.py` adds a prefill profile and the injector class. Lifecycle, bounds, watchdog,
backstop and cleanup come unchanged from `BoundedInjector` and `processes`.

**Tech Stack:** Python 3.13, standard library only (`http.client`, `multiprocessing` spawn),
pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-10-natural-overload-capacity-sweep-design.md`
(section "PR 1: prefill-flood injector").

**Branch:** `feature/prefill-flood-injector`, created from the latest `origin/main`, with the
spec and both plan commits cherry-picked from `feature/natural-overload-capacity-sweep` so
reviewers see the design.

## Global Constraints

- Method name `prefill-flood-openai-chat-v1`; injector `kind = "prefill"`;
  `resource_class = "inference"`; vLLM `/v1/chat/completions` only.
- Bounds: loopback server only; `concurrency` 1–4; `prompt_tokens` 1–3,072; `max_tokens`
  fixed at 1; request timeout ≤ 120 s; `max_duration_seconds` ≤ 1,800 with backstop
  = bound + `BACKSTOP_GRACE_SECONDS`.
- Every request starts with a unique nonce (worker index and sequence number) before the
  filler, so vLLM's prefix cache cannot reuse an earlier prefill.
- Final-review ruling: the nonce is `request {worker}-{sequence}-{run_token}`, where
  `run_token` is random per injector instance, so `filler_prompt` and `PrefillProfile` take
  `run_token`. The per-request digits come first because vLLM caches 16-token blocks.
- Prompts are never logged. The log holds counts and token sizes only.
- Unreported activity is `{"report": "missing"}`, never zeros.
- `CompetingInferenceInjector`'s payloads and report keys stay byte-for-byte as today; every
  existing test in `services/injectors/tests/test_injectors.py` passes unmodified.
- No new dependencies. Injection logs are labeling metadata, never features.
- Commit titles `<type>: <specific change>`; no Claude co-author trailer or footer.

## Review Focus

- **vLLM rejects an over-long prompt (HTTP 400)** when the filler estimate overshoots the
  4,096 context: each rejection must count as a `PrefillRejected` failure with backoff, never a
  completion. Test: `test_rejected_prefill_requests_count_as_failures` (Task 2).
- **A reply without `usage.prompt_tokens`** (proxy, older server): the completion still counts,
  and the token summary reports how many replies had no count instead of inventing one. Test:
  `test_completions_without_prompt_token_counts_are_counted_apart` (Task 2).
- **Two injectors of different kinds in one `InjectionSet`** (prefill with competing chat):
  kinds must differ so the set accepts them. Test:
  `test_prefill_and_competing_injectors_share_a_set` (Task 2).
- **Nonce collisions across workers** with high sequence numbers: prefixes stay unique per
  worker and sequence. Covered by `test_every_request_has_a_unique_prefix` (Task 2) with two
  workers.
- **The refactor silently changes competing-chat payloads** (thinking flag, `max_tokens`):
  the existing vLLM and Ollama payload tests pin them; run them before and after Task 1.

---

### Task 1: Shared request loop with request profiles

**Files:**
- Modify: `services/injectors/src/agentwatch_injectors/inference.py` (`_compete`,
  `CompetingInferenceInjector.payload`, `_launch`)
- Test: `services/injectors/tests/test_injectors.py`

**Interfaces:**
- Produces:
  - `class RequestProfile(Protocol)` with `path: str`,
    `rejection: type[GenerateRejected]` (the error class `request_flood` counts a non-200 or
    unusable reply as; `GenerateRejected` for the competing profile),
    `payload(self, worker: int, sequence: int) -> dict`,
    `measure(self, result: object) -> int | None` (raises `GenerateRejected` for an unusable
    reply), `summarize(self, measurements: list[int], unmeasured: int) -> dict`.
  - `@dataclass(frozen=True) class CompetingChatProfile` (module level, picklable) with
    fields `server: str`, `payload_template: dict`; `measure` wraps `generated_tokens`;
    `summarize` returns `{"generated_tokens": sum(measurements)}`.
  - `request_flood(stop_event, ready, reports, backstop_seconds, url: str,
    profile: RequestProfile, concurrency: int, request_timeout_seconds: float) -> None`
    — the child target (renamed from `_compete`). Report keys: the existing
    `requests_started`, `requests_sent`, `requests_completed`, `requests_failed`,
    `failures_by_type`, `requests_abandoned_in_flight`, `stopped_by_backstop`, merged with
    `profile.summarize(...)`.
  - `_generate` and `loopback_url` unchanged and importable by `prefill.py`.

- [ ] **Step 1: Run the existing injector tests to record the baseline**

Run (WSL Ubuntu, per the run-setup notes):
`uv run pytest services/injectors/tests/test_injectors.py -q`
Expected: all pass. Note `test_workers_that_never_apply_load_fail_the_start` is a known
pre-existing timing flake; rerun it alone if it fails.

- [ ] **Step 2: Write the failing test**

```python
def test_competing_profile_payloads_do_not_vary_by_request():
    profile = CompetingChatProfile(server="vllm", payload_template={"model": "m", "max_tokens": 128})
    assert profile.payload(0, 0) == profile.payload(3, 41) == {"model": "m", "max_tokens": 128}
    assert profile.path == "/v1/chat/completions"
    assert profile.summarize([5, 7], unmeasured=1) == {"generated_tokens": 12}
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `uv run pytest services/injectors/tests/test_injectors.py::test_competing_profile_payloads_do_not_vary_by_request -q`
Expected: FAIL with `ImportError` / `NameError` for `CompetingChatProfile`.

- [ ] **Step 4: Implement the profile and `request_flood` in `inference.py`**

`path` is a property from `SERVERS[server][0]`. Each worker keeps its own sequence counter and
calls `profile.payload(index, sequence)` per request. `measure` returning `None` increments
`unmeasured` instead of appending. A `GenerateRejected` from `_generate` or `measure` is
re-raised as `profile.rejection` with the same message before it is counted.
`CompetingInferenceInjector._launch` passes
`CompetingChatProfile(self.server, self.payload())`; `payload()` stays public and unchanged.

- [ ] **Step 5: Run every injector test**

Run: `uv run pytest services/injectors/tests -q`
Expected: all pass, including the unmodified competing-chat payload and report tests.

- [ ] **Step 6: Commit**

```bash
git add services/injectors/src/agentwatch_injectors/inference.py services/injectors/tests/test_injectors.py
git commit -m "refactor: share the inference injector request loop through request profiles"
```

### Task 2: `PrefillFloodInjector`, its docs and ADR 0011

**Files:**
- Create: `services/injectors/src/agentwatch_injectors/prefill.py`
- Create: `docs/adr/0011-prefill-flood-held-out-injection-method.md`
- Modify: `services/injectors/README.md` (injector table row, lifecycle "applies load" bullet,
  limitations), `docs/adr/README.md` (index entry, if it lists ADRs)
- Test: `services/injectors/tests/test_injectors.py`

**Interfaces:**
- Consumes: `request_flood`, `RequestProfile`, `GenerateRejected`, `_generate`,
  `loopback_url` (Task 1); `BoundedInjector`, `require_count`, `require_seconds`,
  `BACKSTOP_GRACE_SECONDS` (base); `new_group`, `start_children`, `halt_children` (processes).
- Produces (used by the capacity-sweep plan):
  - `PREFILL_METHOD = "prefill-flood-openai-chat-v1"`, `MAX_PREFILL_CONCURRENCY = 4`,
    `MAX_PROMPT_TOKENS = 3072`, `FILLER_TOKENS_PER_LINE: int` (measured, Step 1).
  - `class PrefillRejected(GenerateRejected)`.
  - `filler_prompt(worker: int, sequence: int, prompt_tokens: int) -> str` — nonce line
    `request {worker}-{sequence}` first, then `ceil(prompt_tokens / FILLER_TOKENS_PER_LINE)`
    fixed numbered filler lines.
  - `@dataclass(frozen=True) class PrefillProfile(model: str, prompt_tokens: int)`;
    `measure` returns `usage.prompt_tokens` (raises `PrefillRejected` without a finished
    choice); `summarize` returns `{"prompt_tokens_target": n, "prompt_tokens":
    {"count", "min", "median", "max"} or None, "completions_without_prompt_tokens": k}`.
  - `PrefillFloodInjector(clock, max_duration_seconds, *, url: str, model: str,
    concurrency: int = 2, prompt_tokens: int = 2048, request_timeout_seconds: float = 60.0)`
    with `kind = "prefill"`, `method = PREFILL_METHOD`, `resource_class = "inference"`.
    `bounds()` = `{"concurrency", "prompt_tokens", "max_tokens": 1,
    "request_timeout_seconds", "backstop_seconds"}`; `target()` as in
    `CompetingInferenceInjector` with `server: "vllm"`.
  - Payload: `{"model", "messages": [{"role": "user", "content": filler_prompt(...)}],
    "max_tokens": 1, "temperature": 0, "stream": False,
    "chat_template_kwargs": {"enable_thinking": False}}`.

- [ ] **Step 1: Measure `FILLER_TOKENS_PER_LINE` on the real tokenizer**

Start vLLM (`make vllm-up`), then POST the fixed filler line 100 times to
`http://127.0.0.1:8000/tokenize` with `{"model": "Qwen/Qwen3-1.7B@70d244cc86cc",
"prompt": <text>}` and divide `count` by 100. Record the command and count in ADR 0011. Use
the ceiling. Stop vLLM afterwards (`make vllm-down`).

- [ ] **Step 2: Write the failing tests**

Use `FakeOllama(reply=PREFILL_COMPLETION)` where
`PREFILL_COMPLETION = {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
"usage": {"prompt_tokens": 2050, "completion_tokens": 1}}`.

```python
def test_prefill_requests_are_long_prompts_with_one_output_token(clock):
    # run 0.6 s, concurrency 1, prompt_tokens 256
    assert log["method"] == "prefill-flood-openai-chat-v1"
    assert log["injector"] == "prefill" and log["contended_resource_class"] == "inference"
    assert set(server.paths) == {"/v1/chat/completions"}
    payload = server.payloads[0]
    assert payload["max_tokens"] == 1 and payload["stream"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    lines = payload["messages"][0]["content"].splitlines()
    assert len(lines) == 1 + math.ceil(256 / FILLER_TOKENS_PER_LINE)
    assert activity["prompt_tokens"]["median"] == 2050
    assert activity["prompt_tokens_target"] == 256
    assert log["cleanup"]["confirmed"] is True

def test_every_request_has_a_unique_prefix(clock):
    # concurrency 2, run until >= 6 payloads
    firsts = [p["messages"][0]["content"].splitlines()[0] for p in server.payloads]
    assert len(set(firsts)) == len(firsts)

def test_prompts_are_never_logged(clock):
    assert "request 0-0" not in json.dumps(log)

@pytest.mark.parametrize("setting,value", [("concurrency", 5), ("concurrency", 0),
    ("prompt_tokens", 3073), ("prompt_tokens", 0), ("request_timeout_seconds", 121.0)])
def test_prefill_load_is_bounded(clock, setting, value): ...  # raises ValueError

def test_prefill_injector_only_loads_loopback_servers(clock): ...  # http://10.0.0.5:8000 -> ValueError

def test_rejected_prefill_requests_count_as_failures(clock):
    # FakeOllama(status=400)
    assert activity["requests_completed"] == 0 and activity["requests_failed"] >= 1
    assert activity["failures_by_type"] == {"PrefillRejected": activity["requests_failed"]}

def test_completions_without_prompt_token_counts_are_counted_apart(clock):
    # reply without "usage"
    assert activity["requests_completed"] > 0
    assert activity["prompt_tokens"] is None
    assert activity["completions_without_prompt_tokens"] == activity["requests_completed"]

def test_an_unreachable_server_fails_the_prefill_start(clock, monkeypatch): ...
    # mirror test_an_unreachable_model_server_fails_the_start; cleanup confirmed

def test_prefill_activity_is_missing_when_the_child_never_reported(clock, monkeypatch): ...
    # halt_children returns [None] -> activity == {"report": "missing"}

def test_prefill_and_competing_injectors_share_a_set(clock):
    InjectionSet(clock, [prefill, competing])  # no ValueError; kinds "prefill", "inference"
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `uv run pytest services/injectors/tests/test_injectors.py -k prefill -q`
Expected: FAIL with `ModuleNotFoundError: agentwatch_injectors.prefill`.

- [ ] **Step 4: Implement `prefill.py`**

Mirror `CompetingInferenceInjector` for `__init__` validation, `target`, `_launch` (child
name `agentwatch-prefill-injector`) and `_halt` (missing report → `{"report": "missing"}`).
`PrefillProfile.rejection = PrefillRejected`, so `request_flood` records every rejected
prefill request under `PrefillRejected` in `failures_by_type`.

- [ ] **Step 5: Run every injector test**

Run: `uv run pytest services/injectors/tests -q`
Expected: all pass.

- [ ] **Step 6: Write the docs**

- README table row: `PrefillFloodInjector` | `prefill-flood-openai-chat-v1` | vLLM
  (`inference`) | non-streaming chat requests with a unique nonce and about `prompt_tokens`
  of fixed filler, `max_tokens=1`, from one child process | loopback only; concurrency ≤ 4;
  prompt_tokens ≤ 3,072; request timeout ≤ 120 s.
- Lifecycle bullet: "Prefill: every worker has sent its first request."
- Limitations: filler size is an estimate from `FILLER_TOKENS_PER_LINE`; the log records the
  server-reported `prompt_tokens`. Load shares the device with the agents.
- ADR 0011 (follow the existing ADR format): context (only held-out method is CPU
  `worker_queue_blocker`; both agent injectors are in training replays), decision (prefill-
  heavy versus decode-heavy competing chat; nonce-first prompt defeats prefix caching;
  bounds), the Step 1 measurement, consequences (formal hold-out needs `split-v3`; until
  Rohit and Akshay accept it, runs are held-out test evidence outside the contract gate).

- [ ] **Step 7: Commit**

```bash
git add services/injectors docs/adr
git commit -m "feature: add a bounded prefill-flood vLLM injector as the held-out GPU method"
```

### Task 3: Verify, sync and open the PR

**Files:** none new.

- [ ] **Step 1: Sync with `origin/main`**

`git fetch origin && git merge origin/main` (no rebase). Resolve conflicts by understanding
both sides.

- [ ] **Step 2: Run the checks**

In WSL Ubuntu: `uv run ruff check .` and the full `uv run pytest` with
`AGENTWATCH_TEST_DATABASE_URL` pointing at `agentwatch_test` (see the run-setup notes).
Expected: Ruff clean; all tests pass with 0 skipped. Record counts. A failure of the known
flake is rerun alone and reported, never ignored.

- [ ] **Step 3: Exercise the injector against real vLLM**

Start vLLM, run `PrefillFloodInjector` for 20 s at concurrency 2 and `prompt_tokens=2048`
from a short script in the scratchpad, and record `activity` (completions, failures,
`prompt_tokens` median) plus `vllm:prefix_cache_hits_total` before and after. Expected:
completions > 0, failures 0, median within ±15% of 2048, and prefix-cache hits growing by far
less than prompt tokens sent. Stop vLLM.

- [ ] **Step 4: Review the branch**

`git diff origin/main...HEAD` and `git log --oneline origin/main..HEAD`: only the spec, the
two plans, `inference.py`, `prefill.py`, tests, README and ADR 0011. No Claude trailer
(`git log --format=%(trailers) origin/main..HEAD`).

- [ ] **Step 5: Ask Benoy before pushing and opening the PR**

PR title `feature: add a bounded prefill-flood vLLM injector as the held-out GPU method`.
Body: what and why, the inference-loop refactor (shared interface for the agent's
experiment), checks and counts, the Step 3 real-vLLM result, the limitation that formal
hold-out needs `split-v3`, and merge order (this PR before the capacity-sweep PR).
