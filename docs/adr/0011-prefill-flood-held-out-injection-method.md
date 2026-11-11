# 0011: Add a prefill flood as a held-out GPU injection method

Date: November 10, 2026. Status: proposed. Adds `PrefillFloodInjector`
(`prefill-flood-openai-chat-v1`) to the [bounded injectors](0006-bounded-injectors-and-agent-replay.md).

The only held-out injection method is the CPU `worker_queue_blocker`. Both injectors that
contend for the agent's inference server, `CompetingInferenceInjector` and the in-process
retrieval flood, appear in the training replays, so no GPU or inference contention method is
held out from training. A held-out GPU method must differ from the training one in how it loads
the server, not only in its settings.

**Decision.** `PrefillFloodInjector` keeps `concurrency` non-streaming chat requests in flight
against the run's own vLLM server, from one child process that reuses the shared request loop
(`request_flood` with a `PrefillProfile`). The training method asks the model to generate up to
`num_predict` tokens, so it is decode-heavy. The prefill flood sends a long prompt and
asks for one token (`max_tokens=1`), so the server's time goes to prefill, and generation
barely runs.

- The first line of every prompt is a nonce, `request {run_token}-{worker}-{sequence}`, followed
  by fixed numbered filler lines. `run_token` is random, generated once per injector instance
  (12 hex characters), because the worker and sequence counters restart at 0 in every run and a
  later run against the same server would otherwise repeat an earlier run's prompts. The prompts
  share the chat template and the word "request", but they differ within the first KV-cache
  block (vLLM's block size is 16 tokens), so no cached block can be reused and every request
  pays its full prefill cost. The token is never logged.
- Bounds: loopback endpoints only; concurrency at most 4; `prompt_tokens` at most 3,072;
  request timeout at most 120 s; the usual `max_duration_seconds` and backstop.
- Prompts are fixed, non-sensitive text and are never logged. The log records the target
  `prompt_tokens`, the server-reported `usage.prompt_tokens` (count, minimum, median, maximum)
  and the number of completions that reported none, which is kept apart from a measured count.
  A rejected reply (non-200 or no finished choice) is counted as `PrefillRejected` in
  `failures_by_type`.
- The filler size is an estimate. `FILLER_TOKENS_PER_LINE` records the measurement below; vLLM's
  reported `usage.prompt_tokens` includes the chat template, whose overhead was not measured here.

**Measurement (2026-11-10).** Against vLLM v0.32, `POST /tokenize` with model
`Qwen/Qwen3-1.7B@70d244cc86cc`, `add_special_tokens: false`, on the text of this filler line
followed by a newline, for line number `n`:

```
{n:04d} the quick brown fox jumps over the lazy dog near the quiet river bank
```

This is 19 tokens for n = 1, 57, 999 and 3071, and 200 consecutive lines are 3,800 tokens
(19.0 per line). The nonce line `request 3-104729` followed by a newline is 11 tokens; that was measured for the
worker-sequence form, and the run token in the current nonce adds a few tokens that were not
measured. So
`FILLER_TOKENS_PER_LINE = 19`, and a prompt has `ceil(prompt_tokens / 19)` filler lines plus the
nonce line. A later change records vLLM's reported `usage.prompt_tokens` against the real server
to establish the chat-template overhead.

**Consequences.** The method is new and labeled `prefill`, a different injector kind from
`inference`, so a set can contain both. A formal hold-out of this method needs a new frozen
split (`split-v3`) that keeps it out of training; until Rohit and Akshay accept that, runs
that use it are held-out test evidence outside the contract gate. Nothing here claims the flood
isolates GPU prefill from the host: it shares the device with the agents, and its effect is
observed through the server's measured timing, not assumed.
