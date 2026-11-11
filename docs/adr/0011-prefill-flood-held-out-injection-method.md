# 0011: Add a prefill flood as a held-out GPU injection method

Date: November 10, 2026. Status: proposed. Adds `PrefillFloodInjector`
(`prefill-flood-openai-chat-v1`) to the [bounded injectors](0006-bounded-injectors-and-agent-replay.md).

The only held-out injection method is the CPU `worker_queue_blocker`. The only inference
method (competing chat) and the only retrieval method (the index flood) both appear in the
training replays, so no inference-contention method is held out from training. A held-out GPU
method must differ from the training one in how it loads the server, not only in its settings.

**Decision.** `PrefillFloodInjector` keeps `concurrency` non-streaming chat requests in flight
against the run's own vLLM server, from one child process that reuses the shared request loop
(`request_flood` with a `PrefillProfile`). The training method asks the model to generate up to
`num_predict` tokens, so it is decode-heavy. The prefill flood sends a long prompt and
asks for one token (`max_tokens=1`), so the server's time goes to prefill, and generation
barely runs.

- The first line of every prompt is a nonce, `request {worker}-{sequence}-{run_token}`,
  followed by fixed numbered filler lines. `run_token` is random, generated once per injector
  instance (12 hex characters), because the worker and sequence counters restart at 0 in every
  run and a later run against the same server would otherwise repeat an earlier run's prompts.
  The per-request digits come first because vLLM caches whole 16-token blocks and block 0 also
  holds the chat template and the word "request" (see the measurements). The token is never
  logged.
- Bounds: loopback endpoints only; concurrency at most 4; `prompt_tokens` at most 3,072;
  request timeout at most 120 s; the usual `max_duration_seconds` and backstop. The log's bounds
  also record `filler_lines`. Changing the filler line or `FILLER_TOKENS_PER_LINE` changes the
  load and needs a new method version (`prefill-flood-openai-chat-v2`).
- Prompts are fixed, non-sensitive text and are never logged. The log records the target
  `prompt_tokens`, the server-reported `usage.prompt_tokens` (count, minimum, `median_low`,
  maximum) and the number of completions that reported none (or a non-positive or non-integer
  count), which is kept apart from a measured count. `median_low` is always a count the server
  actually reported. A non-200 reply, or an unfinished or unusable completion, is counted as
  `PrefillRejected` in `failures_by_type`; an unparseable body is counted under its own error
  type (`JSONDecodeError`), as for competing chat.
- The filler size is an estimate: `FILLER_TOKENS_PER_LINE` records the first measurement below,
  and vLLM's reported `usage.prompt_tokens` is what the log records.

**Measurement 1: tokenizer (2026-11-10).** Against vLLM v0.32, `POST /tokenize` with model
`Qwen/Qwen3-1.7B@70d244cc86cc`, `add_special_tokens: false`, on the text of this filler line
followed by a newline, for line number `n`:

```
{n:04d} the quick brown fox jumps over the lazy dog near the quiet river bank
```

This is 19 tokens for n = 1, 57, 999 and 3071, and 200 consecutive lines are 3,800 tokens
(19.0 per line). The nonce line `request 3-104729` followed by a newline is 11 tokens. So
`FILLER_TOKENS_PER_LINE = 19`, and a prompt has `ceil(prompt_tokens / 19)` filler lines plus
the nonce line.

**Measurement 2: real vLLM (2026-11-10).** The injector ran twice for 20 s each against the same
vLLM v0.32 server (`--max-num-seqs 2`, model above) with concurrency 2 and `prompt_tokens` 2048
(108 filler lines, 2,052 filler tokens). Prefix-cache figures are the change in vLLM's
`prefix_cache_hits_total` and `prefix_cache_queries_total` counters across each run.

| Nonce order | Run | Completed / failed | `usage.prompt_tokens` min / median / max | Cache hits / queries |
| --- | --- | --- | --- | --- |
| `request {run_token}-{worker}-{sequence}` | 1 | 104 / 0 | 2082 / 2083 / 2083 | 1,680 / 220,778 |
| | 2 | 102 / 0 | 2082 / 2083 / 2083 | 1,648 / 216,612 |
| `request {worker}-{sequence}-{run_token}` | 1 | 104 / 0 | 2081 / 2082 / 2082 | 0 / 218,590 |
| | 2 | 104 / 0 | 2082 / 2083 / 2083 | 0 / 218,695 |

With the run token first, about 0.76% of queried tokens hit the cache, roughly 16 tokens per
request: one KV block. The token arithmetic suggests the chat template, `request` and the run
token filled block 0, which every request of a run shared; that attribution is inferred, not
measured. With the per-request digits first there were no hits in either
run. The reported prompt is 30 to 31 tokens longer than the 2,052 filler tokens, which is the
chat template and the nonce line. These are two short runs on one laptop GPU, not a
characterisation of the method's effect on other workloads or servers.

**Consequences.** The method is new and labeled `prefill`, a different injector kind from
`inference`, so a set can contain both. A formal hold-out of this method needs a new frozen
split (`split-v3`) that keeps it out of training; until Rohit and Akshay accept that, runs
that use it are held-out test evidence outside the contract gate. Nothing here claims the flood
isolates GPU prefill from the host: it shares the device with the agents, and its effect is
observed through the server's measured timing, not assumed.
