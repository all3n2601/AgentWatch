# 0015: Each policy version names the model server its inference limit controls

Date: November 21, 2026. Status: proposed in the `feature/vllm-inference-admission` PR. Amends
the applicability rule of [ADR 0010](0010-runner-process-admission-limits.md); the policy API is
described in [the concurrency policy](../concurrency-policy.md).

ADR 0010 applies a control only to the service a policy version names, so a limit set for one
workload never limits another. Every version named `ollama` for the inference control: the
`coding-agent` target definition fixed it, and the service copied it into each row. A runner
sending its model requests to vLLM therefore recorded the limit as not enforced, and nothing
could limit vLLM requests. The hardware evaluation (O06) runs the agent on vLLM on the only GPU
laptop, so its inference candidates would have measured an unenforced limit.

**Decision.** A policy change may name the inference service: `inference_service`, `ollama` or
`vllm`, accepted only with `set`. The service stores it in that version's `controlled_services`;
nothing else in the row's shape changes. Every version still names exactly one server.

| Action | Service the new version names |
| --- | --- |
| `set` with `inference_service` | The one requested |
| `set` without it | The active version's, so an older client changing a limit never moves it |
| `disable` | The active version's (it sets no limit, so it limits nothing) |
| `restore` | The restored version's, so restored limits apply to the server they were set for; version 0 names `ollama` |

The runner is unchanged: it already compares the named service with the one it calls (`ollama`
or `vllm`) and records a mismatch as not enforced, with the reason.

**Alternatives.** Naming no particular server (one limit for every model backend) is smaller,
but a version would no longer say what it limited, and an Ollama limit set for CPU work would
also throttle GPU runs. Having the runner apply an `ollama` version to vLLM would make stored
versions claim a service they did not limit. A second policy target per server would split the
controls, history and run links, and every consumer, for one field's worth of difference.

**Compatibility.** The field is optional and additive in the OpenAPI schema and both SDKs. An
omitted service is dropped from the stored request, so clients that predate it send, store and
retry identical bodies, and idempotent retries of earlier requests still match. Stored versions
are untouched; no migration is needed, because `controlled_services` was already stored per row.

**Consequences.** One active version limits one model server. Ollama and vLLM runners cannot be
limited at the same time; a runner on the other server records why its inference limit was not
applied. The Controls page offers the choice and shows the server with every inference limit.
Enforcement stays per runner process (ADR 0010).
