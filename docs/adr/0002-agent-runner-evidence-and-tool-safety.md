# 0002: Instrument the coding agent with live steps, local evidence and reviewed patches

Date: October 16, 2026; revised October 20, 2026 after review. Status: proposed for A02 review.

A02 needs one real agent task whose model calls, tool calls and retrieval are correlated in
the B01 live API, with configuration and failure evidence preserved. Three constraints shaped
the runner. `live-contracts-v1` StepEvents reject extra fields, so they cannot carry model
token counts or server durations. The repository forbids executing model-written code until
an isolation design exists (A03/O03). And the CPU development laptops have no vector database.

Add `services/agent` as the agent runner. Every model call, tool call and retrieval is a
StepEvent under one root step and trace, using one monotonic clock domain per runner process
and opaque per-process resource IDs. Model-server metrics, the conversation, full
configuration, tool results and cleanup details go to a local `agent-evidence.json`, keyed by
the same step and span IDs. The run's configuration digest covers everything except endpoint
URLs. Task identity is `<name>-r<revision>`, with each task's content hash pinned so a task ID
cannot silently cover changed inputs.

The model can only search task documents (in-process BM25, recorded as `retrieval`), read
fixture files, choose one reviewed patch per file, and run the fixture tests in a disposable
copy through `services/sandbox`. It never supplies code or commands. The runner reruns the
tests itself, and only that verification decides task success.

Delivery is background, batched and idempotent, with an append-only journal for replay after
an outage. A rejected batch is resent record by record, both live and on replay, so a closure
race loses only the refused records. A run is closed only once its records are delivered:
otherwise the closure is journaled, and `resend` sends it after the records, then compares
the run's actual state with the journaled closure.

Consequences: live consumers see correlated lifecycle and timing without model metrics, which
stay local until a contract version adds them. The tasks are small reviewed fixtures, not
open-ended coding. Retrieval timing does not represent a network vector database. Decided
in review (Benoy, owner of the former demo): the repository keeps one agent loop. The
`llm_demo` single-tool loop and its `demo-llm`/`record-llm` targets are removed; its Ollama
helpers move to `agentwatch_worker.ollama`, and the GPU diagnostics drive this runner.
Previously saved demo runs remain viewable in the dashboard's Agent tab, but new agent runs
reach only the live API until the live dashboard view (F02) consumes them. Free-form edits and
commands remain out of scope until A03 provides isolation. See
[the runner README](../../services/agent/README.md).
