# 0004: Measure CPU per step with counters, time retrieval service, join steps to their own resource

Date: October 24, 2026. Status: proposed for A04 review.

The first A04 change ([ADR 0003](0003-gpu-and-inference-measurement.md)) left three gaps.
The only CPU pressure reading, PSI `some avg10`, is a 10-second moving average, so it says
little about a step lasting 0.1–1 s. Retrieval steps carried only their step timing, with
no measured service time. And the coverage report joined every sampled resource to every
step, although the live contracts join a step only to its own registered resource. It also
never checked that records belonged to the run or to a registered clock.

**CPU counters.** Three cumulative counters are registered as `cpu_sandbox` microseconds:
host PSI `some total` (`cpu.psi.some.total_usec.v1`), the runner cgroup's `usage_usec`
(`cpu.cgroup.usage_usec.v1`) and that cgroup's own PSI `some total`
(`cpu.cgroup.psi.some.total_usec.v1`). Their increase across a step's window is a measured
amount of stall or CPU time during that window. A decrease anywhere in the window is a reset,
and gives no delta rather than negative load. Each file is read once per round, and every
reading keeps its own availability: a kernel without PSI is `unsupported`, a cgroup without
`cpu.pressure` is `unsupported`, a cpu.stat without a counter is `unavailable`, and a negative
or unparsable value is `malformed`. The existing avg10 and throttling metrics are unchanged.
In WSL2 every process in the distro shares one cgroup and host PSI covers the whole VM,
including Docker. Neither is the runner's exclusive use.

**Retrieval timing.** `search_docs` measures the search itself with the monotonic clock and
thread CPU time (`retrieval-timing-v1`), apart from the step's queue delay (runner
dispatch). The difference between step execution and service time is the runner's own
handling, an estimate. The index is searched synchronously in-process, so its request queue
does not exist. It is recorded as `unsupported`, never as a measured zero. A networked
retrieval service (Qdrant) would report its queue and service time under new method versions.

**Correlation.** `coverage-v2` joins each step to its own registered resource (`own`) and
lists the other sampled resources as labeled `context`. Statuses use the FeatureRow telemetry
vocabulary: `present`, `no_overlap`, `unaligned`, `empty` and `missing`. The model server and
the docs index have no registered metric, so model calls and retrievals report their own
telemetry as `missing`, with host CPU and the GPU as context. The report refuses records of
another run and conflicting records under one identity. It excludes and counts records in an
unregistered clock domain. For every executed step it reports the observed dispatch queue
apart from service time: Ollama's reported total (queue unavailable) or the search alone
(queue unsupported).

Consequences: every run now samples 5 CPU metrics, or 10 with the GPU, so the longest
`--run-timeout` at 250 ms falls to 4,000 s without the GPU and 2,000 s with it. A live API
without these definitions refuses every run's samples, so the service must be deployed
first. The service-side window join and feature computation remain B02 work; `coverage-v2`
is producer-side evidence that applies the same rules to the journal.
