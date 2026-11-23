# Admission overhead

Status: O06 measurement, implemented 2026-11-22 on branch `feature/admission-overhead` and
verified with test doubles only. Nothing has been measured on hardware yet.

The [optimization contract](optimization-contract.md) gates admission overhead: the median
wall-time increase with unchanged limits must be at most `max_median_admission_overhead` (5%),
measured separately from instrumentation. Runner-process admission
([ADR 0010](adr/0010-runner-process-admission-limits.md)) adds, around every model request, a
permit from the admission controller and a policy watcher polling the policy service.

## Protocol `admission-overhead-v1`

`make agent-admission-overhead ADMISSION_ARGS="--model-backend vllm"` (the live API must be
running, since admission reads its policy) sends the runner's first model request for one task
(default `fix-mean-empty`) to the real model server, as `sampling-overhead` does:

1. 3 warm-up requests, discarded.
2. 30 `admission_off` and 30 `admission_on` requests, alternating, so drift affects both.
   - `admission_off`: the request alone, as a `--no-policy` run sends it.
   - `admission_on`: a policy watcher polling the policy service and an inference permit
     acquired before and released after the request, under the active policy. In a run the
     watcher has been polling since the run began, so a request starts at an arbitrary point
     of the poll cycle. To reproduce that, each `admission_on` request starts its own watcher,
     waits an untimed, uniformly random phase of one poll interval (from a recorded seed,
     `poll_phase_seed`), then starts the timer; the watcher stops after the timer. Each
     measurement records `poll_phase_seconds` and `polls_during_request`, and the report counts
     `requests_overlapping_a_poll`.
3. The active policy is read once before the first request and once after the last. A
   version change at any point fails the measurement: limits must stay unchanged.

No resource sampling or live delivery runs in either condition. The report,
`.data/admission-overhead/<task>-<UTC time>-<id>/admission-overhead.json`, keeps the per-request wall
times and server durations, the policy version and limits, whether the limit was enforced, every
admission record, each watcher's summary and the poll interval. A failed or interrupted
measurement is saved with its status and error.

For O06, measure with the policy the experiment's candidates use set on this runner's model
server (for example `inference_requests` 1 on vLLM): with sequential requests the limit is
enforced but never binds, so the permit path a limited run takes is the one timed.

## Verdict

`make overhead-verdict ATTEMPTS="a/admission-overhead.json b/admission-overhead.json"
HOST_LABEL=<label>` judges the attempts with the
[instrumentation overhead protocol's](instrumentation-overhead-protocol.md) rule:
- two completed attempts that agree;
- 30 pairs per attempt;
- a baseline coefficient of variation of at most 15%;
- a paired bootstrap interval compared with the budget.

The budget here is the contract's `max_median_admission_overhead`, not the instrumentation
budget; each verdict group records the budget it was judged against (`budget`). Attempts are
grouped by their admission settings too: the policy's limits, the service they name, whether
the inference limit was enforced, and the poll interval. Attempts under different settings
measure different admission paths, so they never form one verdict. In the report, `admission_off` fills the "uninstrumented" side of each pair and
`admission_on` the "instrumented" side. The optimization gates read that verdict for the
experiment's model.

## Known limitations

- Requests are sequential, so a permit never waits: this measures the cost of admission, not
  queueing under contention, which is what the optimization experiment compares.
- Polling cost is sampled, as in a run: a request includes a poll only when one falls inside
  it. With requests shorter than the poll interval (the A07 GPU requests took 0.28–0.37 s
  against the default 1 s poll), most `admission_on` requests include none;
  `requests_overlapping_a_poll` shows how many did.
- The verdict report's top-level `protocol` names the shared acceptance rule
  (`instrumentation-overhead-acceptance-v1`); which budget applied is per group.
- One task, one model and one host per attempt; the verdict describes that setup only.
