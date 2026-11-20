# A07 handoff to Benoy: GPU overhead attempts and GPU-host redaction

Task A07, "Measure instrumentation overhead and redact traces". Owner Akshayaa, partner
reviewer Benoy. Written 2026-11-20 after PRs #92 to #95 merged. Protocol version
`instrumentation-overhead-acceptance-v1`, declared in
[the overhead protocol](instrumentation-overhead-protocol.md). Read that file first: it fixes
the thresholds, attempt counts and decision rule, and none of them may change after you see
results.

This is a handoff, not evidence. No A07 overhead attempt has been evaluated and redaction has
not been run on GPU-host evidence. The A07 card stays open until the work below is done.

## What is merged

| PR | What it gives you |
| --- | --- |
| #92 | The acceptance protocol: 5% budget, 30 pairs per attempt, two attempts per measurement and host |
| #93 | `make overhead-verdict`: judges saved attempt files against the budget |
| #94 | `sampling_coverage` in saved attempts: the coverage verdict against the declared limits |
| #95 | `redact`: writes a redacted copy of an evidence folder for sharing |

## What only the GPU laptop can do

These are the measurements the protocol assigns to you (RTX 5070 Laptop, WSL2). Akshayaa's
CPU-container attempts are separate and are not combined with yours.

| Measurement | Protocol recorded in the file | Backend |
| --- | --- | --- |
| Sustained sampling, GPU Ollama | `resource-sampling-overhead-v1` | Ollama on the GPU |
| Sustained sampling, GPU vLLM | `resource-sampling-overhead-v2` | vLLM with the DCGM probes |

Each measurement needs **two completed attempts**: 3 discarded warm-ups, then 30
uninstrumented and 30 instrumented repetitions alternating. Keep failed and interrupted
attempts too. At most four attempts per measurement and host, and only a failure caused by
infrastructure justifies a third. Nothing else may use the GPU during an attempt, and contention
injectors stay off.

## Steps

1. Sync and branch from the latest `origin/main`, which includes #93 to #95.
2. Start the model server. For vLLM run `make vllm-up` (stop Ollama first: 8 GB fits one).
   For Ollama run `make llm-up`.
3. Optional device probes. Run `make dcgm-up` for DCGM. For GPM, run `make gpm-exporter` in
   Windows PowerShell and leave it running (ADR 0008).
4. Record the settings listed below before the first attempt. The output files do not hold them.
5. Run an attempt. From WSL:

   ```bash
   uv run --package agentwatch-agent python -m agentwatch_agent sampling-overhead \
     --model-backend vllm \
     --dcgm-url http://127.0.0.1:9400/metrics \
     --gpm-file .data/gpm/gpm.prom \
     --output-dir .data/sampling-overhead/vllm-attempt-1
   ```

   The command always writes `<output-dir>/sampling-overhead.json`, so **give every attempt its
   own `--output-dir`** (`vllm-attempt-1`, `vllm-attempt-2`, `ollama-attempt-1`, ...). Reusing a
   directory overwrites the earlier attempt, which the protocol forbids. Use a relative
   `--output-dir`: the files record absolute host paths otherwise. For the Ollama measurement
   use `--model-backend ollama` and the same device arguments. Run each measurement twice. Exit
   code 1 means a failed or interrupted measurement: keep that directory.
6. Judge each measurement's attempts with one host label, attempts of one measurement only:

   ```bash
   make overhead-verdict \
     ATTEMPTS=".data/sampling-overhead/vllm-attempt-1/sampling-overhead.json .data/sampling-overhead/vllm-attempt-2/sampling-overhead.json" \
     HOST_LABEL=benoy-rtx5070-wsl2
   ```

   Paths containing spaces need the module run directly: `uv run --package agentwatch-data
   python -m agentwatch_data.overhead_verdict` with quoted paths and `--contract
   pipelines/model/optimization-contract.toml --host-label <name>`. The report is
   `.data/reports/overhead/overhead-verdict.json` and a later run overwrites it, so copy it
   aside after each measurement. The command exits 0 whatever the verdict. Do not rerun to
   get a better one.
7. Make the shareable copy and review it by hand:

   ```bash
   uv run --package agentwatch-agent python -m agentwatch_agent redact .data/sampling-overhead --output .data/sampling-overhead-shared
   ```

   Check that no file holds credentials, absolute host paths, user or host names, or emails.
   Then run the same command on one or more real GPU-host agent run folders. This is the part
   that is still unverified: PR #95 re-ran the rules over 104 local run folders with 0
   replacements, none from GPU-host evidence.
8. Write the attempt notes and send the redacted files and the verdict report to Akshayaa.

## Settings to record by hand for every attempt

- Repository commit SHA.
- Model digest or pinned revision, backend, and server image digest.
- Chat options (temperature 0, `num_ctx` 4096, `num_predict` 512) and the 0.25 s sampling
  interval.
- GPU model, driver version, WSL kernel, and whether the laptop was on mains power.
- Whether `--gpm-file` was used.
- Start and end time of each attempt, in UTC.

Attempts whose settings differ are reported separately, never combined into one verdict.

## How to read the result

- Attempt verdicts are `within_budget`, `over_budget` or `inconclusive`. A measurement and host
  gets `within_budget` or `over_budget` only when two attempts completed and both agree.
- `within_budget` needs the 95% interval's upper bound at or below 5%. Small true increases
  will often read `inconclusive`. Report that as it is.
- Report the coverage verdict (`met`, `not_met`, `not_evaluated`) beside it. Fewer than 10
  measured gaps is `not_evaluated`.
- If generated token counts differ between conditions, state it as a limitation of that
  attempt's wall-time comparison. At temperature 0, identical output is not guaranteed.
- The 5% budget is still proposed under O01. If O01 adopts another value, report both.

## Limits of what you can claim

One laptop GPU does not establish L4 or cloud behaviour. Live delivery under sustained sampling
is not measured by either command.

## Where results go

- Verdicts, settings and limitations: the A07 report, with the attempt files, to Akshayaa.
- Data-quality notes (missing samples, unavailable metrics, short-step coverage): Akshay.
- The overhead report: Rohit, for O01.
- After you share evidence, update `docs/project-checklist.md` with the date, the attempt file
  names and the verdicts, including failed and inconclusive ones.
