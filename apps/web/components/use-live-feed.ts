import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createLiveApi, LIVE_API_TIMEOUT_MS } from "@/lib/live-api";
import { gpuEvidenceSchema, type GpuEvidence } from "@/lib/live-feed-data";
import {
  applyFailure,
  applySnapshot,
  connectionAfterFailure,
  initialRunFeed,
  initialRunList,
  nextListPollDelayMs,
  nextPollDelayMs,
  type RunFeed,
  type RunListFeed,
} from "@/lib/live-poll";

// Browser polling of this app's live routes, which validate and forward B01 responses. The
// state transitions live in lib/live-poll.ts; these hooks only own timers and requests.

/** Leaves room for the route's own wait on the live API before the browser gives up. */
const BROWSER_TIMEOUT_MS = LIVE_API_TIMEOUT_MS + 2_000;
const liveApi = createLiveApi("/api/live/runs", {
  timeoutMs: BROWSER_TIMEOUT_MS,
});

type PollStep = (signal: AbortSignal) => Promise<number | null>;

/**
 * Runs `step` now, then again after the delay it returns until it returns null. Pauses while
 * the tab is hidden and reads immediately when it is shown again. `stop` aborts the request
 * in flight, so a response for a previous selection is never applied.
 */
function startPolling(step: PollStep) {
  let stopped = false;
  let running = false;
  let paused = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let controller: AbortController | null = null;

  async function run() {
    clearTimeout(timer);
    timer = undefined;
    if (stopped || running) return;
    if (document.hidden) {
      paused = true;
      return;
    }
    running = true;
    controller = new AbortController();
    const delay = await step(controller.signal);
    running = false;
    if (!stopped && delay !== null) timer = setTimeout(run, delay);
  }
  const resume = () => {
    if (!document.hidden && paused) {
      paused = false;
      void run();
    }
  };
  document.addEventListener("visibilitychange", resume);
  void run();
  return {
    /** Reads again now, skipping any wait; restarts a loop that had stopped. */
    now: () => void run(),
    stop() {
      stopped = true;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", resume);
    },
  };
}

/** Polls the selected run; retry state is independent of the last valid snapshot. */
export function useRunFeed(runId: string | null) {
  const [feed, setFeed] = useState<RunFeed | null>(null);
  const poller = useRef<ReturnType<typeof startPolling> | null>(null);
  useEffect(() => {
    if (!runId) return;
    let current = initialRunFeed(runId);
    const polling = startPolling(async (signal) => {
      const result = await liveApi.runSnapshot(runId, current.cursor, {
        signal,
      });
      if (signal.aborted) return null;
      current = result.ok
        ? applySnapshot(current, result.data, performance.now())
        : applyFailure(current, result, Math.random());
      setFeed(current);
      return nextPollDelayMs(current);
    });
    poller.current = polling;
    return () => {
      polling.stop();
      poller.current = null;
    };
  }, [runId]);
  const readAgain = useCallback(() => poller.current?.now(), []);
  // State from a previous selection is never shown for a new one.
  const visible = useMemo(
    () =>
      runId === null
        ? null
        : feed?.runId === runId
          ? feed
          : initialRunFeed(runId),
    [feed, runId],
  );
  return { feed: visible, readAgain };
}

/** Polls the newest page of live runs, and remembers the newest run when first listed. */
export function useRunList() {
  const [list, setList] = useState<RunListFeed>(initialRunList);
  const [firstNewest, setFirstNewest] = useState<string | null>(null);
  const poller = useRef<ReturnType<typeof startPolling> | null>(null);
  useEffect(() => {
    let current = initialRunList;
    const polling = startPolling(async (signal) => {
      const result = await liveApi.runPage(undefined, signal);
      if (signal.aborted) return null;
      current = result.ok
        ? { page: result.data, connection: { status: "connected" } }
        : {
            ...current,
            connection: connectionAfterFailure(
              current.connection,
              result,
              Math.random(),
            ),
          };
      setList(current);
      if (result.ok)
        setFirstNewest((first) => first ?? result.data.runs[0]?.run_id ?? null);
      return nextListPollDelayMs(current);
    });
    poller.current = polling;
    return () => {
      polling.stop();
      poller.current = null;
    };
  }, []);
  const readAgain = useCallback(() => poller.current?.now(), []);
  return { list, firstNewest, readAgain };
}

const GPU_UNREADABLE: GpuEvidence = {
  missing: "The local GPU diagnostic could not be read.",
};

/**
 * Reads the local GPU diagnostic once per selected run, and again when it closes, since a
 * probe writes its files at the end. Null until the read finishes.
 */
export function useGpuEvidence(runId: string | null, closed: boolean) {
  const key = runId === null ? null : `${runId}:${closed}`;
  const [evidence, setEvidence] = useState<{
    key: string;
    gpu: GpuEvidence;
  } | null>(null);
  useEffect(() => {
    if (key === null || runId === null) return;
    const controller = new AbortController();
    fetch(`/api/live/runs/${encodeURIComponent(runId)}/gpu`, {
      cache: "no-store",
      signal: AbortSignal.any([
        controller.signal,
        AbortSignal.timeout(BROWSER_TIMEOUT_MS),
      ]),
    })
      .then(async (response) => {
        const parsed = gpuEvidenceSchema.safeParse(
          response.ok ? await response.json() : null,
        );
        return parsed.success ? parsed.data : GPU_UNREADABLE;
      })
      .catch(() => GPU_UNREADABLE)
      .then((gpu) => {
        if (!controller.signal.aborted) setEvidence({ key, gpu });
      });
    return () => controller.abort();
  }, [key, runId]);
  return evidence?.key === key ? evidence.gpu : null;
}
