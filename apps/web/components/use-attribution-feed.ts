import { useEffect, useMemo, useRef, useState } from "react";
import { startPolling } from "@/components/use-live-feed";
import {
  applyAttributionFailure,
  applyAttributionPage,
  beginDueReread,
  initialAttributionFeed,
  nextAttributionDelayMs,
  nextReadAfter,
  type AttributionFeed,
} from "@/lib/attribution-poll";
import { attributionRow, type AttributionRow } from "@/lib/attribution-view";
import { createLiveApi, LIVE_API_TIMEOUT_MS } from "@/lib/live-api";

// Polls this app's attributions route for the selected run. The component using it is mounted
// only while the Attribution tab is open, so polling, including the slow re-read from
// `after=0`, starts when the tab opens and stops when it closes.

const liveApi = createLiveApi("/api/live/runs", {
  timeoutMs: LIVE_API_TIMEOUT_MS + 2_000,
});

/** `runOpen` is false once the live run has closed; only the slow re-read continues then. */
export function useAttributionFeed(runId: string, runOpen: boolean) {
  const [feed, setFeed] = useState<AttributionFeed>(() =>
    initialAttributionFeed(runId),
  );
  const open = useRef(runOpen);
  useEffect(() => {
    open.current = runOpen;
  }, [runOpen]);

  useEffect(() => {
    let current = initialAttributionFeed(runId);
    const polling = startPolling(async (signal) => {
      current = beginDueReread(current, performance.now());
      const result = await liveApi.attributions(runId, nextReadAfter(current), {
        signal,
      });
      // A response for a previous selection or a closed tab is never applied.
      if (signal.aborted) return null;
      current = result.ok
        ? applyAttributionPage(current, result.data, Math.random())
        : applyAttributionFailure(current, result, Math.random());
      setFeed(current);
      return nextAttributionDelayMs(current, open.current, performance.now());
    });
    return () => polling.stop();
  }, [runId]);

  const visible = feed.runId === runId ? feed : initialAttributionFeed(runId);
  const servingKey = visible.model?.serving_key ?? null;
  const rows: AttributionRow[] = useMemo(
    () =>
      servingKey === null
        ? []
        : visible.items.map((item) => attributionRow(item, runId, servingKey)),
    [visible.items, runId, servingKey],
  );
  return { feed: visible, rows };
}
