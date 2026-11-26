import { z } from "zod";

/**
 * Helpers for downloading an evidence file in the page (components/evidence-download.tsx)
 * instead of navigating to it, so a failed download is reported in place and never replaces
 * the run page with the route's raw error.
 */

/** A file name only: no path separators or parent references. */
const SAFE_FILENAME = /^[\w.-]{1,200}$/;

/** The name from `Content-Disposition: attachment; filename="…"`, else `fallback`. */
export function downloadFilename(
  disposition: string | null,
  fallback: string,
): string {
  const name = disposition?.match(/filename="([^"]+)"/)?.[1];
  return name && SAFE_FILENAME.test(name) && !name.includes("..")
    ? name
    : fallback;
}

/** The error shape the evidence routes answer with. */
const routeErrorSchema = z.object({ error: z.string().max(300) });

/** A response that was not a file: its status, or null when no response arrived. */
export type DownloadFailure = { status: number | null; body: unknown };

/** Generic wording by status, for a response without the route's own message. */
function genericReason(status: number | null): string {
  if (status === null)
    return "the server could not be reached or did not answer in time";
  if (status === 503) return "the server is unavailable right now";
  if (status === 404) return "it was not found";
  return "the server refused it";
}

function reasonFor({ status, body }: DownloadFailure): string {
  // The route's own message comes first: it knows which service failed (run history, the
  // live API) and what was missing (the run or the file). Only that validated, short
  // message is shown; any other body stays out of the page.
  const routeError = routeErrorSchema.safeParse(body);
  return routeError.success ? routeError.data.error : genericReason(status);
}

/**
 * One signal that aborts when any of `signals` does. Uses AbortSignal.any where the browser
 * has it (not Safari before 17.4 or Firefox before 124) and combines them by hand otherwise,
 * so a download works in either.
 */
export function anySignal(signals: AbortSignal[]): AbortSignal {
  if (typeof AbortSignal.any === "function") return AbortSignal.any(signals);
  const combined = new AbortController();
  for (const signal of signals) {
    if (signal.aborted) {
      combined.abort(signal.reason);
      break;
    }
    signal.addEventListener("abort", () => combined.abort(signal.reason), {
      once: true,
      signal: combined.signal,
    });
  }
  return combined.signal;
}

/** One sentence for the viewer: what failed, why, and that nothing else changed. */
export function downloadFailure(failure: DownloadFailure, file: string) {
  const reason = reasonFor(failure).replace(/\.$/, "");
  const status = failure.status === null ? "" : ` (HTTP ${failure.status})`;
  return `${file} could not be downloaded: ${reason}${status}. The run shown here is unchanged.`;
}
