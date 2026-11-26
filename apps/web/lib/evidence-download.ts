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

function reasonFor({ status, body }: DownloadFailure): string {
  if (status === null)
    return "the server could not be reached or did not answer in time";
  if (status === 503) return "Run history is unavailable right now";
  if (status === 404) return "the file was not found";
  const routeError = routeErrorSchema.safeParse(body);
  // Only the route's own short message is shown; any other body stays out of the page.
  return routeError.success ? routeError.data.error : "the server refused it";
}

/** One sentence for the viewer: what failed, why, and that nothing else changed. */
export function downloadFailure(failure: DownloadFailure, file: string) {
  const reason = reasonFor(failure).replace(/\.$/, "");
  const status = failure.status === null ? "" : ` (HTTP ${failure.status})`;
  return `${file} could not be downloaded: ${reason}${status}. The run shown here is unchanged.`;
}
