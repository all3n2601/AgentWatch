"use client";

import * as React from "react";
import type { VariantProps } from "class-variance-authority";
import { Button, buttonVariants } from "@/components/ui/button";
import { ErrorState } from "@/components/widgets/state-views";
import {
  downloadFailure,
  downloadFilename,
  type DownloadFailure,
} from "@/lib/evidence-download";

/** Long enough for the route's own wait on run history; a stuck download is reported. */
const DOWNLOAD_TIMEOUT_MS = 30_000;

type Download =
  | { status: "idle" }
  | { status: "downloading" }
  | { status: "failed"; message: string };

/** Saves `blob` under `filename` through a temporary link, as a normal download does. */
function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** Reads an error response's JSON body, or null when it has none. */
async function errorBody(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * A link that downloads an evidence file (or a live run's export) in the page. It keeps its
 * `href`, so it can be copied or opened in a new tab and still works without scripts. A plain
 * click fetches the file instead of navigating to it, so a failure (run history down, a file
 * rewritten, the server unreachable) is reported here, and the run page stays as it was.
 */
export function EvidenceDownload({
  href,
  file,
  children,
  "aria-label": ariaLabel,
  variant = "glass",
  size = "pill",
  className,
}: {
  href: string;
  /** The file's name, for the saved name's fallback and for the error message. */
  file: string;
  children: React.ReactNode;
  "aria-label"?: string;
  className?: string;
} & Pick<VariantProps<typeof buttonVariants>, "variant" | "size">) {
  const [download, setDownload] = React.useState<Download>({
    status: "idle",
  });
  const request = React.useRef<AbortController | null>(null);
  React.useEffect(() => () => request.current?.abort(), []);

  async function start() {
    if (download.status === "downloading") return;
    const controller = new AbortController();
    request.current = controller;
    setDownload({ status: "downloading" });
    let failure: DownloadFailure;
    try {
      const response = await fetch(href, {
        cache: "no-store",
        signal: AbortSignal.any([
          controller.signal,
          AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS),
        ]),
      });
      if (response.ok) {
        saveBlob(
          await response.blob(),
          downloadFilename(response.headers.get("Content-Disposition"), file),
        );
        setDownload({ status: "idle" });
        return;
      }
      failure = { status: response.status, body: await errorBody(response) };
    } catch {
      // Aborted because the link unmounted (another run opened): nothing to report.
      if (controller.signal.aborted) return;
      // No response, or none within DOWNLOAD_TIMEOUT_MS.
      failure = { status: null, body: null };
    }
    setDownload({ status: "failed", message: downloadFailure(failure, file) });
  }

  function onClick(event: React.MouseEvent<HTMLAnchorElement>) {
    // Modified clicks keep the browser's behaviour (a new tab or window).
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    )
      return;
    event.preventDefault();
    void start();
  }

  const downloading = download.status === "downloading";
  return (
    <>
      <Button asChild variant={variant} size={size} className={className}>
        <a
          href={href}
          aria-label={ariaLabel}
          aria-busy={downloading || undefined}
          onClick={onClick}
        >
          {children}
        </a>
      </Button>
      {download.status === "failed" && (
        <ErrorState
          title={`${file} not downloaded`}
          detail={download.message}
          className="w-full basis-full py-4"
          action={
            <Button variant="glass" size="pill" onClick={() => void start()}>
              Try again
            </Button>
          }
        />
      )}
    </>
  );
}
