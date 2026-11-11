import Link from "next/link";
import type { Route } from "next";
import { ExternalLink, FileText } from "lucide-react";
import { StatusBadge } from "@/components/ui/status-badge";

const SHORT_ID_LENGTH = 8;

/**
 * One run as a compact record: short id, status, title and a link to its run
 * page. Without a page (a dataset run with no imported live run) it says so
 * instead of linking somewhere else.
 */
export function RunRecordRow({
  runId,
  title,
  status,
  href,
}: {
  runId: string;
  title: string;
  status: string;
  href: Route | null;
}) {
  const shortId = runId.slice(0, SHORT_ID_LENGTH);
  const label = status.charAt(0).toUpperCase() + status.slice(1);
  return (
    <li className="glass-inset flex items-center gap-3 px-3 py-2.5">
      <FileText className="size-5 shrink-0 text-ink" aria-hidden />
      {/* Two lines so the id never shares its width with the badge in a narrow rail. */}
      <span className="min-w-0 flex-1 space-y-1">
        <span
          className="block font-mono text-sm font-semibold text-ink"
          title={runId}
        >
          {shortId}
          <span className="sr-only"> (run {runId})</span>
        </span>
        <span className="flex min-w-0 items-center gap-2">
          <StatusBadge label={label} className="py-0 text-xs" />
          <span className="truncate text-xs text-ink-muted" title={title}>
            {title}
          </span>
        </span>
      </span>
      {href ? (
        <Link
          href={href}
          className="glass-raised inline-flex h-10 shrink-0 items-center gap-1.5 rounded-xl px-3 text-sm font-medium text-ink outline-none hover:bg-glass focus-visible:ring-[3px] focus-visible:ring-brand/40"
        >
          <ExternalLink className="size-4" aria-hidden />
          Open record
          <span className="sr-only"> for run {shortId}</span>
        </Link>
      ) : (
        <span
          className="shrink-0 text-xs text-ink-subtle"
          title="Import this run to open its record"
        >
          Not imported
        </span>
      )}
    </li>
  );
}
