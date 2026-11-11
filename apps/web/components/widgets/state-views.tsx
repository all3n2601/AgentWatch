import * as React from "react";
import {
  CircleDashed,
  Inbox,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

function StateFrame({
  icon: Icon,
  title,
  detail,
  action,
  role,
  className,
}: {
  icon: LucideIcon;
  title: string;
  detail?: React.ReactNode;
  action?: React.ReactNode;
  role?: "status" | "alert";
  className?: string;
}) {
  return (
    <div
      role={role}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-line px-6 py-10 text-center",
        className,
      )}
    >
      <Icon className="size-7 text-ink-subtle" aria-hidden />
      <p className="text-base font-semibold text-ink">{title}</p>
      {detail && <p className="max-w-prose text-sm text-ink-muted">{detail}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

/** Nothing recorded yet: a real, successful empty result. */
export function EmptyState(props: {
  title: string;
  detail?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return <StateFrame icon={Inbox} role="status" {...props} />;
}

/**
 * The evidence could not be read (source down, not instrumented, malformed).
 * Distinct from EmptyState so missing history is never shown as no history.
 */
export function UnavailableState({
  title = "Evidence unavailable",
  reason,
  action,
  className,
}: {
  title?: string;
  reason: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <StateFrame
      icon={CircleDashed}
      role="status"
      title={title}
      detail={reason}
      action={action}
      className={className}
    />
  );
}

/** A request failed; offer a way to recover. */
export function ErrorState({
  title = "Something went wrong",
  detail,
  action,
  className,
}: {
  title?: string;
  detail: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <StateFrame
      icon={TriangleAlert}
      role="alert"
      title={title}
      detail={detail}
      action={action}
      className={className}
    />
  );
}

/** Placeholder block while data loads; announce progress once via `label`. */
export function Skeleton({
  className,
  label,
}: {
  className?: string;
  label?: string;
}) {
  return (
    <div
      role={label ? "status" : undefined}
      aria-label={label}
      className={cn(
        "animate-pulse rounded-xl bg-bar-track motion-reduce:animate-none",
        className,
      )}
    />
  );
}
