import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Page title block. `scope` states what the numbers on the page cover
 * ("598 runs · 5,603 measured steps · steps-v1.3"); every page names its scope.
 */
export function PageHeader({
  title,
  description,
  scope,
  actions,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  scope?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-end justify-between gap-4",
        className,
      )}
    >
      <div className="min-w-0">
        <h1 className="text-[1.75rem] leading-tight font-bold tracking-tight text-ink sm:text-4xl">
          {title}
        </h1>
        {description && (
          <p className="mt-1 text-base text-ink-muted sm:text-lg">
            {description}
          </p>
        )}
        {scope && (
          <p className="mt-1 text-base text-ink-muted tabular-nums sm:text-lg">
            {scope}
          </p>
        )}
      </div>
      {actions && (
        <div className="flex flex-wrap items-center gap-2">{actions}</div>
      )}
    </div>
  );
}
