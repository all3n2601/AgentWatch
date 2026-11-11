import * as React from "react";
import Link from "next/link";
import { SlidersHorizontal } from "lucide-react";
import { BrandMark } from "@/components/shell/brand-mark";
import { ThemeToggle } from "@/components/theme-toggle";
import { SearchField } from "@/components/ui/search-field";
import { cn } from "@/lib/utils";

const ROUND_ICON =
  "glass-raised flex size-11 items-center justify-center rounded-full text-ink transition-colors hover:bg-glass focus-visible:ring-[3px] focus-visible:ring-brand/40 outline-none";

/**
 * Global header: brand, workspace name, search and global actions. Page
 * specific toggles (scene or graph view) go in `actions`, so the bar never
 * shows a control the current page cannot honour.
 */
export function TopBar({
  workspace,
  searchAction = "/runs",
  actions,
  className,
}: {
  workspace?: string;
  searchAction?: string;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "flex flex-wrap items-center gap-x-6 gap-y-3 px-4 pt-4 sm:px-6 lg:flex-nowrap",
        className,
      )}
    >
      <Link
        href="/"
        className="flex shrink-0 items-center gap-3 rounded-2xl outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
      >
        <BrandMark />
        <span className="text-[1.375rem] font-semibold tracking-tight text-ink">
          AgentWatch
        </span>
      </Link>
      {workspace && (
        <span className="hidden border-l border-line pl-6 text-sm text-ink-muted md:block">
          {workspace}
        </span>
      )}
      <SearchField
        action={searchAction}
        className="order-last basis-full lg:order-none lg:mx-auto lg:max-w-xl lg:basis-auto"
      />
      <div className="ml-auto flex items-center gap-2 lg:ml-0">
        {actions}
        <ThemeToggle
          compact
          className={cn(
            ROUND_ICON,
            "h-11 w-11 border-line bg-glass-strong text-ink",
          )}
        />
        <Link
          href="/settings"
          aria-label="Workspace settings"
          className={ROUND_ICON}
        >
          <SlidersHorizontal className="size-5" aria-hidden />
        </Link>
      </div>
    </header>
  );
}

/** Round glass icon action for `TopBar.actions` and panel corners. */
export function RoundIconButton({
  label,
  className,
  children,
  ...props
}: React.ComponentProps<"button"> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(ROUND_ICON, className)}
      {...props}
    >
      {children}
    </button>
  );
}
