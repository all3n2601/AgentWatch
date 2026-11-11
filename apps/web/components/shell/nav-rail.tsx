"use client";

import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { usePreferences } from "@/components/shell/use-preferences";
import { cn } from "@/lib/utils";

export interface NavRailItem {
  href: Route;
  label: string;
  /**
   * A rendered icon element (`<House />`), not a component: the rail is a
   * client component and server layouts can only pass it serializable props.
   */
  icon: ReactNode;
  /** Pinned to the rail's foot (settings, help). */
  placement?: "main" | "footer";
}

function isActive(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

function RailLink({
  item,
  active,
  expanded,
}: {
  item: NavRailItem;
  active: boolean;
  expanded: boolean;
}) {
  return (
    <Link
      href={item.href}
      aria-current={active ? "page" : undefined}
      // Collapsed, the label is the accessible name and the native tooltip.
      aria-label={expanded ? undefined : item.label}
      title={expanded ? undefined : item.label}
      className={cn(
        "relative flex h-11 shrink-0 items-center gap-3 rounded-2xl text-sm font-medium whitespace-nowrap text-ink transition-colors outline-none hover:bg-glass-strong",
        "focus-visible:ring-[3px] focus-visible:ring-brand/40",
        expanded
          ? "w-11 justify-center md:w-full md:justify-start md:px-3"
          : "w-11 justify-center",
        active &&
          "border border-success-line bg-brand-soft text-brand-ink shadow-glow hover:bg-brand-soft",
      )}
    >
      <span aria-hidden className="shrink-0 [&_svg]:size-5">
        {item.icon}
      </span>
      {/* Narrow screens show icons only; the label stays the accessible name. */}
      {expanded && <span className="max-md:sr-only">{item.label}</span>}
    </Link>
  );
}

/**
 * The workspace sidebar. The workspace layout renders it once, so it keeps its
 * place and size on every page and only the active item changes. It lists
 * every page. "Collapsed" (Settings → Sidebar) shows icons only; on narrow
 * screens it is a scrolling bar under the header.
 */
export function NavRail({
  items,
  className,
}: {
  items: readonly NavRailItem[];
  className?: string;
}) {
  const pathname = usePathname();
  const { preferences } = usePreferences();
  const expanded = preferences.sidebar === "expanded";
  const main = items.filter((item) => item.placement !== "footer");
  const footer = items.filter((item) => item.placement === "footer");
  return (
    <nav
      aria-label="Main navigation"
      className={cn(
        "glass-panel flex gap-1.5 overflow-x-auto rounded-3xl p-1.5 md:flex-col md:justify-between md:overflow-x-visible md:overflow-y-auto md:py-3",
        expanded && "md:w-48",
        className,
      )}
    >
      <ul className="flex gap-1.5 md:flex-col">
        {main.map((item) => (
          <li key={item.href}>
            <RailLink
              item={item}
              active={isActive(pathname, item.href)}
              expanded={expanded}
            />
          </li>
        ))}
      </ul>
      {footer.length > 0 && (
        <ul className="ml-auto flex gap-1.5 border-l border-line pl-1.5 md:ml-0 md:flex-col md:border-t md:border-l-0 md:pt-1.5 md:pl-0">
          {footer.map((item) => (
            <li key={item.href}>
              <RailLink
                item={item}
                active={isActive(pathname, item.href)}
                expanded={expanded}
              />
            </li>
          ))}
        </ul>
      )}
    </nav>
  );
}
