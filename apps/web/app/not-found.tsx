import type { Metadata } from "next";
import Link from "next/link";
import { CircleDashed } from "lucide-react";
import { routes } from "@/lib/routes";

export const metadata: Metadata = { title: "Page not found" };

/** Unknown URLs, and runs no store holds (`notFound()` in the run route). */
export default function NotFound() {
  return (
    <main className="aw-canvas grid min-h-dvh place-items-center p-6">
      <div className="glass-panel w-full max-w-md space-y-4 p-6 text-center">
        <CircleDashed className="mx-auto size-8 text-ink-subtle" aria-hidden />
        <h1 className="text-2xl font-semibold tracking-tight text-ink">
          Page not found
        </h1>
        <p className="text-sm text-ink-muted">
          This address does not match a page, or the run it names is not in the
          live or saved history. No other run is shown in its place.
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          <Link
            href={routes.section("overview")}
            className="inline-flex h-11 items-center rounded-full bg-brand-strong px-5 text-sm font-medium text-white outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
          >
            Go to Overview
          </Link>
          <Link
            href={routes.section("runs")}
            className="glass-raised inline-flex h-11 items-center rounded-full px-5 text-sm font-medium text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
          >
            Browse runs
          </Link>
        </div>
      </div>
    </main>
  );
}
