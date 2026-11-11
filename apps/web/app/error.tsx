"use client";

import Link from "next/link";
import { TriangleAlert } from "lucide-react";
import { routes } from "@/lib/routes";

/**
 * A page failed to render. The message is generic on purpose: server error
 * details stay in the server log, and `digest` lets an operator find them.
 */
export default function RouteError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="aw-canvas grid min-h-dvh place-items-center p-6">
      <div
        role="alert"
        className="glass-panel w-full max-w-md space-y-4 p-6 text-center"
      >
        <TriangleAlert className="mx-auto size-8 text-danger" aria-hidden />
        <h1 className="text-2xl font-semibold tracking-tight text-ink">
          This page could not load
        </h1>
        <p className="text-sm text-ink-muted">
          A data source may be unavailable. Try again, or open another page.
          {error.digest && (
            <span className="mt-2 block font-mono text-xs">
              Reference {error.digest}
            </span>
          )}
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          <button
            type="button"
            onClick={reset}
            className="inline-flex h-11 items-center rounded-full bg-brand-strong px-5 text-sm font-medium text-white outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
          >
            Try again
          </button>
          <Link
            href={routes.section("overview")}
            className="glass-raised inline-flex h-11 items-center rounded-full px-5 text-sm font-medium text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
          >
            Go to Overview
          </Link>
        </div>
      </div>
    </main>
  );
}
