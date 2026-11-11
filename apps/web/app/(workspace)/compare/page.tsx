import type { Metadata } from "next";
import { ComparePageView } from "@/components/pages/compare/compare-page";
import { buildComparePage, compareTab } from "@/lib/compare-page";
import { loadRequestedRuns } from "@/lib/experiments";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Compare runs",
  description: "Timing differences between recorded workloads",
};

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  // A repeated query key is ambiguous, so it selects nothing rather than a guess.
  const single = (value: string | string[] | undefined) =>
    typeof value === "string" ? value : undefined;
  const requested = {
    run: single(query.run),
    baseline: single(query.baseline),
  };
  const data = await loadRequestedRuns([requested.run, requested.baseline]);
  return (
    <ComparePageView
      model={buildComparePage(data, requested)}
      initialTab={compareTab(single(query.view))}
    />
  );
}
