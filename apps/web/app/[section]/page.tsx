import { notFound } from "next/navigation";
import { Dashboard } from "@/components/dashboard";
import { loadRequestedRuns } from "@/lib/experiments";
import { liveReplayData } from "@/lib/live-replay";
import { pages, type Page } from "@/lib/dashboard-data";
export const dynamic = "force-dynamic";
export default async function Section({
  params,
  searchParams,
}: {
  params: Promise<{ section: string }>;
  searchParams: Promise<{
    run?: string;
    baseline?: string;
    replay?: string;
    liveRun?: string;
  }>;
}) {
  const { section } = await params;
  if (!pages.includes(section as Page)) notFound();
  const query = await searchParams;
  const live =
    section === "live"
      ? liveReplayData(query.replay, {
          // Synthetic replays must never appear in a production build.
          replayEnabled: process.env.NODE_ENV !== "production",
        })
      : undefined;
  if (live && !query.replay)
    live.liveFeed = { requested: query.liveRun ?? null };
  return (
    <Dashboard
      key={`${section}:${query.run || ""}:${query.baseline || ""}:${query.replay || ""}:${query.liveRun || ""}`}
      initialData={await loadRequestedRuns([query.run, query.baseline])}
      page={section as Page}
      initialRunId={query.run}
      initialBaselineId={query.baseline}
      live={live}
    />
  );
}
