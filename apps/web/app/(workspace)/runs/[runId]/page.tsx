import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { RunDetailPage } from "@/components/pages/run-detail/run-detail-page";
import { insightConfig } from "@/lib/insights";
import { loadRunView } from "@/lib/run-load";
export const dynamic = "force-dynamic";

/** The tab names the run by its short id; the page itself still loads the run. */
export async function generateMetadata({
  params,
}: {
  params: Promise<{ runId: string }>;
}): Promise<Metadata> {
  const { runId } = await params;
  return { title: `Run ${runId.slice(0, 8)}` };
}

/**
 * One run's page, whichever store holds it. An unknown UUID is a 404; no other run is shown.
 * The open tab (`?tab=`) is read by the tabs themselves.
 */
export default async function RunRoute({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const loaded = await loadRunView(runId);
  if (!loaded) notFound();
  return (
    <RunDetailPage
      key={runId}
      view={loaded.view}
      data={loaded.data}
      insightsEnabled={insightConfig(process.env) !== null}
    />
  );
}
