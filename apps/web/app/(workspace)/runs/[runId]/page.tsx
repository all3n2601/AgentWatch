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

/** One run's page, whichever store holds it. An unknown UUID is a 404; no other run is shown. */
export default async function RunRoute({
  params,
  searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ tab?: string }>;
}) {
  const { runId } = await params;
  const { tab } = await searchParams;
  const loaded = await loadRunView(runId);
  if (!loaded) notFound();
  return (
    <RunDetailPage
      key={runId}
      view={loaded.view}
      data={loaded.data}
      initialTab={tab}
      insightsEnabled={insightConfig(process.env) !== null}
    />
  );
}
