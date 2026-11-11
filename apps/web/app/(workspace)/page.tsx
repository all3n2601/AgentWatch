import type { Metadata } from "next";
import {
  ModelOverviewPage,
  type ModelOverviewState,
} from "@/components/model-overview";
import { loadDatasetSnapshot } from "@/lib/dataset-load";
import { loadImportMap } from "@/lib/import-map";
import { insightConfig } from "@/lib/insights";
import { buildModelOverview } from "@/lib/model-overview";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  // The root layout's title template only applies to nested routes.
  title: { absolute: "Overview · AgentWatch" },
  description:
    "One agent model's runs: outcomes, waiting by stage and the largest observed slowdown",
};

async function overviewState(model?: string): Promise<ModelOverviewState> {
  const snapshot = await loadDatasetSnapshot();
  if (snapshot.status !== "ready")
    return { status: "unavailable", reason: snapshot.reason };
  const overview = buildModelOverview(snapshot, await loadImportMap(), model);
  return overview
    ? { status: "ready", overview }
    : { status: "no-models", release: snapshot.manifest.dataset_version };
}

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ model?: string }>;
}) {
  const { model } = await searchParams;
  return (
    <ModelOverviewPage
      state={await overviewState(model)}
      chatEnabled={insightConfig(process.env) !== null}
    />
  );
}
