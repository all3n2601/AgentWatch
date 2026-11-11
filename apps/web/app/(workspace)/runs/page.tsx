import type { Metadata } from "next";
import { Suspense } from "react";
import {
  RunsData,
  RunsLoading,
  RunsPageFrame,
} from "@/components/pages/runs/runs-page";
import { insightConfig } from "@/lib/insights";
import { parseRunsQuery, parseRunsView } from "@/lib/runs-page";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Runs",
  description: "Every run, live or saved, in one list",
};

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const query = parseRunsQuery(params);
  return (
    <RunsPageFrame chatEnabled={insightConfig(process.env) !== null}>
      {/* Keyed by the query so a new search or live page remounts the list
          state and shows the loading state while both stores are read. */}
      <Suspense
        key={`${query.liveBefore ?? "newest"}:${query.query}`}
        fallback={<RunsLoading />}
      >
        <RunsData query={query} initialView={parseRunsView(params)} />
      </Suspense>
    </RunsPageFrame>
  );
}
