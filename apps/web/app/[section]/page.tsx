import { notFound } from "next/navigation";
import { Dashboard } from "@/components/dashboard";
import { loadRequestedRuns } from "@/lib/experiments";
import { pages, type Page } from "@/lib/dashboard-data";
export const dynamic = "force-dynamic";
export default async function Section({
  params,
  searchParams,
}: {
  params: Promise<{ section: string }>;
  searchParams: Promise<{ run?: string; baseline?: string }>;
}) {
  const { section } = await params;
  if (!pages.includes(section as Page)) notFound();
  const query = await searchParams;
  return (
    <Dashboard
      key={`${section}:${query.run || ""}:${query.baseline || ""}`}
      initialData={await loadRequestedRuns([query.run, query.baseline])}
      page={section as Page}
      initialRunId={query.run}
      initialBaselineId={query.baseline}
    />
  );
}
