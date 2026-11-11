import type { Metadata } from "next";
import { ExperimentsPage } from "@/components/pages/experiments/experiments-page";
import { loadExperimentsPage } from "@/lib/experiments-page-load";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Experiments",
  description: "Replays under injected contention, against their baseline",
};

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ view?: string | string[] }>;
}) {
  const { view } = await searchParams;
  return (
    <ExperimentsPage
      state={await loadExperimentsPage()}
      initialTab={typeof view === "string" ? view : undefined}
    />
  );
}
