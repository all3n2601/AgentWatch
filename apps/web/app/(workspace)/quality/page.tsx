import type { Metadata } from "next";
import {
  QualityPage,
  type QualityPageState,
} from "@/components/pages/quality/quality-page";
import { loadDatasetSnapshot } from "@/lib/dataset-load";
import { buildQualityPage } from "@/lib/quality-page";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Data quality",
  description: "Telemetry coverage, labels, splits and quality checks",
};

async function qualityState(): Promise<QualityPageState> {
  const snapshot = await loadDatasetSnapshot();
  return snapshot.status === "ready"
    ? { status: "ready", data: buildQualityPage(snapshot) }
    : { status: "unavailable", reason: snapshot.reason };
}

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ view?: string | string[] }>;
}) {
  const { view } = await searchParams;
  return (
    <QualityPage
      state={await qualityState()}
      initialTab={typeof view === "string" ? view : undefined}
    />
  );
}
