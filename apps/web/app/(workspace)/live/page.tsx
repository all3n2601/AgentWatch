import type { Metadata } from "next";
import { LivePage } from "@/components/pages/live/live-page";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Live runs",
  description: "Active steps, failures and feed status for runs in progress",
};

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ liveRun?: string }>;
}) {
  const { liveRun } = await searchParams;
  return <LivePage requested={liveRun ?? null} />;
}
