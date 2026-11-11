import type { Metadata } from "next";
import { ControlsPage } from "@/components/pages/controls/controls-page";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Controls",
  description: "Versioned manual concurrency limits for coding-agent runs",
};

export default function Page() {
  return <ControlsPage />;
}
