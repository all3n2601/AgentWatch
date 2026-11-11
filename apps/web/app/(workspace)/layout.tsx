import type { ReactNode } from "react";
import { WorkspaceFrame } from "@/components/shell/app-shell";

/**
 * Every workspace page shares this layout, so the header and sidebar are
 * rendered once and stay put while pages change.
 */
export default function WorkspaceLayout({ children }: { children: ReactNode }) {
  return <WorkspaceFrame>{children}</WorkspaceFrame>;
}
