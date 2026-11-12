import { Gauge } from "lucide-react";
import { ConcurrencyControls } from "@/components/concurrency-controls";
import { WorkspacePage } from "@/components/shell/app-shell";
import { PageHeader } from "@/components/shell/page-header";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";

/**
 * Concurrency controls on the workspace shell. The versioned-policy editor
 * (components/concurrency-controls.tsx) is built from the glass primitives.
 */
export function ControlsPage() {
  return (
    <WorkspacePage>
      <div className="flex flex-col gap-4 fit:h-full fit:min-h-0">
        <PageHeader
          title="Controls"
          description="Versioned manual concurrency limits for coding-agent runs"
          scope="Policy API · changes apply to runs admitted after they are saved"
        />
        <GlassPanel
          as="section"
          aria-labelledby="controls-policy"
          className="min-w-0 fit:min-h-0 fit:flex-1 fit:overflow-y-auto"
        >
          <PanelHeader
            titleId="controls-policy"
            icon={Gauge}
            title="Concurrency policy"
          />
          <ConcurrencyControls />
        </GlassPanel>
      </div>
    </WorkspacePage>
  );
}
