import { AssistantChat } from "@/components/assistant-chat";
import { WorkspacePage } from "@/components/shell/app-shell";
import { AssistantRail } from "@/components/shell/assistant-rail";
import { AlertBanner } from "@/components/widgets/alert-banner";
import type { DashboardData } from "@/lib/experiment-schema";
import { buildRunDetail } from "@/lib/run-detail";
import type { RunView } from "@/lib/unified-runs";
import { RunHeader } from "./run-header";
import { RunTabs } from "./run-tabs";

/** Which store answered, stated plainly; the local-file fallback is never called history. */
function sourceLabel(view: RunView, data: DashboardData): string {
  if (view.source === "live") return "Live API";
  return data.source === "local" ? "Local file fallback" : "Saved history";
}

/** One run's page in the glass workspace, whichever store holds it. */
export function RunDetailPage({
  view,
  data,
  initialTab,
  insightsEnabled,
}: {
  view: RunView;
  data: DashboardData;
  initialTab?: string;
  /** Whether the server has an AI endpoint configured (summary and assistant). */
  insightsEnabled: boolean;
}) {
  const model = buildRunDetail(view);
  const shortId = model.runId.slice(0, 8);
  // A live run's page also loads saved history, whose warnings say nothing about this run.
  const warnings = view.source === "saved" ? data.warnings : [];
  return (
    <WorkspacePage
      rail={
        <AssistantRail title="Ask about this run">
          <AssistantChat
            enabled={insightsEnabled}
            suggestions={[
              `Why did run ${shortId} wait?`,
              `Which step of run ${shortId} queued longest?`,
            ]}
          />
        </AssistantRail>
      }
    >
      {/* On fit screens the header and tab bar stay put and the active tab fills the rest. */}
      <div className="space-y-4 fit:flex fit:h-full fit:min-h-0 fit:flex-col fit:gap-3 fit:space-y-0">
        <RunHeader
          model={model}
          view={view}
          sourceLabel={sourceLabel(view, data)}
        />
        {warnings.length > 0 && (
          <AlertBanner
            tone="warning"
            title="Some history could not be read"
            detail={warnings.join(" ")}
            headingLevel="h2"
            className="fit:shrink-0"
          />
        )}
        <RunTabs
          view={view}
          model={model}
          initialTab={initialTab}
          insightsEnabled={insightsEnabled}
        />
      </div>
    </WorkspacePage>
  );
}
