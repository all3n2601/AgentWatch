import { WorkspacePage } from "@/components/shell/app-shell";
import { PageHeader } from "@/components/shell/page-header";
import { SettingsForm, type SettingsConnection } from "./settings-form";

/**
 * Workspace settings: browser display preferences and read-only server
 * status, one tab per section so each fits a desktop screen without scrolling.
 */
export function SettingsPage({
  connection,
  initialTab,
}: {
  connection: SettingsConnection;
  initialTab?: string;
}) {
  return (
    <WorkspacePage>
      <SettingsForm
        connection={connection}
        initialTab={initialTab}
        header={
          <PageHeader
            title="Workspace settings"
            description="Configure how AgentWatch looks, refreshes and connects to evidence"
            scope="Display preferences are saved in this browser · connection checked when the page loads"
          />
        }
      />
    </WorkspacePage>
  );
}
