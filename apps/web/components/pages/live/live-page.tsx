import { LiveFeed } from "@/components/live-feed";
import { WorkspacePage } from "@/components/shell/app-shell";
import { PageHeader } from "@/components/shell/page-header";

/**
 * Live runs: the newest runs from the live API beside the selected run's Status, Timeline,
 * Resources and Evidence tabs. `requested` is the `?liveRun=` UUID, or null to follow the
 * newest run.
 */
export function LivePage({ requested }: { requested: string | null }) {
  return (
    <WorkspacePage>
      <div className="flex flex-col gap-4 fit:h-full fit:min-h-0 fit:gap-3">
        <PageHeader
          title="Live runs"
          scope="Read from the live API while this page is open · polling pauses while the tab is hidden"
        />
        <div className="min-w-0 fit:min-h-0 fit:flex-1">
          <LiveFeed requested={requested} />
        </div>
      </div>
    </WorkspacePage>
  );
}
