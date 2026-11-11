import { FlaskConical } from "lucide-react";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { RatioChip } from "@/components/widgets/delta-chip";
import { Measurement } from "@/components/widgets/measurement";
import {
  conditionShortName,
  INJECTED_CONDITIONS,
  type TaskGroupRows,
} from "@/lib/experiments-page";
import { formatQuantity } from "@/lib/measurement-format";
import { formatCv, formatDateTime } from "./format";

const count = (value: number) => formatQuantity(value, "count");
const COLUMNS = 5 + INJECTED_CONDITIONS.length;

/** Every replay experiment of the release, grouped by task, newest first. */
export function ExperimentTable({
  groups,
  experiments,
}: {
  groups: TaskGroupRows[];
  experiments: number;
}) {
  return (
    // On fit screens the panel fills its tab and the table scrolls inside it,
    // under a sticky header, instead of the page.
    <GlassPanel
      as="section"
      aria-labelledby="experiment-list"
      className="fit:flex fit:h-full fit:min-h-0 fit:flex-col"
    >
      <PanelHeader
        titleId="experiment-list"
        icon={FlaskConical}
        title="Experiments"
        description={`${count(experiments)} experiments. Ratios are each condition's replay over that experiment's clean median; a dash means the condition was not replayed. Clean spread is the coefficient of variation of the clean replays.`}
      />
      <div
        role="region"
        aria-label="Experiments table"
        tabIndex={0}
        className="relative -mx-1 max-h-[36rem] overflow-auto rounded-xl px-1 outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:max-h-none fit:min-h-0 fit:flex-1"
      >
        <table className="w-full min-w-[56rem] text-left text-sm text-ink">
          <thead className="sticky top-0 z-10 bg-glass-strong text-[13px] text-ink-muted backdrop-blur">
            <tr className="border-b border-line">
              <th scope="col" className="py-2 pr-3 font-medium">
                Recorded (UTC) · experiment
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Replays
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Clean median
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Clean spread
              </th>
              {INJECTED_CONDITIONS.map((condition) => (
                <th
                  key={condition}
                  scope="col"
                  className="py-2 pr-3 font-medium"
                >
                  {conditionShortName(condition)}
                </th>
              ))}
              <th scope="col" className="py-2 font-medium">
                Task check
              </th>
            </tr>
          </thead>
          {groups.map((group) => (
            <tbody key={group.taskGroup} className="divide-y divide-line">
              <tr>
                <th
                  scope="rowgroup"
                  colSpan={COLUMNS}
                  className="bg-glass-inset py-2 pl-2 text-[15px] font-semibold"
                >
                  {group.taskGroup}{" "}
                  <span className="font-normal text-ink-muted">
                    ({count(group.experiments.length)} experiments)
                  </span>
                </th>
              </tr>
              {group.experiments.map((row) => (
                <tr key={row.id}>
                  <th scope="row" className="py-2 pr-3 font-normal">
                    <span className="block font-medium tabular-nums">
                      {row.recordedAt === null
                        ? "Time not recorded"
                        : formatDateTime(row.recordedAt)}
                    </span>
                    <span
                      className="block font-mono text-xs break-all text-ink-muted"
                      title={row.id}
                    >
                      {row.shortId}
                    </span>
                  </th>
                  <td className="py-2 pr-3 text-right tabular-nums">
                    {count(row.replays)}
                    <span className="block text-xs text-ink-muted">
                      {count(row.cleanReplays)} clean
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-right">
                    <Measurement value={row.cleanMedianSeconds} unit="s" />
                  </td>
                  <td className="py-2 pr-3 text-right tabular-nums">
                    {row.cleanCv === null ? (
                      <Measurement value={null} unit="percent" />
                    ) : (
                      formatCv(row.cleanCv)
                    )}
                  </td>
                  {INJECTED_CONDITIONS.map((condition) => (
                    <td key={condition} className="py-2 pr-3">
                      <RatioChip ratio={row.ratios[condition]} />
                    </td>
                  ))}
                  <td className="py-2">
                    {row.failedReplays > 0 ? (
                      <StatusBadge
                        label={`${count(row.failedReplays)} failed`}
                        tone="warning"
                      />
                    ) : row.unrecordedReplays > 0 ? (
                      <StatusBadge
                        label={`${count(row.unrecordedReplays)} not recorded`}
                        tone="info"
                      />
                    ) : (
                      <StatusBadge label="All passed" tone="success" />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>
    </GlassPanel>
  );
}
