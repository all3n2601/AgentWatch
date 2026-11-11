import { ScanSearch, Table } from "lucide-react";
import { Chip } from "@/components/ui/chip";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { formatQuantity, MISSING_VALUE } from "@/lib/measurement-format";
import {
  statusText,
  type AucRow,
  type QualityPageData,
} from "@/lib/quality-page";
import { ScrollTable, shareText } from "./count-shares";

const count = (value: number) => formatQuantity(value, "count");
const auc = (value: number) => value.toFixed(3);

const HEAD = "px-3 py-2 text-left text-[13px] font-semibold text-ink-muted";
const CELL = "px-3 py-2 text-sm text-ink";
// The header stays visible while a table scrolls inside its panel; the strong
// glass keeps scrolled rows from showing through it.
const STICKY_HEAD =
  "sticky top-0 z-10 bg-glass-strong shadow-[0_1px_0_var(--color-line)] backdrop-blur";
// On one-screen desktops a table takes the panel's remaining height and
// scrolls inside it; below that it grows with its rows and the page scrolls.
const FILL_SCROLL = "fit:min-h-0 fit:flex-1 fit:overflow-y-auto";
// A panel or card that hands its remaining height to a table on fit screens.
const FILL_COLUMN = "fit:flex fit:min-h-0 fit:flex-col";

function AucTable({
  label,
  rows,
  showClass,
}: {
  label: string;
  rows: readonly AucRow[];
  showClass: boolean;
}) {
  if (rows.length === 0)
    return <p className="text-sm text-ink-muted">No features recorded.</p>;
  return (
    <ScrollTable label={label} className={FILL_SCROLL}>
      <table className="w-full min-w-[20rem] border-collapse">
        <caption className="sr-only">{label}</caption>
        <thead className={STICKY_HEAD}>
          <tr className="border-b border-line">
            {showClass && (
              <th scope="col" className={HEAD}>
                Class
              </th>
            )}
            <th scope="col" className={HEAD}>
              Feature
            </th>
            <th scope="col" className={`${HEAD} text-right`}>
              AUC
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((row) => (
            <tr key={row.id}>
              {showClass && <td className={CELL}>{row.className}</td>}
              <td className={`${CELL} font-mono text-[13px] break-all`}>
                {row.feature}
              </td>
              <td className={`${CELL} text-right tabular-nums`}>
                {auc(row.auc)}
                {row.aboveBound && (
                  <StatusBadge
                    label="Above bound"
                    tone="danger"
                    className="ml-2"
                  />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollTable>
  );
}

export function FeatureSignalPanel({ data }: { data: QualityPageData }) {
  const { leakage, designEncoding } = data;
  return (
    <GlassPanel
      as="section"
      aria-labelledby="quality-signal"
      className={`${FILL_COLUMN} fit:h-full`}
    >
      <PanelHeader
        titleId="quality-signal"
        icon={ScanSearch}
        title="Single-feature AUC"
        description="How well one feature alone separates the label. 0.5 means no signal."
      />
      <div className="grid gap-4 lg:grid-cols-2 fit:min-h-0 fit:flex-1">
        <InsetCard className={FILL_COLUMN}>
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-[15px] font-semibold text-ink">
              Leakage check, strongest per class
            </h3>
            <Chip size="sm">
              {statusText(leakage.status)} · bound {leakage.bound.toFixed(2)}
            </Chip>
          </div>
          <p className="mb-2 text-sm text-ink-muted">
            No single feature may exceed the bound.
          </p>
          <AucTable
            label="Strongest feature per class, AUC"
            rows={leakage.rows}
            showClass
          />
        </InsetCard>
        <InsetCard className={FILL_COLUMN}>
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-[15px] font-semibold text-ink">
              Design encoding
            </h3>
            <Chip size="sm">{statusText(designEncoding.status)}</Chip>
          </div>
          <p className="mb-2 text-sm text-ink-muted">
            Contended against clean-control replay steps labelled no excess, in
            training · {count(designEncoding.rows)} rows. Reported, not a gate.
          </p>
          <AucTable
            label="Design-encoding AUC per feature"
            rows={designEncoding.features}
            showClass={false}
          />
        </InsetCard>
      </div>
    </GlassPanel>
  );
}

export function FieldHealthPanel({ data }: { data: QualityPageData }) {
  const { rows, completeFields } = data.fieldHealth;
  return (
    <GlassPanel
      as="section"
      aria-labelledby="quality-fields"
      className={`${FILL_COLUMN} fit:h-full`}
    >
      <PanelHeader
        titleId="quality-fields"
        icon={Table}
        title="Field health"
        description={`Empty values per field across ${count(data.rows)} steps, highest share first. Empty means not recorded; it is never charted or averaged as zero.`}
        actions={
          <Chip size="sm">
            {count(rows.length)} with empty values · {count(completeFields)}{" "}
            complete
          </Chip>
        }
      />
      {rows.length === 0 ? (
        <p className="text-sm text-ink-muted">
          Every field has a value in every step.
        </p>
      ) : (
        <ScrollTable label="Empty values per field" className={FILL_SCROLL}>
          <table className="w-full min-w-[32rem] border-collapse">
            <caption className="sr-only">
              Empty values per field, highest share first
            </caption>
            <thead className={STICKY_HEAD}>
              <tr className="border-b border-line">
                <th scope="col" className={HEAD}>
                  Field
                </th>
                <th scope="col" className={`${HEAD} text-right`}>
                  Empty
                </th>
                <th scope="col" className={HEAD}>
                  Empty share
                </th>
                <th scope="col" className={`${HEAD} text-right`}>
                  With a value
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {rows.map((row) => (
                <tr key={row.field}>
                  <th
                    scope="row"
                    className={`${CELL} text-left font-mono text-[13px] font-normal break-all`}
                  >
                    {row.field}
                  </th>
                  <td className={`${CELL} text-right tabular-nums`}>
                    {count(row.missing)}
                  </td>
                  <td className={CELL}>
                    {row.share === null ? (
                      <StatusBadge
                        label="Exceeds step count"
                        tone="danger"
                        appearance="plain"
                      />
                    ) : (
                      <span className="flex items-center gap-3">
                        <span className="w-14 shrink-0 text-right tabular-nums">
                          {shareText(row.missing, data.rows)}
                        </span>
                        <span
                          aria-hidden
                          className="h-2 w-full min-w-16 overflow-hidden rounded-full bg-bar-track"
                        >
                          <span
                            className="block h-full rounded-full bg-warning-solid"
                            style={{ width: `${row.share * 100}%` }}
                          />
                        </span>
                      </span>
                    )}
                  </td>
                  <td className={`${CELL} text-right tabular-nums`}>
                    {row.recorded === null
                      ? MISSING_VALUE
                      : count(row.recorded)}
                    {row.recorded === null && (
                      <span className="sr-only">Unavailable</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollTable>
      )}
      <p className="mt-3 text-sm text-ink-muted">
        Many fields apply to one workload kind only, so a high share can be
        expected rather than a fault.
      </p>
    </GlassPanel>
  );
}
