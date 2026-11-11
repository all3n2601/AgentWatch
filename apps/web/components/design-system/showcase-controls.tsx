"use client";

import { useState } from "react";
import { Boxes, Cpu, Layers } from "lucide-react";
import { FieldRow, FieldSection, GlassInput } from "@/components/ui/field-row";
import { FilterSelect } from "@/components/ui/filter-select";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { Switch } from "@/components/ui/switch";

const DENSITY = [
  { value: "comfortable", label: "Comfortable" },
  { value: "compact", label: "Compact" },
] as const;
const REFRESH = [
  { value: "manual", label: "Manual" },
  { value: "15", label: "15 sec" },
  { value: "30", label: "30 sec" },
  { value: "60", label: "60 sec" },
] as const;
const WORKLOADS = [
  { value: "fix-window-count", label: "fix-window-count" },
  { value: "index-docs", label: "index-docs" },
];
const RESOURCES = [
  { value: "cpu", label: "CPU sandbox" },
  { value: "inference", label: "Inference" },
  { value: "retrieval", label: "Retrieval" },
];

/** Local-state demos of the interactive primitives; nothing is persisted. */
export function ShowcaseFilters() {
  const [workload, setWorkload] = useState<string>();
  const [resource, setResource] = useState<string>();
  const [task, setTask] = useState<string>();
  return (
    <div className="flex flex-wrap gap-3">
      <FilterSelect
        label="Workload"
        icon={Boxes}
        value={workload}
        onValueChange={setWorkload}
        options={WORKLOADS}
      />
      <FilterSelect
        label="Resource"
        icon={Cpu}
        value={resource}
        onValueChange={setResource}
        options={RESOURCES}
      />
      <FilterSelect
        label="Task"
        caption
        icon={Layers}
        value={task}
        onValueChange={setTask}
        options={WORKLOADS}
      />
    </div>
  );
}

export function ShowcaseSettingsForm() {
  const [density, setDensity] =
    useState<(typeof DENSITY)[number]["value"]>("comfortable");
  const [refresh, setRefresh] =
    useState<(typeof REFRESH)[number]["value"]>("manual");
  const [summaries, setSummaries] = useState(false);
  return (
    <div className="max-w-2xl space-y-5">
      <FieldSection title="Workspace" id="ds-workspace">
        <FieldRow label="Workspace name" htmlFor="ds-workspace-name">
          <GlassInput id="ds-workspace-name" defaultValue="AgentWatch local" />
        </FieldRow>
      </FieldSection>
      <FieldSection title="Display" id="ds-display">
        <FieldRow label="Density" labelId="ds-density">
          <SegmentedControl
            labelledBy="ds-density"
            value={density}
            onValueChange={setDensity}
            options={DENSITY}
          />
        </FieldRow>
        <FieldRow label="Refresh" labelId="ds-refresh">
          <SegmentedControl
            labelledBy="ds-refresh"
            value={refresh}
            onValueChange={setRefresh}
            options={REFRESH}
          />
        </FieldRow>
      </FieldSection>
      <FieldSection title="Insights" id="ds-insights">
        <FieldRow
          label="AI summaries"
          labelId="ds-summaries"
          hint="Only validated aggregates are sent."
        >
          <Switch
            aria-labelledby="ds-summaries"
            checked={summaries}
            onCheckedChange={setSummaries}
          />
        </FieldRow>
      </FieldSection>
    </div>
  );
}
