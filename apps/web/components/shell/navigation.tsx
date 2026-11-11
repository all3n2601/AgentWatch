import {
  FlaskConical,
  Gauge,
  GitCompareArrows,
  House,
  Play,
  RadioTower,
  Settings,
  ShieldCheck,
} from "lucide-react";
import type { NavRailItem } from "@/components/shell/nav-rail";
import { SECTION_PATHS } from "@/lib/routes";

/**
 * Every page of the workspace, in the order of the diagnosis workflow
 * (docs/ui-redesign-plan.md section 3). The sidebar always shows all of them.
 */
export const WORKSPACE_NAVIGATION: NavRailItem[] = [
  { href: SECTION_PATHS.overview, label: "Overview", icon: <House /> },
  { href: SECTION_PATHS.runs, label: "Runs", icon: <Play /> },
  {
    href: SECTION_PATHS.experiments,
    label: "Experiments",
    icon: <FlaskConical />,
  },
  { href: SECTION_PATHS.compare, label: "Compare", icon: <GitCompareArrows /> },
  { href: SECTION_PATHS.quality, label: "Data quality", icon: <ShieldCheck /> },
  { href: SECTION_PATHS.live, label: "Live runs", icon: <RadioTower /> },
  { href: SECTION_PATHS.controls, label: "Controls", icon: <Gauge /> },
  {
    href: SECTION_PATHS.settings,
    label: "Settings",
    icon: <Settings />,
    placement: "footer",
  },
];
