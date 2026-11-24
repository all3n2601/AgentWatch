// Viewports the F04 audit checks (apps/web/AGENTS.md: ~360–390 px mobile, 768 px tablet,
// 1280–1440 px desktop and 200% zoom). Shared by the Playwright config and the evidence run.

export type AuditViewport = {
  name: string;
  viewport: { width: number; height: number };
  /** Phone emulation: touch input and the mobile meta-viewport. */
  mobile: boolean;
  deviceScaleFactor?: number;
};

export const AUDIT_VIEWPORTS: readonly AuditViewport[] = [
  { name: "mobile-360", viewport: { width: 360, height: 740 }, mobile: true },
  { name: "mobile-390", viewport: { width: 390, height: 844 }, mobile: true },
  { name: "tablet-768", viewport: { width: 768, height: 1024 }, mobile: true },
  // Just below the one-screen `fit` layout, which needs at least 1280×760.
  {
    name: "desktop-1280x720",
    viewport: { width: 1280, height: 720 },
    mobile: false,
  },
  {
    name: "desktop-1280x800",
    viewport: { width: 1280, height: 800 },
    mobile: false,
  },
  {
    name: "desktop-1440x900",
    viewport: { width: 1440, height: 900 },
    mobile: false,
  },
  // A 1280×800 window at 200% browser zoom lays out as 640×400 CSS pixels.
  {
    name: "zoom-200",
    viewport: { width: 640, height: 400 },
    mobile: false,
    deviceScaleFactor: 2,
  },
];
