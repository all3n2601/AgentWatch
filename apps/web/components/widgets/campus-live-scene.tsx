"use client";

import dynamic from "next/dynamic";
import Image from "next/image";
import {
  ArrowLeft,
  CircleCheck,
  Cpu,
  FileText,
  Layers,
  Pause,
  Play,
  Settings,
  X,
  type LucideIcon,
} from "lucide-react";
import {
  type CSSProperties,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import type { ScenePalette } from "@/components/widgets/campus-canvas";
import { StatusBadge, TONE_DOT } from "@/components/ui/status-badge";
import { KeyValueList } from "@/components/widgets/evidence-lists";
import type { RoadTraffic, StageId, StageStatus } from "@/lib/campus-status";
import {
  componentModels,
  OVERVIEW_SCENE_IMAGES,
  campus3d,
  overviewScene,
  SCENE_ANCHOR_IDS,
} from "@/lib/overview-scene";
import { TONES, type Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

const CampusCanvas = dynamic(
  () => import("@/components/widgets/campus-canvas"),
  { ssr: false },
);

const STAGE_ICONS: Record<StageId, LucideIcon> = {
  gateway: FileText,
  model: Cpu,
  retrieval: Layers,
  cpuTool: Settings,
  complete: CircleCheck,
};

/** Status words for the detail card; they describe emphasis, not a diagnosis. */
const TONE_WORDS: Record<Tone, string> = {
  success: "Within threshold",
  warning: "Attention",
  caution: "Degraded",
  danger: "Severe",
  info: "Entry point",
};

/**
 * Content floating over the scene in the one-screen layout: above the canvas,
 * and transparent to the pointer except for its own controls.
 */
const FLOATING =
  "fit:relative fit:z-10 fit:pointer-events-none fit:[&_a]:pointer-events-auto fit:[&_button]:pointer-events-auto fit:[&_[data-slot=glass-panel]]:pointer-events-auto";

/** Campus-wide overlays fade in and out with the open stage. */
const CAMPUS_ONLY = "transition-opacity duration-300";
/** In the one-screen layout, overlays that would cover an opened stage. */
const STEPPED_ASIDE = "fit:pointer-events-none fit:opacity-0";

/** Anchors this close to either edge (as a share of width) open their card inward. */
const EDGE_ANCHOR_SHARE = 0.72;

const SCENE_ALT =
  "Campus of the agent pipeline: runs enter at the agent gateway, call the model runtime, loop through retrieval and the CPU tool, and leave at completion.";

// ------------------------------------------------------------- browser state

function mediaStore(query: string) {
  return {
    subscribe(onChange: () => void) {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    get: () => window.matchMedia(query).matches,
  };
}
const reducedMotion = mediaStore("(prefers-reduced-motion: reduce)");
/** Must match `@custom-variant fit` in app/design-system.css. */
const fitLayout = mediaStore("(min-width: 80rem) and (min-height: 47.5rem)");
/** The docked stage card's width plus its margin, px (`w-72` + `right-2`). */
const DOCKED_CARD_PX = 296;

let webglSupport: boolean | undefined;
function supportsWebGL() {
  if (webglSupport === undefined) {
    try {
      webglSupport = Boolean(
        document.createElement("canvas").getContext("webgl2"),
      );
    } catch {
      webglSupport = false;
    }
  }
  return webglSupport;
}
const noSubscribe = () => () => {};

function subscribeTheme(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return () => observer.disconnect();
}
const themeSnapshot = (): "light" | "dark" =>
  document.documentElement.classList.contains("dark") ? "dark" : "light";

/** `theme` is part of the result so the palette is re-read whenever the theme class changes. */
function readPalette(theme: "light" | "dark"): ScenePalette {
  const style = getComputedStyle(document.documentElement);
  const token = (name: string) => style.getPropertyValue(name).trim();
  return {
    theme,
    structure: token("--aw-scene-structure"),
    shade: token("--aw-scene-shade"),
    recess: token("--aw-scene-recess"),
    glass: token("--aw-scene-glass"),
    road: token("--aw-scene-road"),
    ground: token("--aw-scene-ground"),
    bead: token("--aw-scene-bead"),
    brand: token("--aw-brand"),
    tones: Object.fromEntries(
      TONES.map((tone) => [tone, token(`--aw-${tone}-solid`)]),
    ) as Record<Tone, string>,
  };
}

// ------------------------------------------------------------- pieces

function StageCard({
  stage,
  onClose,
}: {
  stage: StageStatus;
  onClose?: () => void;
}) {
  return (
    <div className="glass-panel w-full max-w-xs space-y-3 rounded-2xl bg-surface! p-4 shadow-raised">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-base font-semibold text-ink">{stage.name}</p>
          <p className="mt-0.5 text-sm text-ink-muted">{stage.summary}</p>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label={`Close ${stage.name} details`}
            className="glass-raised flex size-11 shrink-0 items-center justify-center rounded-full text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
          >
            <X className="size-4" aria-hidden />
          </button>
        )}
      </div>
      <StatusBadge
        label={stage.tone ? TONE_WORDS[stage.tone] : "No data"}
        tone={stage.tone ?? "info"}
      />
      <KeyValueList
        dense
        items={stage.facts.map((fact) => ({
          id: fact.label,
          label: fact.label,
          value: fact.value,
        }))}
      />
    </div>
  );
}

function Legend({
  releaseName,
  interactive,
  control,
  className,
}: {
  releaseName: string;
  interactive: boolean;
  className?: string;
  /** The play/pause control, when traffic is animated. */
  control?: ReactNode;
}) {
  const items = [
    {
      id: "normal",
      label: "Step",
      className: "bg-[var(--aw-scene-bead)] border border-line",
    },
    {
      id: "excess",
      label: "Step with excess wait",
      className: "bg-caution-solid",
    },
    { id: "passed", label: "Passed run", className: "bg-success-solid" },
    { id: "failed", label: "Failed run", className: "bg-danger-solid" },
  ];
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-ink-muted fit:max-w-[38rem]",
        FLOATING,
        className,
      )}
    >
      <ul
        className="flex flex-wrap gap-x-4 gap-y-1.5"
        aria-label="Traffic legend"
      >
        {items.map((item) => (
          <li key={item.id} className="flex items-center gap-1.5 text-ink">
            <i
              aria-hidden
              className={cn("size-2.5 rounded-full", item.className)}
            />
            {item.label}
          </li>
        ))}
      </ul>
      <p className="min-w-0">
        Recorded in {releaseName}, not live traffic
        {interactive && " · select a building to open it"}
      </p>
      {control}
    </div>
  );
}

// ------------------------------------------------------------- scene

/**
 * The Overview's interactive campus. The static render paints first (and stays
 * when WebGL is unavailable or the model fails to load); the 3D canvas then
 * takes over with status halos and traffic computed from measured counts.
 */
export function CampusLiveScene({
  stages,
  traffic,
  labels,
  releaseName,
  overlay,
  header,
  className,
}: {
  stages: Record<StageId, StageStatus>;
  traffic: Record<string, RoadTraffic>;
  /** Label text per stage; the model label carries the model name and run count. */
  labels: Record<StageId, string>;
  releaseName: string;
  overlay?: ReactNode;
  /** Page title block; in the one-screen layout it floats over the scene. */
  header?: ReactNode;
  className?: string;
}) {
  const webgl = useSyncExternalStore(noSubscribe, supportsWebGL, () => false);
  const reduced = useSyncExternalStore(
    reducedMotion.subscribe,
    reducedMotion.get,
    () => false,
  );
  const fit = useSyncExternalStore(
    fitLayout.subscribe,
    fitLayout.get,
    () => false,
  );
  const theme = useSyncExternalStore(
    subscribeTheme,
    themeSnapshot,
    () => "light" as const,
  );
  const palette = useMemo(
    () => (webgl ? readPalette(theme) : null),
    [webgl, theme],
  );

  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [paused, setPaused] = useState(false);
  const [visible, setVisible] = useState(true);
  const [hovered, setHovered] = useState<StageId | null>(null);
  const [selected, setSelected] = useState<StageId | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const labelRefs = useRef<Partial<Record<StageId, HTMLElement | null>>>({});

  // Stop rendering frames while the scene is scrolled out of view.
  useEffect(() => {
    const element = frame.current;
    if (!element) return;
    const observer = new IntersectionObserver(([entry]) =>
      setVisible(entry.isIntersecting),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!selected) return;
    const close = (event: KeyboardEvent) =>
      event.key === "Escape" && setSelected(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [selected]);

  const onReady = useCallback(() => setReady(true), []);
  const onError = useCallback(() => setFailed(true), []);
  // Labels live in the right-aligned art box; canvas pixels are frame-relative.
  const artOffsetPx = useRef(0);
  const art = useRef<HTMLDivElement>(null);
  const [frameWidth, setFrameWidth] = useState(0);
  useEffect(() => {
    const frameElement = frame.current;
    const artElement = art.current;
    if (!frameElement || !artElement) return;
    const measure = () => {
      artOffsetPx.current = frameElement.clientWidth - artElement.clientWidth;
      setFrameWidth(frameElement.clientWidth);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(frameElement);
    observer.observe(artElement);
    return () => observer.disconnect();
  }, []);
  const onProject = useCallback((id: StageId, x: number, y: number) => {
    const element = labelRefs.current[id];
    if (!element) return;
    element.style.left = `${x - artOffsetPx.current}px`;
    element.style.top = `${y}px`;
  }, []);
  const toggle = useCallback(
    (id: StageId) => setSelected((current) => (current === id ? null : id)),
    [],
  );
  const clearSelection = useCallback(() => setSelected(null), []);

  const live = webgl && !failed && palette !== null;
  // Where an opened stage may sit: the whole canvas, minus the docked card in
  // the one-screen layout (the callouts fade out while a stage is open).
  const focusRegion = useMemo(() => {
    const margin = 0.04;
    const cardShare =
      fit && frameWidth > 0 ? (2 * DOCKED_CARD_PX) / frameWidth : 0;
    return { x0: -1 + margin, x1: 1 - margin - cardShare };
  }, [fit, frameWidth]);
  const animate = live && ready && visible && !paused && !reduced;
  const shown = selected ?? hovered;
  const stageTones = useMemo(
    () =>
      Object.fromEntries(
        SCENE_ANCHOR_IDS.map((id) => [id, stages[id].tone]),
      ) as Record<StageId, Tone | null>,
    [stages],
  );
  const { widthPx, heightPx } = overviewScene.image;
  const pauseControl =
    live && !reduced ? (
      <button
        type="button"
        onClick={() => setPaused((value) => !value)}
        aria-pressed={paused}
        className="glass-raised inline-flex h-9 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
      >
        {paused ? (
          <Play className="size-4" aria-hidden />
        ) : (
          <Pause className="size-4" aria-hidden />
        )}
        {paused ? "Play traffic" : "Pause traffic"}
      </button>
    ) : null;

  return (
    // One DOM order, two layouts. Stacked (narrow or short screens): header,
    // scene, card, callouts, legend. `fit` (the one-screen desktop layout): the
    // scene fills the hero behind a left column of header, legend and callouts.
    <div
      className={cn(
        "@container relative flex flex-col gap-3 overflow-x-clip fit:h-full fit:gap-2",
        className,
      )}
    >
      {header && <div className={cn("order-1", FLOATING)}>{header}</div>}
      <div
        ref={frame}
        className="relative order-2 aspect-(--scene-aspect) fit:absolute fit:inset-x-0 fit:top-14 fit:bottom-0 fit:aspect-auto"
        style={{ "--scene-aspect": campus3d.camera.aspect } as CSSProperties}
      >
        <div
          ref={art}
          className="pointer-events-none absolute inset-y-0 right-0 aspect-(--scene-aspect) max-w-full"
        >
          {(["light", "dark"] as const).map((variant) => (
            <Image
              key={variant}
              src={OVERVIEW_SCENE_IMAGES[variant]}
              alt={variant === "light" ? SCENE_ALT : ""}
              aria-hidden={variant === "dark" ? true : undefined}
              width={widthPx}
              height={heightPx}
              sizes="(min-width: 1280px) 1100px, 100vw"
              loading="eager"
              className={cn(
                "pointer-events-none absolute inset-0 h-full w-full object-contain transition-opacity duration-500 select-none",
                variant === "light" ? "dark:hidden" : "hidden dark:block",
                live && ready && "opacity-0",
              )}
            />
          ))}
        </div>
        {live && (
          <div
            className={cn(
              "absolute inset-0 transition-opacity duration-500",
              ready ? "opacity-100" : "opacity-0",
            )}
          >
            <CampusCanvas
              manifest={campus3d}
              stageTones={stageTones}
              traffic={traffic}
              active={shown}
              focus={selected}
              focusRegion={focusRegion}
              components={componentModels}
              animate={animate}
              palette={palette}
              onProject={onProject}
              onHover={setHovered}
              onSelect={toggle}
              onMiss={clearSelection}
              onReady={onReady}
              onError={onError}
            />
          </div>
        )}
        <ul
          aria-label="Pipeline stages"
          className="pointer-events-none absolute inset-y-0 right-0 aspect-(--scene-aspect) max-w-full"
        >
          {SCENE_ANCHOR_IDS.map((id) => {
            const stage = stages[id];
            const Icon = STAGE_ICONS[id];
            const point = overviewScene.anchors[id];
            return (
              <li
                key={id}
                ref={(element) => {
                  labelRefs.current[id] = element;
                }}
                className={cn(
                  "absolute -translate-x-1/2 -translate-y-full transition-opacity duration-300",
                  shown === id && "z-10",
                  live && selected !== null && "pointer-events-none opacity-0",
                )}
                aria-hidden={live && selected !== null ? true : undefined}
                style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
              >
                <button
                  type="button"
                  aria-pressed={selected === id}
                  aria-label={`${labels[id]}: ${stage.summary}`}
                  onClick={() => toggle(id)}
                  onMouseEnter={() => setHovered(id)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(id)}
                  onBlur={() => setHovered(null)}
                  className={cn(
                    "glass-raised pointer-events-auto inline-flex min-h-8 items-center gap-1.5 rounded-xl px-2 py-1 text-xs font-semibold whitespace-nowrap text-ink transition-shadow outline-none @4xl:min-h-9 @4xl:gap-2 @4xl:px-3 @4xl:text-sm",
                    "focus-visible:ring-[3px] focus-visible:ring-brand/40",
                    shown === id && "shadow-glow",
                  )}
                >
                  <Icon className="size-3.5 shrink-0 @4xl:size-4" aria-hidden />
                  {labels[id]}
                  {stage.tone && stage.tone !== "info" && (
                    <i
                      aria-hidden
                      className={cn(
                        "size-2 rounded-full",
                        TONE_DOT[stage.tone],
                      )}
                    />
                  )}
                </button>
                {hovered === id && selected === null && (
                  // Anchored under its own label on wide scenes; edge stages align inward.
                  <div
                    aria-live="polite"
                    className={cn(
                      "pointer-events-auto absolute top-full mt-2 hidden w-72 fit:block",
                      point.x > EDGE_ANCHOR_SHARE
                        ? "right-0"
                        : point.x < 1 - EDGE_ANCHOR_SHARE
                          ? "left-0"
                          : "left-1/2 -translate-x-1/2",
                    )}
                  >
                    <StageCard stage={stage} />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
        {selected && (
          <div
            className="absolute top-2 right-2 z-20 hidden w-72 space-y-2 fit:block"
            aria-live="polite"
          >
            {live && (
              <button
                type="button"
                onClick={clearSelection}
                className="glass-raised ml-auto flex h-9 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
              >
                <ArrowLeft className="size-4" aria-hidden />
                Back to campus
              </button>
            )}
            <StageCard stage={stages[selected]} onClose={clearSelection} />
          </div>
        )}
        {pauseControl && (
          <div className="absolute right-2 bottom-8 hidden fit:block">
            {pauseControl}
          </div>
        )}
      </div>
      {shown && (
        <div className="order-3 fit:hidden" aria-live="polite">
          <StageCard
            stage={stages[shown]}
            onClose={selected ? () => setSelected(null) : undefined}
          />
        </div>
      )}
      {overlay && (
        <div
          className={cn(
            "order-4 grid gap-2.5 @xl:grid-cols-3 fit:order-3 fit:mt-2 fit:w-[min(17rem,36%)] fit:grid-cols-1",
            FLOATING,
            CAMPUS_ONLY,
            live && selected !== null && STEPPED_ASIDE,
          )}
        >
          {overlay}
        </div>
      )}
      <Legend
        className={cn(
          "order-5 fit:order-2",
          CAMPUS_ONLY,
          live && selected !== null && STEPPED_ASIDE,
        )}
        releaseName={releaseName}
        interactive={live}
        control={
          pauseControl && <span className="fit:hidden">{pauseControl}</span>
        }
      />
    </div>
  );
}
