import * as React from "react";
import Image from "next/image";
import type { LucideIcon } from "lucide-react";
import {
  OVERVIEW_SCENE_IMAGES,
  overviewScene,
  type SceneAnchorId,
} from "@/lib/overview-scene";
import { cn } from "@/lib/utils";

export interface SceneLabel {
  anchor: SceneAnchorId;
  label: string;
  icon?: LucideIcon;
}

const SCENE_ALT =
  "Illustrated campus of the agent pipeline: requests enter at the agent gateway, pass the model runtime, loop through retrieval and the CPU tool, and leave at completion.";

/**
 * The Overview's Blender-rendered campus (scenes/overview_campus.py) with
 * labels pinned to projected building anchors. The render is decorative and
 * holds no data; anything measured belongs in `labels` text or `overlay`.
 *
 * Both theme renders are in the DOM and CSS shows one, so the theme can change
 * without a client component or a flash of the wrong image.
 */
export function CampusScene({
  labels,
  overlay,
  className,
}: {
  labels: readonly SceneLabel[];
  /** Panels laid over the scene's open left side when it is wide, stacked below it when narrow. */
  overlay?: React.ReactNode;
  className?: string;
}) {
  const { widthPx, heightPx } = overviewScene.image;
  return (
    // Sizes follow the scene's own width (container queries), not the
    // viewport, so labels and overlays fit wherever the scene is placed.
    <div className={cn("@container relative", className)}>
      <div
        className="relative"
        style={{ aspectRatio: `${widthPx} / ${heightPx}` }}
      >
        {(["light", "dark"] as const).map((theme) => (
          <Image
            key={theme}
            src={OVERVIEW_SCENE_IMAGES[theme]}
            alt={theme === "light" ? SCENE_ALT : ""}
            aria-hidden={theme === "dark" ? true : undefined}
            width={widthPx}
            height={heightPx}
            sizes="(min-width: 1280px) 900px, 100vw"
            loading="eager"
            className={cn(
              "pointer-events-none h-full w-full object-contain select-none",
              theme === "light"
                ? "dark:hidden"
                : "absolute inset-0 hidden dark:block",
            )}
          />
        ))}
        <ul aria-label="Pipeline stages">
          {labels.map((item) => {
            const point = overviewScene.anchors[item.anchor];
            return (
              <li
                key={item.anchor}
                className="absolute -translate-x-1/2 -translate-y-full"
                style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
              >
                <span className="glass-raised inline-flex items-center gap-1.5 rounded-xl px-2 py-1 text-xs font-semibold whitespace-nowrap text-ink @4xl:gap-2 @4xl:px-3.5 @4xl:py-2 @4xl:text-[15px]">
                  {item.icon && (
                    <item.icon
                      className="size-3.5 shrink-0 @4xl:size-5"
                      aria-hidden
                    />
                  )}
                  {item.label}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
      {overlay && (
        <div className="mt-4 grid gap-3 @xl:grid-cols-2 @4xl:absolute @4xl:top-[6%] @4xl:left-0 @4xl:mt-0 @4xl:w-[min(19rem,28%)] @4xl:grid-cols-1">
          {overlay}
        </div>
      )}
    </div>
  );
}
