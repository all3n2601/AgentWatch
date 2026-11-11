import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

const glassPanelVariants = cva("glass-panel relative text-ink", {
  variants: {
    padding: {
      none: "",
      sm: "p-3",
      default: "p-4 sm:p-5",
      lg: "p-5 sm:p-6",
    },
    // A tone draws the left accent seen on callouts and banners.
    tone: {
      none: "",
      success: "border-l-[3px] border-l-success-solid",
      warning: "border-l-[3px] border-l-warning-solid",
      caution: "border-l-[3px] border-l-caution-solid",
      danger: "border-l-[3px] border-l-danger-solid",
      info: "border-l-[3px] border-l-info-solid",
    },
  },
  defaultVariants: { padding: "default", tone: "none" },
});

type GlassPanelProps = React.ComponentProps<"div"> &
  VariantProps<typeof glassPanelVariants> & {
    /** Render as a landmark-bearing element when the panel is a page region. */
    as?: "div" | "section" | "aside" | "article";
  };

function GlassPanel({
  as: Comp = "div",
  className,
  padding,
  tone,
  ...props
}: GlassPanelProps) {
  return (
    <Comp
      data-slot="glass-panel"
      className={cn(glassPanelVariants({ padding, tone }), className)}
      {...props}
    />
  );
}

/** A nested card inside a panel. It does not blur, so nesting stays cheap. */
function InsetCard({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="inset-card"
      className={cn("glass-inset p-3 sm:p-4", className)}
      {...props}
    />
  );
}

type HeadingLevel = "h2" | "h3" | "h4";

type PanelHeaderProps = Omit<React.ComponentProps<"div">, "title"> & {
  title: React.ReactNode;
  icon?: LucideIcon;
  description?: React.ReactNode;
  /** Right-aligned slot for legends, meta chips or actions. */
  actions?: React.ReactNode;
  level?: HeadingLevel;
  titleId?: string;
};

function PanelHeader({
  title,
  icon: Icon,
  description,
  actions,
  level: Heading = "h2",
  titleId,
  className,
  ...props
}: PanelHeaderProps) {
  return (
    <div
      data-slot="panel-header"
      className={cn(
        "mb-3 flex flex-wrap items-start justify-between gap-x-4 gap-y-2",
        className,
      )}
      {...props}
    >
      <div className="flex min-w-0 items-center gap-3">
        {Icon && (
          <span className="glass-raised flex size-9 shrink-0 items-center justify-center rounded-xl">
            <Icon className="size-5 text-ink" aria-hidden />
          </span>
        )}
        <div className="min-w-0">
          <Heading
            id={titleId}
            className="text-base leading-tight font-semibold tracking-tight text-ink sm:text-lg"
          >
            {title}
          </Heading>
          {description && (
            <p className="mt-0.5 text-sm text-ink-muted">{description}</p>
          )}
        </div>
      </div>
      {actions && (
        <div className="flex flex-wrap items-center gap-2">{actions}</div>
      )}
    </div>
  );
}

export { GlassPanel, InsetCard, PanelHeader, glassPanelVariants };
