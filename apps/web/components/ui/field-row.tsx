import * as React from "react";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Settings form row: label column and control column, stacking on narrow
 * screens. `htmlFor` links a native control; grouped controls (segmented,
 * switch) reference `labelId` through `aria-labelledby` instead.
 */
function FieldRow({
  label,
  htmlFor,
  labelId,
  hint,
  children,
  className,
}: {
  label: string;
  htmlFor?: string;
  labelId?: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid gap-2 py-1.5 sm:grid-cols-[minmax(10rem,13rem)_1fr] sm:items-center sm:gap-6",
        className,
      )}
    >
      <label id={labelId} htmlFor={htmlFor} className="text-[15px] text-ink">
        {label}
      </label>
      <div className="min-w-0">
        {children}
        {hint && (
          <p className="mt-2 flex items-center gap-1.5 text-sm text-ink-muted">
            <Info className="size-4 shrink-0" aria-hidden />
            {hint}
          </p>
        )}
      </div>
    </div>
  );
}

/** Section of a settings form with a heading and a divider above the next. */
function FieldSection({
  title,
  id,
  children,
}: {
  title: string;
  id?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      aria-labelledby={id}
      className="border-b border-line pb-5 last:border-b-0"
    >
      <h2
        id={id}
        className="mb-2 text-xl font-semibold tracking-tight text-ink"
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

/** Native text input in the glass style, for FieldRow controls. */
function GlassInput({ className, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      className={cn(
        "h-11 w-full rounded-xl border border-line bg-glass-strong px-4 text-[15px] text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40",
        className,
      )}
      {...props}
    />
  );
}

export { FieldRow, FieldSection, GlassInput };
