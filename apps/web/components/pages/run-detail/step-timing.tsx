import { formatTiming } from "@/lib/run-detail";
import { cn } from "@/lib/utils";

/** A step timing in ms or s; a missing reading is a dash announced as unavailable. */
export function StepTiming({
  seconds,
  className,
}: {
  seconds: number | null;
  className?: string;
}) {
  const timing = formatTiming(seconds);
  return (
    <span
      className={cn(
        "tabular-nums",
        !timing.hasValue && "text-ink-subtle",
        className,
      )}
    >
      <span aria-hidden>{timing.text}</span>
      <span className="sr-only">{timing.accessibleText}</span>
    </span>
  );
}
