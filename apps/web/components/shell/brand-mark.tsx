import { cn } from "@/lib/utils";

/** The AgentWatch waveform mark on its glowing green tile. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn(
        "flex size-11 shrink-0 items-center justify-center rounded-2xl bg-linear-to-br from-[#b9f39f] to-[#5fd63d] text-[#14400d] shadow-glow",
        className,
      )}
    >
      <svg viewBox="0 0 32 32" className="size-7" fill="none">
        <path
          d="M3 18c3.5 0 3.5-10 7-10s3.5 16 7 16 3.5-13 7-13 3.5 7 5 7"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
        />
      </svg>
    </span>
  );
}
