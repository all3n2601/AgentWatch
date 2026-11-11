import * as React from "react";
import { Search } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Glass search pill. It is a plain GET form by default so search works without
 * client JavaScript; pass `action` for the results route.
 */
function SearchField({
  label = "Search runs, agents, tools",
  name = "q",
  action,
  defaultValue,
  placeholder = "Search runs, agents, tools…",
  className,
}: {
  label?: string;
  name?: string;
  action?: string;
  defaultValue?: string;
  placeholder?: string;
  className?: string;
}) {
  const id = React.useId();
  return (
    <form
      role="search"
      action={action}
      className={cn("relative w-full", className)}
    >
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      <Search
        className="pointer-events-none absolute top-1/2 left-4 size-5 -translate-y-1/2 text-ink-muted"
        aria-hidden
      />
      <input
        id={id}
        type="search"
        name={name}
        defaultValue={defaultValue}
        placeholder={placeholder}
        className="glass-raised h-12 w-full rounded-2xl pr-4 pl-12 text-[15px] text-ink outline-none placeholder:text-ink-subtle focus-visible:ring-[3px] focus-visible:ring-brand/40"
      />
    </form>
  );
}

export { SearchField };
