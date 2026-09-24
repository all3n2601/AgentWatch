import type { ButtonHTMLAttributes } from "react";

export function Button({ className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`rounded-lg bg-emerald-400 px-5 py-3 font-medium text-zinc-950 transition hover:bg-emerald-300 ${className}`}
      {...props}
    />
  );
}

