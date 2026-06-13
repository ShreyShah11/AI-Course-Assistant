import * as React from "react";
import { cn } from "@/lib/utils";

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "h-10 w-full rounded-md border border-border bg-card px-3 text-sm outline-none ring-primary/25 transition placeholder:text-muted focus:ring-4",
        props.className,
      )}
    />
  );
}

export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cn(
        "min-h-24 w-full resize-y rounded-md border border-border bg-card px-3 py-2 text-sm outline-none ring-primary/25 transition placeholder:text-muted focus:ring-4",
        props.className,
      )}
    />
  );
}
