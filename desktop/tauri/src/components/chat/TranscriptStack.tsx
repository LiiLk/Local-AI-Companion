import { useShellStore } from "../../store/shellStore";

export function TranscriptStack() {
  const transcripts = useShellStore((state) => state.transcripts.slice(-6));

  return (
    <div className="flex max-h-[320px] flex-col gap-3 overflow-y-auto pr-1">
      {transcripts.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-4 text-sm text-shell-muted">
          Voice stays primary. This panel is only a quiet fallback when you need it.
        </div>
      ) : null}

      {transcripts.map((entry) => (
        <div
          key={entry.id}
          className={`rounded-2xl border px-4 py-3 text-sm leading-6 ${
            entry.role === "assistant"
              ? "border-shell-cyan/15 bg-shell-cyan/[0.08] text-white"
              : "border-white/10 bg-white/[0.04] text-shell-muted"
          }`}
        >
          {entry.text}
        </div>
      ))}
    </div>
  );
}
