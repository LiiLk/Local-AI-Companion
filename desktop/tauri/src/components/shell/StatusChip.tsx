export function StatusChip({
  label,
  meta,
  tone,
}: {
  label: string;
  meta: string;
  tone: "ready" | "busy" | "error" | "muted";
}) {
  const toneClass =
    tone === "ready"
      ? "bg-shell-success shadow-[0_0_18px_rgba(145,244,180,0.6)]"
      : tone === "error"
        ? "bg-shell-error shadow-[0_0_18px_rgba(255,138,165,0.6)]"
        : tone === "muted"
          ? "bg-shell-warning shadow-[0_0_18px_rgba(255,201,130,0.55)]"
          : "bg-shell-cyan shadow-[0_0_18px_rgba(121,231,255,0.55)]";

  return (
    <div className="glass-panel noise-overlay pointer-events-auto absolute right-2 top-3 z-30 flex min-w-[210px] items-center gap-3 rounded-2xl px-4 py-3">
      <span className={`h-2.5 w-2.5 rounded-full ${toneClass}`} />
      <div className="min-w-0">
        <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-white">{label}</div>
        <div className="truncate text-xs text-shell-muted">{meta}</div>
      </div>
    </div>
  );
}
