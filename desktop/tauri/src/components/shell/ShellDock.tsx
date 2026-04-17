const BUTTON_BASE =
  "group flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.04] text-white/90 transition hover:border-shell-cyan/40 hover:bg-white/[0.08]";

function Icon({ path }: { path: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 fill-none stroke-current stroke-[1.8]">
      <path d={path} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function ShellDock({
  micMuted,
  compact,
  onMute,
  onInterrupt,
  onChat,
  onSettings,
  onExpand,
}: {
  micMuted: boolean;
  compact: boolean;
  onMute: () => void;
  onInterrupt: () => void;
  onChat: () => void;
  onSettings: () => void;
  onExpand: () => void;
}) {
  return (
    <div className="glass-panel pointer-events-auto flex items-center gap-2 rounded-[30px] px-3 py-3">
      <button type="button" className={BUTTON_BASE} onClick={onMute}>
        <Icon path={micMuted ? "M6 6l12 12M9 9v3a3 3 0 0 0 5.2 2M15 11V8a3 3 0 0 0-6 0v4m-3 0a6 6 0 0 0 9.7 4.7M12 19v3m-4 0h8" : "M12 3a3 3 0 0 1 3 3v6a3 3 0 1 1-6 0V6a3 3 0 0 1 3-3Zm-6 9a6 6 0 1 0 12 0m-6 6v3m-4 0h8"} />
      </button>
      <button type="button" className={BUTTON_BASE} onClick={onInterrupt}>
        <Icon path="M7 7h10v10H7z" />
      </button>
      <button type="button" className={BUTTON_BASE} onClick={onChat}>
        <Icon path="M5 6.5A2.5 2.5 0 0 1 7.5 4h9A2.5 2.5 0 0 1 19 6.5v6A2.5 2.5 0 0 1 16.5 15H11l-4 4v-4.5A2.5 2.5 0 0 1 5 12.5z" />
      </button>
      {!compact ? (
        <button type="button" className={BUTTON_BASE} onClick={onSettings}>
          <Icon path="M12 8.5A3.5 3.5 0 1 1 8.5 12 3.5 3.5 0 0 1 12 8.5Zm7 3.5-1.6-.5a6 6 0 0 0-.6-1.5l.9-1.5-1.7-1.7-1.5.9a6 6 0 0 0-1.5-.6L12 5l-2 .4a6 6 0 0 0-1.5.6L7 5.1 5.3 6.8l.9 1.5a6 6 0 0 0-.6 1.5L4 12l1.6.5c.1.5.3 1 .6 1.5l-.9 1.5L7 17.2l1.5-.9c.5.3 1 .5 1.5.6l2 .1 2-.1c.5-.1 1-.3 1.5-.6l1.5.9 1.7-1.7-.9-1.5c.3-.5.5-1 .6-1.5Z" />
        </button>
      ) : null}
      <button
        type="button"
        className="rounded-2xl border border-white/10 bg-white/[0.06] px-5 py-4 text-xs font-semibold uppercase tracking-[0.24em] text-white/90 transition hover:border-shell-cyan/40 hover:bg-white/[0.1]"
        onClick={onExpand}
      >
        {compact ? "Expand" : "Reduce"}
      </button>
    </div>
  );
}
