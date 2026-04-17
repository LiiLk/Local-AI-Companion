import { FormEvent } from "react";
import { useShellStore } from "../../store/shellStore";

export function Composer({ onSubmit }: { onSubmit: (text: string) => void }) {
  const chatDraft = useShellStore((state) => state.chatDraft);
  const setChatDraft = useShellStore((state) => state.setChatDraft);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const text = chatDraft.trim();
    if (!text) return;
    onSubmit(text);
    setChatDraft("");
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4 flex items-end gap-3">
      <textarea
        value={chatDraft}
        onChange={(event) => setChatDraft(event.target.value)}
        rows={3}
        placeholder="Type only if voice is inconvenient right now."
        className="min-h-[78px] flex-1 resize-none rounded-[26px] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-white outline-none transition placeholder:text-shell-muted focus:border-shell-cyan/40 focus:bg-white/[0.06]"
      />
      <button
        type="submit"
        className="rounded-[24px] border border-shell-cyan/30 bg-shell-cyan/[0.14] px-5 py-4 text-xs font-semibold uppercase tracking-[0.24em] text-white transition hover:bg-shell-cyan/[0.22]"
      >
        Send
      </button>
    </form>
  );
}
