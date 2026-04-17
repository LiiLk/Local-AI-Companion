import { motion } from "framer-motion";
import { Composer } from "./Composer";
import { TranscriptStack } from "./TranscriptStack";
import { panelMotion } from "../../hooks/useShellMotion";

export function ExpandPanel({ onSubmit }: { onSubmit: (text: string) => void }) {
  return (
    <motion.aside
      {...panelMotion}
      className="glass-panel noise-overlay pointer-events-auto absolute bottom-8 right-[420px] z-30 w-[560px] rounded-[38px] p-6"
    >
      <div className="mb-5 flex items-end justify-between">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-shell-magenta">Expanded Focus</div>
          <div className="mt-2 text-2xl font-semibold text-white">A larger fallback panel, still companion-first.</div>
        </div>
      </div>
      <TranscriptStack />
      <Composer onSubmit={onSubmit} />
    </motion.aside>
  );
}
