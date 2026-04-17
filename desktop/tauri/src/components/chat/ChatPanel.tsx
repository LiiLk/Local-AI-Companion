import { motion } from "framer-motion";
import { Composer } from "./Composer";
import { TranscriptStack } from "./TranscriptStack";
import { panelMotion } from "../../hooks/useShellMotion";

export function ChatPanel({ onSubmit }: { onSubmit: (text: string) => void }) {
  return (
    <motion.aside
      {...panelMotion}
      className="glass-panel noise-overlay pointer-events-auto absolute bottom-8 right-[310px] z-30 w-[390px] rounded-[34px] p-5"
    >
      <div className="mb-4 flex items-end justify-between">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-shell-cyan">Fallback Chat</div>
          <div className="mt-2 text-xl font-semibold text-white">Quietly attached to the avatar.</div>
        </div>
      </div>
      <TranscriptStack />
      <Composer onSubmit={onSubmit} />
    </motion.aside>
  );
}
