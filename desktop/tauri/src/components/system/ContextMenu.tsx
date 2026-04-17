import { motion } from "framer-motion";

const ITEMS = [
  { action: "toggle-mute", label: "Mute / unmute" },
  { action: "toggle-chat", label: "Toggle chat" },
  { action: "toggle-expand", label: "Toggle expand" },
  { action: "open-settings", label: "Settings" },
];

export function ContextMenu({
  onAction,
  onClose,
}: {
  onAction: (action: string) => void;
  onClose: () => void;
}) {
  return (
    <>
      <button type="button" aria-label="close menu" className="absolute inset-0 z-40 bg-transparent" onClick={onClose} />
      <motion.div
        initial={{ opacity: 0, scale: 0.94, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.98, y: 4 }}
        className="glass-panel pointer-events-auto absolute bottom-[148px] right-[22px] z-50 w-[220px] rounded-[24px] p-2"
      >
        {ITEMS.map((item) => (
          <button
            key={item.action}
            type="button"
            onClick={() => onAction(item.action)}
            className="flex w-full rounded-2xl px-4 py-3 text-left text-sm text-white transition hover:bg-white/[0.08]"
          >
            {item.label}
          </button>
        ))}
      </motion.div>
    </>
  );
}
