import { motion } from "framer-motion";
import { CharacterPreset } from "../../characters/presets";
import { panelMotion } from "../../hooks/useShellMotion";
import { useShellStore } from "../../store/shellStore";

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
      <div>
        <div className="text-sm font-medium text-white">{label}</div>
        <div className="mt-1 text-xs text-shell-muted">{description}</div>
      </div>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

export function SettingsSheet({
  character,
  onClose,
  onPreferencesChange,
}: {
  character: CharacterPreset;
  onClose: () => void;
  onPreferencesChange: (preferences: {
    alwaysOnTop: boolean;
    startMinimizedToTray: boolean;
    autoHideEnabled: boolean;
  }) => void;
}) {
  const alwaysOnTop = useShellStore((state) => state.alwaysOnTop);
  const startMinimizedToTray = useShellStore((state) => state.startMinimizedToTray);
  const autoHideEnabled = useShellStore((state) => state.autoHideEnabled);

  return (
    <motion.aside
      {...panelMotion}
      className="glass-panel noise-overlay pointer-events-auto absolute bottom-8 right-[310px] z-40 w-[360px] rounded-[34px] p-5"
    >
      <div className="mb-5 flex items-center justify-between">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-shell-cyan">Settings</div>
          <div className="mt-2 text-xl font-semibold text-white">{character.name}</div>
        </div>
        <button type="button" onClick={onClose} className="rounded-2xl border border-white/10 px-3 py-2 text-xs uppercase tracking-[0.24em] text-shell-muted">
          Close
        </button>
      </div>

      <div className="space-y-3">
        <ToggleRow
          label="Always on top"
          description="Keep the companion floating above normal windows."
          checked={alwaysOnTop}
          onChange={(checked) => onPreferencesChange({ alwaysOnTop: checked, startMinimizedToTray, autoHideEnabled })}
        />
        <ToggleRow
          label="Start minimized"
          description="Open in the tray first and reveal only on demand."
          checked={startMinimizedToTray}
          onChange={(checked) => onPreferencesChange({ alwaysOnTop, startMinimizedToTray: checked, autoHideEnabled })}
        />
        <ToggleRow
          label="Auto hide chat"
          description="Collapse visual panels after inactivity while staying voice-first."
          checked={autoHideEnabled}
          onChange={(checked) => onPreferencesChange({ alwaysOnTop, startMinimizedToTray, autoHideEnabled: checked })}
        />
      </div>
    </motion.aside>
  );
}
