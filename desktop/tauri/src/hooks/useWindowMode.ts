import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { LogicalSize, getCurrentWindow } from "@tauri-apps/api/window";
import { UiMode, useShellStore } from "../store/shellStore";

const WINDOW_PRESETS: Record<UiMode, { width: number; height: number }> = {
  pet: { width: 840, height: 720 },
  chat: { width: 1040, height: 720 },
  expand: { width: 1220, height: 780 },
};

export function useWindowMode() {
  const uiMode = useShellStore((state) => state.uiMode);
  const settingsVisible = useShellStore((state) => state.settingsVisible);
  const alwaysOnTop = useShellStore((state) => state.alwaysOnTop);

  useEffect(() => {
    const run = async () => {
      const window = getCurrentWindow();
      const preset = settingsVisible ? WINDOW_PRESETS.chat : WINDOW_PRESETS[uiMode];
      await window.setSize(new LogicalSize(preset.width, preset.height));
      await window.setAlwaysOnTop(alwaysOnTop);
      await window.setSkipTaskbar(uiMode === "pet");
      await invoke("sync_window_surface", {
        surface: settingsVisible ? "settings" : uiMode,
      });
    };

    void run();
  }, [alwaysOnTop, settingsVisible, uiMode]);
}
