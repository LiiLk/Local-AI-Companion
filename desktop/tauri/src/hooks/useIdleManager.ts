import { useEffect } from "react";
import { useShellStore } from "../store/shellStore";

const AUTO_HIDE_DELAY_MS = 14000;

export function useIdleManager() {
  const autoHideEnabled = useShellStore((state) => state.autoHideEnabled);
  const uiMode = useShellStore((state) => state.uiMode);
  const runtime = useShellStore((state) => state.runtime);
  const lastInteractionAt = useShellStore((state) => state.lastInteractionAt);
  const setChatVisible = useShellStore((state) => state.setChatVisible);
  const setSettingsVisible = useShellStore((state) => state.setSettingsVisible);

  useEffect(() => {
    if (!autoHideEnabled || uiMode === "pet") {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      const isBusy = runtime?.response_active || runtime?.playback_active;
      if (isBusy) {
        return;
      }
      setSettingsVisible(false);
      setChatVisible(false);
    }, Math.max(1000, AUTO_HIDE_DELAY_MS - (Date.now() - lastInteractionAt)));

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [autoHideEnabled, lastInteractionAt, runtime?.playback_active, runtime?.response_active, setChatVisible, setSettingsVisible, uiMode]);
}
