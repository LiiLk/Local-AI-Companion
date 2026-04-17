import { useEffect, useMemo, useRef } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useShellStore } from "../store/shellStore";

const PASS_THROUGH_DEBOUNCE_MS = 80;

export function useClickThrough() {
  const uiMode = useShellStore((state) => state.uiMode);
  const settingsVisible = useShellStore((state) => state.settingsVisible);
  const pointerInsideInteractive = useRef(0);
  const timerRef = useRef<number | null>(null);

  const setIgnoreCursorEvents = async (ignore: boolean) => {
    try {
      await getCurrentWindow().setIgnoreCursorEvents(ignore);
    } catch (error) {
      console.debug("setIgnoreCursorEvents unavailable", error);
    }
  };

  const schedule = (ignore: boolean) => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(() => {
      void setIgnoreCursorEvents(ignore);
    }, PASS_THROUGH_DEBOUNCE_MS);
  };

  useEffect(() => {
    const shouldIgnore = uiMode === "pet" && !settingsVisible && pointerInsideInteractive.current === 0;
    schedule(shouldIgnore);
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, [settingsVisible, uiMode]);

  return useMemo(
    () => ({
      bindInteractiveRegion() {
        return {
          onMouseEnter() {
            pointerInsideInteractive.current += 1;
            schedule(false);
          },
          onMouseLeave() {
            pointerInsideInteractive.current = Math.max(0, pointerInsideInteractive.current - 1);
            schedule(uiMode === "pet" && !settingsVisible && pointerInsideInteractive.current === 0);
          },
        };
      },
    }),
    [settingsVisible, uiMode],
  );
}
