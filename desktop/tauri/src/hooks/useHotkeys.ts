import { useEffect } from "react";
import { listen } from "@tauri-apps/api/event";

export function useHotkeys(onAction: (action: string) => void) {
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    void listen<{ action: string }>("host://action", (event) => {
      onAction(event.payload.action);
    }).then((dispose) => {
      unlisten = dispose;
    });

    return () => {
      unlisten?.();
    };
  }, [onAction]);
}
