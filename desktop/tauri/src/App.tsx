import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { AnimatePresence } from "framer-motion";
import { getCharacterPreset } from "./characters/presets";
import { AvatarStageHandle } from "./components/avatar/AvatarStage";
import { ChatPanel } from "./components/chat/ChatPanel";
import { ExpandPanel } from "./components/chat/ExpandPanel";
import { SettingsSheet } from "./components/settings/SettingsSheet";
import { PetShell } from "./components/shell/PetShell";
import { ContextMenu } from "./components/system/ContextMenu";
import { WindowChromeController } from "./components/system/WindowChromeController";
import { useBackendSocket } from "./hooks/useBackendSocket";
import { useHotkeys } from "./hooks/useHotkeys";
import { useIdleManager } from "./hooks/useIdleManager";
import { usePlaybackQueue } from "./hooks/usePlaybackQueue";
import { useShellStore } from "./store/shellStore";

interface BootstrapPayload {
  bridgePort: number;
  runtime: Record<string, unknown>;
  preferences: {
    alwaysOnTop: boolean;
    startMinimizedToTray: boolean;
    autoHideEnabled: boolean;
    characterId: string;
  };
}

export default function App() {
  const [bridgePort, setBridgePort] = useState<number | null>(null);
  const uiMode = useShellStore((state) => state.uiMode);
  const settingsVisible = useShellStore((state) => state.settingsVisible);
  const contextMenuVisible = useShellStore((state) => state.contextMenuVisible);
  const characterId = useShellStore((state) => state.characterId);
  const runtime = useShellStore((state) => state.runtime);
  const applyRuntime = useShellStore((state) => state.applyRuntime);
  const setPreferenceFlags = useShellStore((state) => state.setPreferenceFlags);
  const setChatVisible = useShellStore((state) => state.setChatVisible);
  const setSettingsVisible = useShellStore((state) => state.setSettingsVisible);
  const setUiMode = useShellStore((state) => state.setUiMode);
  const setContextMenuVisible = useShellStore((state) => state.setContextMenuVisible);
  const touchInteraction = useShellStore((state) => state.touchInteraction);
  const avatarRef = useRef<AvatarStageHandle | null>(null);

  const { sendCommand } = useBackendSocket(bridgePort);
  usePlaybackQueue(avatarRef);
  useIdleManager();

  const character = useMemo(() => getCharacterPreset(characterId), [characterId]);

  useEffect(() => {
    void invoke<BootstrapPayload>("bootstrap_app").then((payload) => {
      setBridgePort(payload.bridgePort);
      applyRuntime(payload.runtime);
      setPreferenceFlags({
        alwaysOnTop: payload.preferences.alwaysOnTop,
        startMinimizedToTray: payload.preferences.startMinimizedToTray,
        autoHideEnabled: payload.preferences.autoHideEnabled,
      });
    });
  }, [applyRuntime, setPreferenceFlags]);

  useEffect(() => {
    let unlistenBackendReady: (() => void) | null = null;
    let unlistenBackendExited: (() => void) | null = null;
    let unlistenBackendRestarted: (() => void) | null = null;

    void listen("host://backend-ready", () => {
      void invoke<BootstrapPayload>("bootstrap_app").then((payload) => {
        setBridgePort(payload.bridgePort);
      });
    }).then((dispose) => {
      unlistenBackendReady = dispose;
    });

    void listen("host://backend-restarted", () => {
      void invoke<BootstrapPayload>("bootstrap_app").then((payload) => {
        setBridgePort(payload.bridgePort);
      });
    }).then((dispose) => {
      unlistenBackendRestarted = dispose;
    });

    void listen("host://backend-exited", () => {
      setBridgePort(null);
      applyRuntime({ backend_state: "error", runtime_error: "Desktop backend exited unexpectedly." });
    }).then((dispose) => {
      unlistenBackendExited = dispose;
    });

    return () => {
      unlistenBackendReady?.();
      unlistenBackendExited?.();
      unlistenBackendRestarted?.();
    };
  }, [applyRuntime]);

  const handleHostAction = useCallback(
    (action: string) => {
      touchInteraction();
      switch (action) {
        case "toggle-chat":
          setSettingsVisible(false);
          setChatVisible(uiMode === "pet");
          break;
        case "toggle-expand":
          setUiMode(uiMode === "expand" ? "pet" : "expand");
          break;
        case "toggle-mute":
          void sendCommand("toggle_mute");
          break;
        case "interrupt":
          void sendCommand("interrupt");
          break;
        case "toggle-debug":
          void sendCommand("toggle_debug");
          break;
        case "open-settings":
          setSettingsVisible(true);
          break;
        default:
          break;
      }
    },
    [sendCommand, setChatVisible, setSettingsVisible, setUiMode, touchInteraction, uiMode],
  );

  useHotkeys(handleHostAction);

  const handleSubmitChat = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      touchInteraction();
      setChatVisible(true);
      await sendCommand("send_text", { text });
    },
    [sendCommand, setChatVisible, touchInteraction],
  );

  return (
    <div className="min-h-screen w-screen bg-transparent text-shell-text antialiased">
      <WindowChromeController />

      <div className="relative flex min-h-screen w-screen items-end justify-end p-8">
        <PetShell
          avatarRef={avatarRef}
          character={character}
          onAction={handleHostAction}
          onSubmitChat={handleSubmitChat}
          onRequestContextMenu={() => setContextMenuVisible(true)}
          runtime={runtime}
        />

        <AnimatePresence initial={false}>
          {settingsVisible ? (
            <SettingsSheet
              key="settings-sheet"
              character={character}
              onClose={() => setSettingsVisible(false)}
              onPreferencesChange={(preferences) => {
                setPreferenceFlags(preferences);
                void invoke("set_host_preferences", { preferences: { ...preferences, characterId } });
              }}
            />
          ) : null}

          {!settingsVisible && uiMode === "chat" ? (
            <ChatPanel key="chat-panel" onSubmit={handleSubmitChat} />
          ) : null}

          {!settingsVisible && uiMode === "expand" ? (
            <ExpandPanel key="expand-panel" onSubmit={handleSubmitChat} />
          ) : null}
        </AnimatePresence>

        {contextMenuVisible ? (
          <ContextMenu
            onAction={(action) => {
              setContextMenuVisible(false);
              handleHostAction(action);
            }}
            onClose={() => setContextMenuVisible(false)}
          />
        ) : null}
      </div>
    </div>
  );
}
