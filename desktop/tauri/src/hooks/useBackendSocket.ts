import { emit } from "@tauri-apps/api/event";
import { useEffect, useMemo, useRef, useState } from "react";
import { BackendRuntime, useShellStore } from "../store/shellStore";

type CommandName = "send_text" | "interrupt" | "toggle_mute" | "get_runtime_state" | "toggle_debug";

interface CommandRequest {
  type: "command";
  request_id: number;
  name: CommandName;
  text?: string;
}

interface BackendReadyMessage {
  type: "backend_ready";
  runtime: BackendRuntime;
}

interface FrontendEventMessage {
  type: "frontend_event";
  name: string;
  args: unknown[];
  runtime: BackendRuntime;
}

interface CommandResultMessage {
  type: "command_result";
  request_id: number;
  ok: boolean;
  result?: BackendRuntime;
  error?: string;
}

type BridgeMessage = BackendReadyMessage | FrontendEventMessage | CommandResultMessage;

export function useBackendSocket(bridgePort: number | null) {
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const nextRequestId = useRef(1);
  const pending = useRef(new Map<number, { resolve: (value: BackendRuntime | undefined) => void; reject: (error: Error) => void }>());

  const applyRuntime = useShellStore((state) => state.applyRuntime);
  const upsertUserTranscript = useShellStore((state) => state.upsertUserTranscript);
  const beginAssistantTurn = useShellStore((state) => state.beginAssistantTurn);
  const appendAssistantChunk = useShellStore((state) => state.appendAssistantChunk);
  const finalizeAssistantTurn = useShellStore((state) => state.finalizeAssistantTurn);
  const setAvatarExpression = useShellStore((state) => state.setAvatarExpression);
  const setDebugOverlayVisible = useShellStore((state) => state.setDebugOverlayVisible);

  useEffect(() => {
    if (!bridgePort) {
      return;
    }

    const socket = new WebSocket(`ws://127.0.0.1:${bridgePort}`);
    socketRef.current = socket;

    socket.addEventListener("open", () => {
      setConnected(true);
    });

    socket.addEventListener("close", () => {
      setConnected(false);
      socketRef.current = null;
      pending.current.forEach(({ reject }) => reject(new Error("desktop bridge disconnected")));
      pending.current.clear();
    });

    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data)) as BridgeMessage;
      if ("runtime" in message && message.runtime) {
        applyRuntime(message.runtime);
      }

      if (message.type === "backend_ready") {
        return;
      }

      if (message.type === "command_result") {
        const resolver = pending.current.get(message.request_id);
        if (!resolver) {
          return;
        }
        pending.current.delete(message.request_id);
        if (!message.ok) {
          resolver.reject(new Error(message.error ?? "desktop bridge command failed"));
          return;
        }
        resolver.resolve(message.result);
        if (message.result) {
          applyRuntime(message.result);
        }
        return;
      }

      if (message.type === "frontend_event") {
        const [arg0, arg1] = message.args;
        switch (message.name) {
          case "onBackendReady":
            break;
          case "onMicStateChange":
            applyRuntime({ ...(message.runtime ?? {}), mic_state: String(arg0 ?? "loading") });
            break;
          case "onSpeechStart":
            beginAssistantTurn((arg0 as number | null | undefined) ?? null);
            break;
          case "onTranscription":
            upsertUserTranscript(String(arg0 ?? ""), (arg1 as number | null | undefined) ?? null);
            break;
          case "onResponseStart":
            beginAssistantTurn((arg0 as number | null | undefined) ?? null);
            break;
          case "onResponseChunk":
            appendAssistantChunk(String(arg0 ?? ""), (arg1 as number | null | undefined) ?? null);
            break;
          case "onResponseEnd":
            finalizeAssistantTurn(String(arg0 ?? ""), (arg1 as number | null | undefined) ?? null);
            break;
          case "onAudioReady":
            void emit("host://audio-ready", arg0);
            break;
          case "onExpressionChange":
            setAvatarExpression(String(arg0 ?? "neutral"));
            break;
          case "onError":
            finalizeAssistantTurn(String(arg0 ?? "The assistant hit an error."), (arg1 as number | null | undefined) ?? null);
            break;
          case "onPlaybackStop":
            void emit("host://audio-stop");
            setDebugOverlayVisible(false);
            break;
          default:
            break;
        }
      }
    });

    return () => {
      socket.close();
    };
  }, [
    appendAssistantChunk,
    applyRuntime,
    beginAssistantTurn,
    bridgePort,
    finalizeAssistantTurn,
    setAvatarExpression,
    setDebugOverlayVisible,
    upsertUserTranscript,
  ]);

  const sendCommand = useMemo(
    () =>
      (name: CommandName, payload?: { text?: string }) =>
        new Promise<BackendRuntime | undefined>((resolve, reject) => {
          const socket = socketRef.current;
          if (!socket || socket.readyState !== WebSocket.OPEN) {
            reject(new Error("desktop bridge is not connected"));
            return;
          }
          const requestId = nextRequestId.current++;
          pending.current.set(requestId, { resolve, reject });
          const message: CommandRequest = {
            type: "command",
            request_id: requestId,
            name,
            text: payload?.text,
          };
          socket.send(JSON.stringify(message));
        }),
    [],
  );

  return { connected, sendCommand };
}
