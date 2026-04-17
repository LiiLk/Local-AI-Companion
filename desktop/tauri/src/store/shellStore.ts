import { create } from "zustand";
import { getCharacterPreset, normalizeCharacterId } from "../characters/presets";

export type UiMode = "pet" | "chat" | "expand";
export type BackendState = "warming_up" | "ready" | "degraded" | "error";
export type MicState = "loading" | "muted" | "listening" | "speaking";
export type CameraPreset = "petBust" | "chatBust" | "fullBody";

export interface BackendRuntime {
  backend?: string;
  backend_state?: BackendState;
  mic_state?: MicState | string;
  active_turn_id?: number | null;
  response_active?: boolean;
  playback_active?: boolean;
  debug_visible?: boolean;
  character_name?: string;
  character_id?: string;
  active_language?: string | null;
  active_llm_model?: string | null;
  active_tts_provider?: string | null;
  degraded_reason?: string | null;
  runtime_error?: string | null;
  [key: string]: unknown;
}

export interface TranscriptEntry {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  turnId?: number | null;
}

interface ShellState {
  uiMode: UiMode;
  cameraPreset: CameraPreset;
  chatVisible: boolean;
  settingsVisible: boolean;
  contextMenuVisible: boolean;
  debugOverlayVisible: boolean;
  backendState: BackendState;
  micState: MicState;
  runtime: BackendRuntime | null;
  characterId: string;
  characterName: string;
  avatarExpression: string;
  activeTurnId: number | null;
  alwaysOnTop: boolean;
  startMinimizedToTray: boolean;
  autoHideEnabled: boolean;
  lastInteractionAt: number;
  chatDraft: string;
  transcripts: TranscriptEntry[];
  setUiMode: (mode: UiMode) => void;
  setChatVisible: (visible: boolean) => void;
  setSettingsVisible: (visible: boolean) => void;
  setContextMenuVisible: (visible: boolean) => void;
  setDebugOverlayVisible: (visible: boolean) => void;
  setAvatarExpression: (expression: string) => void;
  setChatDraft: (value: string) => void;
  touchInteraction: () => void;
  applyRuntime: (runtime: BackendRuntime) => void;
  setPreferenceFlags: (flags: Partial<Pick<ShellState, "alwaysOnTop" | "startMinimizedToTray" | "autoHideEnabled">>) => void;
  upsertUserTranscript: (text: string, turnId?: number | null) => void;
  beginAssistantTurn: (turnId?: number | null) => void;
  appendAssistantChunk: (chunk: string, turnId?: number | null) => void;
  finalizeAssistantTurn: (text: string, turnId?: number | null) => void;
  clearTranscripts: () => void;
}

function resolveCameraPreset(mode: UiMode): CameraPreset {
  if (mode === "expand") return "fullBody";
  if (mode === "chat") return "chatBust";
  return "petBust";
}

function updateTranscript(
  transcripts: TranscriptEntry[],
  role: "user" | "assistant",
  text: string,
  turnId?: number | null,
): TranscriptEntry[] {
  if (!text.trim()) {
    return transcripts;
  }

  const lastEntry = transcripts[transcripts.length - 1];
  if (lastEntry && lastEntry.role === role && lastEntry.turnId === turnId) {
    return [
      ...transcripts.slice(0, -1),
      {
        ...lastEntry,
        text,
      },
    ];
  }

  return [
    ...transcripts,
    {
      id: `${role}-${turnId ?? "global"}-${Date.now()}`,
      role,
      text,
      turnId,
    },
  ];
}

export const useShellStore = create<ShellState>((set) => ({
  uiMode: "pet",
  cameraPreset: "petBust",
  chatVisible: false,
  settingsVisible: false,
  contextMenuVisible: false,
  debugOverlayVisible: false,
  backendState: "warming_up",
  micState: "loading",
  runtime: null,
  characterId: "march7th",
  characterName: getCharacterPreset("march7th").name,
  avatarExpression: "neutral",
  activeTurnId: null,
  alwaysOnTop: true,
  startMinimizedToTray: false,
  autoHideEnabled: true,
  lastInteractionAt: Date.now(),
  chatDraft: "",
  transcripts: [],
  setUiMode: (uiMode) =>
    set(() => ({
      uiMode,
      cameraPreset: resolveCameraPreset(uiMode),
      chatVisible: uiMode !== "pet",
      settingsVisible: false,
      contextMenuVisible: false,
      lastInteractionAt: Date.now(),
    })),
  setChatVisible: (chatVisible) =>
    set((state) => ({
      chatVisible,
      uiMode: chatVisible ? (state.uiMode === "expand" ? "expand" : "chat") : "pet",
      cameraPreset: resolveCameraPreset(chatVisible ? (state.uiMode === "expand" ? "expand" : "chat") : "pet"),
      lastInteractionAt: Date.now(),
    })),
  setSettingsVisible: (settingsVisible) =>
    set((state) => ({
      settingsVisible,
      chatVisible: settingsVisible ? false : state.chatVisible,
      contextMenuVisible: false,
      lastInteractionAt: Date.now(),
    })),
  setContextMenuVisible: (contextMenuVisible) => set(() => ({ contextMenuVisible })),
  setDebugOverlayVisible: (debugOverlayVisible) => set(() => ({ debugOverlayVisible })),
  setAvatarExpression: (avatarExpression) => set(() => ({ avatarExpression })),
  setChatDraft: (chatDraft) => set(() => ({ chatDraft })),
  touchInteraction: () => set(() => ({ lastInteractionAt: Date.now() })),
  applyRuntime: (runtime) =>
    set((state) => {
      const characterId = normalizeCharacterId(
        (runtime.character_id as string | null | undefined) ?? (runtime.character_name as string | null | undefined),
      );
      const preset = getCharacterPreset(characterId);

      return {
        runtime,
        backendState: (runtime.backend_state as BackendState | undefined) ?? state.backendState,
        micState: (runtime.mic_state as MicState | undefined) ?? state.micState,
        activeTurnId: (runtime.active_turn_id as number | null | undefined) ?? state.activeTurnId,
        debugOverlayVisible:
          typeof runtime.debug_visible === "boolean" ? runtime.debug_visible : state.debugOverlayVisible,
        characterId,
        characterName: (runtime.character_name as string | undefined) ?? preset.name,
      };
    }),
  setPreferenceFlags: (flags) => set((state) => ({ ...state, ...flags })),
  upsertUserTranscript: (text, turnId) =>
    set((state) => ({
      transcripts: updateTranscript(state.transcripts, "user", text, turnId),
      activeTurnId: turnId ?? state.activeTurnId,
      lastInteractionAt: Date.now(),
    })),
  beginAssistantTurn: (turnId) =>
    set((state) => ({
      transcripts: updateTranscript(state.transcripts, "assistant", "", turnId),
      activeTurnId: turnId ?? state.activeTurnId,
    })),
  appendAssistantChunk: (chunk, turnId) =>
    set((state) => {
      const lastAssistant = [...state.transcripts]
        .reverse()
        .find((entry) => entry.role === "assistant" && entry.turnId === turnId);
      const nextText = `${lastAssistant?.text ?? ""}${chunk}`;
      return {
        transcripts: updateTranscript(state.transcripts, "assistant", nextText, turnId),
        activeTurnId: turnId ?? state.activeTurnId,
      };
    }),
  finalizeAssistantTurn: (text, turnId) =>
    set((state) => ({
      transcripts: updateTranscript(state.transcripts, "assistant", text, turnId),
      activeTurnId: turnId ?? state.activeTurnId,
      lastInteractionAt: Date.now(),
    })),
  clearTranscripts: () => set(() => ({ transcripts: [] })),
}));
