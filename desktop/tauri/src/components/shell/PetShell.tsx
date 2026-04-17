import { RefObject, useMemo } from "react";
import { motion } from "framer-motion";
import { CharacterPreset } from "../../characters/presets";
import { useClickThrough } from "../../hooks/useClickThrough";
import { shellMotion } from "../../hooks/useShellMotion";
import { BackendRuntime, useShellStore } from "../../store/shellStore";
import { AvatarHalo } from "../avatar/AvatarHalo";
import { AvatarOrb } from "../avatar/AvatarOrb";
import { AvatarStage, AvatarStageHandle } from "../avatar/AvatarStage";
import { ShellDock } from "./ShellDock";
import { StatusChip } from "./StatusChip";

function statusFromRuntime(runtime: BackendRuntime | null) {
  const backendState = runtime?.backend_state ?? "warming_up";
  const micState = runtime?.mic_state ?? "loading";

  if (backendState === "error") {
    return {
      label: "Error",
      meta: String(runtime?.runtime_error ?? "Backend unavailable"),
      tone: "error" as const,
    };
  }

  if (backendState === "degraded") {
    return {
      label: "Degraded",
      meta: String(runtime?.degraded_reason ?? "Reduced pipeline available"),
      tone: "busy" as const,
    };
  }

  if (micState === "muted") {
    return {
      label: "Muted",
      meta: "Voice capture is paused.",
      tone: "muted" as const,
    };
  }

  if (runtime?.response_active || runtime?.playback_active) {
    return {
      label: "Speaking",
      meta: "The companion is actively answering.",
      tone: "busy" as const,
    };
  }

  return {
    label: "Listening",
    meta: "Wake her with voice, click, or fallback chat.",
    tone: "ready" as const,
  };
}

export function PetShell({
  avatarRef,
  character,
  onAction,
  onSubmitChat,
  onRequestContextMenu,
  runtime,
}: {
  avatarRef: RefObject<AvatarStageHandle | null>;
  character: CharacterPreset;
  onAction: (action: string) => void;
  onSubmitChat: (text: string) => void;
  onRequestContextMenu: () => void;
  runtime: BackendRuntime | null;
}) {
  const uiMode = useShellStore((state) => state.uiMode);
  const cameraPreset = useShellStore((state) => state.cameraPreset);
  const avatarExpression = useShellStore((state) => state.avatarExpression);
  const debugOverlayVisible = useShellStore((state) => state.debugOverlayVisible);
  const touchInteraction = useShellStore((state) => state.touchInteraction);
  const setChatVisible = useShellStore((state) => state.setChatVisible);
  const setSettingsVisible = useShellStore((state) => state.setSettingsVisible);
  const setUiMode = useShellStore((state) => state.setUiMode);
  const { bindInteractiveRegion } = useClickThrough();

  const interactive = bindInteractiveRegion();
  const compact = uiMode === "pet";
  const status = useMemo(() => statusFromRuntime(runtime), [runtime]);

  return (
    <motion.section
      {...shellMotion}
      className={`pointer-events-none relative z-20 ${compact ? "h-[620px] w-[560px]" : "h-[670px] w-[500px]"}`}
    >
      <div
        {...interactive}
        data-tauri-drag-region
        className={`pointer-events-auto absolute z-10 rounded-full bg-transparent ${
          compact ? "right-[84px] top-10 h-12 w-[220px]" : "right-[86px] top-8 h-12 w-[260px]"
        }`}
      />

      <StatusChip {...status} />

      <div className={`absolute ${compact ? "bottom-[112px] right-6" : "bottom-[86px] right-0"}`} {...interactive}>
        <AvatarHalo
          glowPrimary={character.glow.primary}
          glowSecondary={character.glow.secondary}
          compact={compact}
          className={compact ? "h-[420px] w-[312px]" : "h-[520px] w-[420px]"}
        >
          <AvatarOrb
            compact={compact}
            onClick={() => {
              touchInteraction();
              if (compact) {
                setChatVisible(true);
              }
            }}
            onDoubleClick={() => {
              onAction("toggle-mute");
            }}
            onContextMenu={(event) => {
              event.preventDefault();
              touchInteraction();
              onRequestContextMenu();
            }}
          >
            <AvatarStage
              ref={avatarRef}
              character={character}
              cameraPreset={cameraPreset}
              expression={avatarExpression}
              debugVisible={debugOverlayVisible}
            />
          </AvatarOrb>
        </AvatarHalo>
      </div>

      <div className={`absolute bottom-0 right-0 flex items-end gap-3 ${compact ? "w-[520px]" : "w-[500px]"}`} {...interactive}>
        <div className="pointer-events-auto min-w-0 flex-1 rounded-[28px] border border-white/10 bg-black/32 px-4 py-3 backdrop-blur-sm">
          <div className="text-sm font-semibold text-white">{character.name}</div>
          <div className="mt-1 text-xs text-shell-muted">
            {compact ? "Voice-first portrait shell with a clean head-and-shoulders crop." : "Expanded mode keeps the avatar visible while text stays secondary."}
          </div>
        </div>

        <ShellDock
          micMuted={runtime?.mic_state === "muted"}
          compact={compact}
          onMute={() => onAction("toggle-mute")}
          onInterrupt={() => onAction("interrupt")}
          onChat={() => {
            setSettingsVisible(false);
            setChatVisible(uiMode === "pet");
          }}
          onSettings={() => setSettingsVisible(true)}
          onExpand={() => {
            setUiMode(compact ? "expand" : "pet");
          }}
        />
      </div>

      <button
        type="button"
        className="sr-only"
        onClick={() => {
          void onSubmitChat("Hello");
        }}
      >
        Hidden chat submit helper
      </button>
    </motion.section>
  );
}
