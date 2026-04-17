import {
  ForwardedRef,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { CameraPresetId, CharacterPreset } from "../../characters/presets";

export interface AvatarStageHandle {
  setExpression: (expression: string) => void;
  setLipSync: (value: number) => void;
  setCameraPreset: (preset: CameraPresetId) => void;
  playMotion: (group: string, index?: number) => void;
  setIdleEnabled: (enabled: boolean) => void;
  connectAudioElement: (audio: HTMLAudioElement) => void;
  stopLipSync: () => void;
}

const CANVAS_ID = "live2d-stage";
let runtimeScriptsPromise: Promise<void> | null = null;

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[data-runtime-src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === "true") {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = false;
    script.dataset.runtimeSrc = src;
    script.addEventListener(
      "load",
      () => {
        script.dataset.loaded = "true";
        resolve();
      },
      { once: true },
    );
    script.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)), { once: true });
    document.head.appendChild(script);
  });
}

async function ensureLive2DRuntime() {
  if (!runtimeScriptsPromise) {
    runtimeScriptsPromise = (async () => {
      await loadScript("/runtime-assets/live2d_sdk_web/Core/live2dcubismcore.min.js");
      await loadScript("/live2d/live2d.js");
    })();
  }

  return runtimeScriptsPromise;
}

function AvatarStageInner(
  {
    character,
    cameraPreset,
    expression,
    debugVisible,
  }: {
    character: CharacterPreset;
    cameraPreset: CameraPresetId;
    expression: string;
    debugVisible: boolean;
  },
  ref: ForwardedRef<AvatarStageHandle>,
) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);
  const activePreset = useMemo(() => character.cameraPresets[cameraPreset], [cameraPreset, character.cameraPresets]);

  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      await ensureLive2DRuntime();
      if (cancelled) return;

      await window.Live2DManager?.init({
        canvasId: CANVAS_ID,
        modelPath: character.modelPath,
        modelName: character.modelName,
        defaultExpression: character.defaultExpression,
        scale: activePreset.scale,
        position: activePreset.position,
        debug: debugVisible,
      });

      if (cancelled) return;
      setReady(true);
    };

    void init();

    return () => {
      cancelled = true;
    };
  }, [activePreset.position, activePreset.scale, character.defaultExpression, character.modelName, character.modelPath, debugVisible]);

  useEffect(() => {
    if (!ready) return;
    window.Live2DManager?.setScale(activePreset.scale);
    window.Live2DManager?.setPosition(activePreset.position.x, activePreset.position.y);
  }, [activePreset, ready]);

  useEffect(() => {
    if (!ready || !expression) return;
    window.Live2DManager?.setExpression(expression);
  }, [expression, ready]);

  useEffect(() => {
    if (!ready) return;
    if (debugVisible) {
      window.Live2DManager?.toggleDebug?.();
    }
  }, [debugVisible, ready]);

  useEffect(() => {
    if (!stageRef.current || !ready) return;
    const observer = new ResizeObserver(() => {
      window.Live2DManager?.setScale(activePreset.scale);
      window.Live2DManager?.setPosition(activePreset.position.x, activePreset.position.y);
    });
    observer.observe(stageRef.current);
    return () => observer.disconnect();
  }, [activePreset, ready]);

  useImperativeHandle(
    ref,
    () => ({
      setExpression(nextExpression) {
        window.Live2DManager?.setExpression(nextExpression);
      },
      setLipSync(value) {
        window.Live2DManager?.setLipSync(value);
      },
      setCameraPreset(nextPreset) {
        const preset = character.cameraPresets[nextPreset];
        window.Live2DManager?.setScale(preset.scale);
        window.Live2DManager?.setPosition(preset.position.x, preset.position.y);
      },
      playMotion(group, index) {
        window.Live2DManager?.playMotion(group, index);
      },
      setIdleEnabled(enabled) {
        if (!enabled) {
          window.Live2DManager?.setLipSync(0);
        }
      },
      connectAudioElement(audio) {
        window.Live2DManager?.connectAudioElement?.(audio);
      },
      stopLipSync() {
        window.Live2DManager?.stopLipSync?.();
        window.Live2DManager?.setLipSync(0);
      },
    }),
    [character.cameraPresets],
  );

  return (
    <div ref={stageRef} className="absolute inset-0">
      <canvas id={CANVAS_ID} className="h-full w-full bg-transparent" />
    </div>
  );
}

export const AvatarStage = forwardRef(AvatarStageInner);
