export type CameraPresetId = "petBust" | "chatBust" | "fullBody";

export interface CameraPreset {
  scale: number;
  position: { x: number; y: number };
}

export interface CharacterPreset {
  id: string;
  name: string;
  modelPath: string;
  modelName: string;
  defaultExpression: string;
  panelSide: "left" | "right";
  glow: {
    primary: string;
    secondary: string;
  };
  cameraPresets: Record<CameraPresetId, CameraPreset>;
}

export const CHARACTER_PRESETS: Record<string, CharacterPreset> = {
  march7th: {
    id: "march7th",
    name: "March 7th",
    modelPath: "/runtime-assets/models/march7th_tauri/",
    modelName: "march7th.model3.json",
    defaultExpression: "neutral",
    panelSide: "left",
    glow: {
      primary: "#79E7FF",
      secondary: "#FF6FC9",
    },
    cameraPresets: {
      petBust: {
        scale: 1.88,
        position: { x: 1.22, y: 0.52 },
      },
      chatBust: {
        scale: 1.38,
        position: { x: 0.9, y: 0.16 },
      },
      fullBody: {
        scale: 0.92,
        position: { x: 0.58, y: -0.06 },
      },
    },
  },
};

export function normalizeCharacterId(value?: string | null): string {
  const normalized = (value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "")
    .trim();

  if (normalized.includes("march7")) {
    return "march7th";
  }

  return CHARACTER_PRESETS[normalized] ? normalized : "march7th";
}

export function getCharacterPreset(characterId?: string | null): CharacterPreset {
  return CHARACTER_PRESETS[normalizeCharacterId(characterId)] ?? CHARACTER_PRESETS.march7th;
}
