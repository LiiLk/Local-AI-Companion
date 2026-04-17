import { ReactNode } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

export function AvatarHalo({
  children,
  glowPrimary,
  glowSecondary,
  className = "",
}: {
  children: ReactNode;
  glowPrimary: string;
  glowSecondary: string;
  className?: string;
}) {
  return (
    <div
      className={`relative ${className}`}
      onMouseDown={(event) => {
        if (event.button === 0) {
          void getCurrentWindow().startDragging();
        }
      }}
    >
      <div
        className="absolute inset-2 rounded-full blur-2xl opacity-75"
        style={{
          background: `radial-gradient(circle at 50% 30%, ${glowPrimary}55, transparent 56%), radial-gradient(circle at 60% 80%, ${glowSecondary}44, transparent 44%)`,
        }}
      />
      <div className="absolute inset-0 rounded-full border border-white/10 bg-white/[0.02] shadow-halo" />
      {children}
    </div>
  );
}
