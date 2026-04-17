import { ReactNode } from "react";

export function AvatarHalo({
  children,
  glowPrimary,
  glowSecondary,
  compact = false,
  className = "",
}: {
  children: ReactNode;
  glowPrimary: string;
  glowSecondary: string;
  compact?: boolean;
  className?: string;
}) {
  return (
    <div className={`relative ${className}`}>
      <div
        className={`absolute blur-[42px] opacity-80 ${compact ? "inset-x-5 inset-y-6 rounded-[42px]" : "inset-2 rounded-[42px]"}`}
        style={{
          background: compact
            ? `radial-gradient(circle at 52% 18%, ${glowPrimary}5f, transparent 40%), radial-gradient(circle at 55% 82%, ${glowSecondary}30, transparent 36%)`
            : `radial-gradient(circle at 50% 30%, ${glowPrimary}55, transparent 56%), radial-gradient(circle at 60% 80%, ${glowSecondary}44, transparent 44%)`,
        }}
      />
      {compact ? null : (
        <div className="absolute inset-0 rounded-[42px] border border-white/10 bg-white/[0.02] shadow-halo" />
      )}
      {children}
    </div>
  );
}
