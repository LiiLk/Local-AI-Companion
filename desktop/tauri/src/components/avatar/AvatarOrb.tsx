import { ReactNode } from "react";

export function AvatarOrb({
  children,
  compact,
  onClick,
  onDoubleClick,
  onContextMenu,
}: {
  children: ReactNode;
  compact: boolean;
  onClick?: () => void;
  onDoubleClick?: () => void;
  onContextMenu?: React.MouseEventHandler<HTMLButtonElement>;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      onContextMenu={onContextMenu}
      className={`group relative overflow-hidden border border-white/10 bg-transparent text-left transition ${
        compact
          ? "h-[228px] w-[228px] rounded-[44px]"
          : "h-[520px] w-[420px] rounded-[48px]"
      }`}
    >
      <div className="absolute inset-0 bg-gradient-to-b from-white/[0.08] via-transparent to-black/10 opacity-70" />
      {children}
    </button>
  );
}
