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
      className={`group relative overflow-hidden bg-transparent text-left transition ${
        compact
          ? "h-[420px] w-[312px] rounded-[38px]"
          : "h-[520px] w-[420px] rounded-[48px] border border-white/10"
      }`}
    >
      <div
        className={`absolute inset-0 opacity-70 ${
          compact
            ? "bg-[linear-gradient(180deg,rgba(255,255,255,0.06),transparent_26%,rgba(5,10,18,0.18)_100%)]"
            : "bg-gradient-to-b from-white/[0.08] via-transparent to-black/10"
        }`}
      />
      {children}
    </button>
  );
}
