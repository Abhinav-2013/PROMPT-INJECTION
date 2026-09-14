import type { ReactNode } from 'react'

interface GlassPanelProps {
  children: ReactNode
  className?: string
}

function GlassPanel({
  children,
  className = '',
}: GlassPanelProps) {
  return (
    <div
      className={`
        rounded-[2rem]
        border border-white/70
        bg-white/35
        shadow-[0_20px_60px_rgba(30,41,59,0.06)]
        backdrop-blur-2xl
        ${className}
      `}
    >
      {children}
    </div>
  )
}

export default GlassPanel