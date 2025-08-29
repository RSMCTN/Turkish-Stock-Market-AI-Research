"use client"

import * as React from "react"

interface ProgressProps {
  value: number;
  className?: string;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, ...props }, ref) => (
    <div
      ref={ref}
      className={`relative h-2 w-full overflow-hidden rounded-full bg-slate-200 ${className}`}
      {...props}
    >
      <div
        className="h-full bg-gradient-to-r from-emerald-400 to-emerald-600 transition-all duration-300 ease-in-out rounded-full"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
)

Progress.displayName = "Progress"

export { Progress }