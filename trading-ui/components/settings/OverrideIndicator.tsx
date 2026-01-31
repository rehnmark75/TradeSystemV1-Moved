"use client";

interface OverrideIndicatorProps {
  active?: boolean;
}

export default function OverrideIndicator({ active }: OverrideIndicatorProps) {
  return <span className={`override-indicator ${active ? "active" : ""}`} />;
}
