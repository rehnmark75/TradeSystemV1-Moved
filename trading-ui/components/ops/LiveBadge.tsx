"use client";
import { useEffect, useState } from "react";

interface Props {
  lastUpdated: Date | null;
  staleAfterMs?: number;
}

function ago(d: Date): string {
  const s = Math.floor((Date.now() - d.getTime()) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

export default function LiveBadge({ lastUpdated, staleAfterMs = 30000 }: Props) {
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick(n => n + 1), 5000);
    return () => clearInterval(id);
  }, []);

  const isStale = lastUpdated && Date.now() - lastUpdated.getTime() > staleAfterMs;
  const color = !lastUpdated ? "var(--muted)" : isStale ? "var(--warn)" : "var(--good)";

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "6px", fontSize: "0.78rem", color, fontWeight: 500 }}>
      <span style={{
        width: "7px", height: "7px", borderRadius: "50%", background: color,
        animation: !isStale && lastUpdated ? "livepin 2s ease-in-out infinite" : undefined,
      }} />
      {lastUpdated ? ago(lastUpdated) : "—"}
    </span>
  );
}
