"use client";
export type HealthState = "healthy" | "degraded" | "down" | "unknown";

const STATE_CONFIG: Record<HealthState, { color: string; icon: string; label: string }> = {
  healthy:  { color: "var(--good)",    icon: "●", label: "Healthy" },
  degraded: { color: "var(--warn)",    icon: "▲", label: "Degraded" },
  down:     { color: "var(--bad)",     icon: "✕", label: "Down" },
  unknown:  { color: "var(--muted)",   icon: "?", label: "Unknown" },
};

interface Props {
  state: HealthState;
  label?: string;
  size?: "sm" | "md";
}

export default function StatusPill({ state, label, size = "md" }: Props) {
  const cfg = STATE_CONFIG[state] ?? STATE_CONFIG.unknown;
  const fs = size === "sm" ? "0.75rem" : "0.82rem";
  return (
    <span
      aria-label={label ?? cfg.label}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "5px",
        padding: size === "sm" ? "2px 8px" : "4px 10px",
        borderRadius: "999px",
        background: `${cfg.color}18`,
        border: `1px solid ${cfg.color}44`,
        color: cfg.color,
        fontSize: fs,
        fontWeight: 600,
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ fontSize: size === "sm" ? "0.5rem" : "0.6rem" }} aria-hidden="true">{cfg.icon}</span>
      {label ?? cfg.label}
    </span>
  );
}
