"use client";
import { useState } from "react";
import StatusPill from "./StatusPill";
import type { HealthState } from "./StatusPill";
import ConfirmDialog from "./ConfirmDialog";

export interface ContainerInfo {
  name: string;
  image?: string;
  state?: string;
  status?: string;
  uptimeSeconds?: number;
  restartCount?: number;
  cpuPercent?: number;
  memUsageMb?: number;
  memLimitMb?: number;
  is_critical?: boolean;
  warnings?: string[];
  errors?: string[];
}

interface Props {
  container: ContainerInfo;
  onViewLogs?: (name: string) => void;
  onRestart?: (name: string) => Promise<void>;
}

function fmtUptime(s?: number): string {
  if (!s) return "—";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

function toHealthState(state?: string, status?: string): HealthState {
  if (status === "healthy") return "healthy";
  if (status === "degraded") return "degraded";
  if (status === "down" || state === "exited") return "down";
  if (state === "running") return "healthy";
  return "unknown";
}

export default function ContainerCard({ container: c, onViewLogs, onRestart }: Props) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const hs = toHealthState(c.state, c.status);
  const borderAccent = hs === "down" ? "var(--bad)" : hs === "degraded" ? "var(--warn)" : c.is_critical && hs !== "healthy" ? "var(--bad)" : "var(--border)";

  const doRestart = async () => {
    if (!onRestart) return;
    setRestarting(true);
    try { await onRestart(c.name); } finally { setRestarting(false); setConfirmOpen(false); }
  };

  return (
    <>
      <div style={{
        background: "var(--panel)",
        border: `1px solid ${borderAccent}`,
        borderLeft: `3px solid ${borderAccent}`,
        borderRadius: "10px",
        padding: "14px 16px",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        position: "relative",
      }}>
        {c.is_critical && (
          <span style={{ position: "absolute", top: "10px", right: "12px", fontSize: "0.7rem", fontWeight: 700, color: "var(--bad)", letterSpacing: "0.05em" }}>CRITICAL</span>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", paddingRight: "60px" }}>
          <StatusPill state={hs} size="sm" />
          <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: "0.9rem" }}>{c.name}</span>
        </div>
        {c.image && <span style={{ fontSize: "0.72rem", color: "var(--muted)", fontFamily: "monospace" }}>{c.image.split(":")[0]}</span>}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px", fontSize: "0.78rem" }}>
          <span><span style={{ color: "var(--muted)" }}>Uptime </span>{fmtUptime(c.uptimeSeconds)}</span>
          <span><span style={{ color: "var(--muted)" }}>Restarts </span>{c.restartCount ?? 0}</span>
          {c.cpuPercent !== undefined && <span><span style={{ color: "var(--muted)" }}>CPU </span>{c.cpuPercent.toFixed(1)}%</span>}
          {c.memUsageMb !== undefined && <span><span style={{ color: "var(--muted)" }}>Mem </span>{c.memUsageMb.toFixed(0)} MB</span>}
        </div>
        {(c.warnings?.length || c.errors?.length) ? (
          <div style={{ fontSize: "0.75rem" }}>
            {c.errors?.slice(0, 1).map((e, i) => <div key={i} style={{ color: "var(--bad)" }}>✕ {e}</div>)}
            {c.warnings?.slice(0, 1).map((w, i) => <div key={i} style={{ color: "var(--warn)" }}>▲ {w}</div>)}
          </div>
        ) : null}
        <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
          {onViewLogs && (
            <button onClick={() => onViewLogs(c.name)} style={{ flex: 1, padding: "5px 0", fontSize: "0.78rem", border: "1px solid var(--border)", borderRadius: "6px", background: "transparent", cursor: "pointer", fontWeight: 500 }}>
              Logs
            </button>
          )}
          {onRestart && (
            <button onClick={() => setConfirmOpen(true)} disabled={restarting} style={{ flex: 1, padding: "5px 0", fontSize: "0.78rem", border: "1px solid #ecc", borderRadius: "6px", background: "#fff5f5", color: "var(--bad)", cursor: "pointer", fontWeight: 500 }}>
              {restarting ? "…" : "Restart"}
            </button>
          )}
        </div>
      </div>
      <ConfirmDialog
        open={confirmOpen}
        title={`Restart ${c.name}?`}
        description="The container will stop and restart. Active connections will be dropped. This action has no auth layer — operator use only."
        confirmLabel="Restart"
        danger
        loading={restarting}
        onConfirm={doRestart}
        onCancel={() => setConfirmOpen(false)}
      />
    </>
  );
}
