"use client";
import { useState } from "react";
import ConfirmDialog from "./ConfirmDialog";

export interface AlertItem {
  id: string;
  severity: "info" | "warning" | "error" | "critical";
  source?: string;
  message: string;
  created_at?: string;
  status?: string;
}

const SEV_COLOR: Record<string, string> = {
  info:     "#1971c2",
  warning:  "var(--warn)",
  error:    "var(--bad)",
  critical: "#8b1e2b",
};

interface Props {
  alert: AlertItem;
  onAck?: (id: string) => Promise<void>;
  onResolve?: (id: string) => Promise<void>;
}

function fmtTs(ts?: string) {
  if (!ts) return "";
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

export default function AlertRow({ alert: a, onAck, onResolve }: Props) {
  const [dialog, setDialog] = useState<"ack" | "resolve" | null>(null);
  const [loading, setLoading] = useState(false);
  const color = SEV_COLOR[a.severity] ?? "var(--muted)";
  const isResolved = a.status === "resolved";

  const act = async (fn?: (id: string) => Promise<void>) => {
    if (!fn) return;
    setLoading(true);
    try { await fn(a.id); } finally { setLoading(false); setDialog(null); }
  };

  return (
    <>
      <div style={{
        display: "flex", gap: "12px", alignItems: "flex-start",
        padding: "10px 14px", borderRadius: "8px",
        background: isResolved ? "#fafaf8" : `${color}08`,
        border: `1px solid ${color}33`,
        opacity: isResolved ? 0.55 : 1,
        marginBottom: "6px",
      }}>
        <span style={{ fontSize: "0.7rem", fontWeight: 700, color, textTransform: "uppercase", letterSpacing: "0.06em", paddingTop: "2px", minWidth: "54px" }}>
          {a.severity}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: "0.85rem", fontWeight: 500, wordBreak: "break-word" }}>{a.message}</div>
          <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "2px" }}>
            {a.source && <span>{a.source} · </span>}{fmtTs(a.created_at)}
          </div>
        </div>
        {!isResolved && (
          <div style={{ display: "flex", gap: "6px", flexShrink: 0 }}>
            {a.status !== "acknowledged" && onAck && (
              <button onClick={() => setDialog("ack")} style={{ fontSize: "0.75rem", padding: "3px 9px", border: "1px solid var(--border)", borderRadius: "5px", background: "transparent", cursor: "pointer" }}>Ack</button>
            )}
            {onResolve && (
              <button onClick={() => setDialog("resolve")} style={{ fontSize: "0.75rem", padding: "3px 9px", border: "1px solid #c5e8d0", borderRadius: "5px", background: "#f0faf5", color: "var(--good)", cursor: "pointer" }}>Resolve</button>
            )}
          </div>
        )}
      </div>
      <ConfirmDialog
        open={dialog === "ack"}
        title="Acknowledge alert?"
        description={`Marking "${a.message.substring(0, 80)}${a.message.length > 80 ? "…" : ""}" as acknowledged.`}
        confirmLabel="Acknowledge"
        loading={loading}
        onConfirm={() => act(onAck)}
        onCancel={() => setDialog(null)}
      />
      <ConfirmDialog
        open={dialog === "resolve"}
        title="Resolve alert?"
        description={`Marking "${a.message.substring(0, 80)}${a.message.length > 80 ? "…" : ""}" as resolved.`}
        confirmLabel="Resolve"
        loading={loading}
        onConfirm={() => act(onResolve)}
        onCancel={() => setDialog(null)}
      />
    </>
  );
}
