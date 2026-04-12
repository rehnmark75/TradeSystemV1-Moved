"use client";
import { useEffect, useRef } from "react";

interface Props {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  danger?: boolean;
  loading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({ open, title, description, confirmLabel = "Confirm", danger = false, loading, onConfirm, onCancel }: Props) {
  const btnRef = useRef<HTMLButtonElement>(null);
  useEffect(() => { if (open) btnRef.current?.focus(); }, [open]);
  if (!open) return null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="cd-title"
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        background: "rgba(10,10,10,0.45)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
      onClick={e => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div style={{
        background: "var(--panel)", border: "1px solid var(--border)",
        borderRadius: "14px", padding: "28px 32px", maxWidth: "400px", width: "90%",
        boxShadow: "0 20px 40px rgba(0,0,0,.16)",
      }}>
        <h3 id="cd-title" style={{ margin: "0 0 10px", fontFamily: "'Space Grotesk', sans-serif", fontSize: "1.1rem" }}>{title}</h3>
        <p style={{ margin: "0 0 24px", color: "var(--muted)", fontSize: "0.9rem", lineHeight: 1.5 }}>{description}</p>
        <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end" }}>
          <button onClick={onCancel} style={{
            padding: "8px 18px", border: "1px solid var(--border)", borderRadius: "8px",
            background: "transparent", cursor: "pointer", fontWeight: 600,
          }}>Cancel</button>
          <button ref={btnRef} onClick={onConfirm} disabled={loading} style={{
            padding: "8px 18px", borderRadius: "8px", border: "none",
            background: danger ? "var(--bad)" : "var(--accent)",
            color: "#fff", cursor: loading ? "not-allowed" : "pointer",
            fontWeight: 600, opacity: loading ? 0.7 : 1,
          }}>
            {loading ? "…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
