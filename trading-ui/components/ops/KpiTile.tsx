"use client";
import Sparkline from "./Sparkline";

interface Props {
  label: string;
  value: string | number;
  delta?: string;
  deltaPositive?: boolean;
  sparkData?: number[];
  sparkColor?: string;
  accent?: string;
  loading?: boolean;
}

export default function KpiTile({ label, value, delta, deltaPositive, sparkData, sparkColor, accent, loading }: Props) {
  return (
    <div style={{
      background: "var(--panel)",
      border: "1px solid var(--border)",
      borderRadius: "12px",
      padding: "14px 16px",
      display: "flex",
      flexDirection: "column",
      gap: "4px",
      position: "relative",
      overflow: "hidden",
    }}>
      {accent && (
        <div style={{ position: "absolute", top: 0, left: 0, bottom: 0, width: "3px", background: accent, borderRadius: "12px 0 0 12px" }} />
      )}
      <span style={{ fontSize: "0.75rem", color: "var(--muted)", fontWeight: 500, letterSpacing: "0.03em" }}>{label}</span>
      {loading ? (
        <div style={{ height: "28px", background: "#e4ddd2", borderRadius: "6px", animation: "pulse 1.4s ease infinite" }} />
      ) : (
        <span style={{ fontSize: "1.5rem", fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>{value}</span>
      )}
      {(delta || sparkData) && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "4px" }}>
          {delta && (
            <span style={{ fontSize: "0.78rem", color: deltaPositive ? "var(--good)" : "var(--bad)", fontWeight: 600 }}>
              {deltaPositive ? "▲" : "▼"} {delta}
            </span>
          )}
          {sparkData && <Sparkline data={sparkData} width={80} height={24} color={sparkColor ?? "var(--accent)"} />}
        </div>
      )}
    </div>
  );
}
