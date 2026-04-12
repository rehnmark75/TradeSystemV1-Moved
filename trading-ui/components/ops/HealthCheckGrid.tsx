"use client";

export interface HealthCheckItem {
  service?: string;
  probe?: string;
  name?: string;
  status?: string;
  latency_ms?: number;
  consecutive_failures?: number;
  error?: string;
  last_run?: string;
  response_time?: number;
}

interface Props { checks: HealthCheckItem[]; }

const statusColor = (s?: string) => {
  if (!s) return "var(--muted)";
  if (s === "healthy" || s === "ok" || s === "up") return "var(--good)";
  if (s === "degraded" || s === "warning") return "var(--warn)";
  return "var(--bad)";
};

export default function HealthCheckGrid({ checks }: Props) {
  if (!checks.length) return <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>No health checks available.</p>;
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
        <thead>
          <tr style={{ borderBottom: "2px solid var(--border)" }}>
            {["Service", "Status", "Latency", "Failures", "Error"].map(h => (
              <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "var(--muted)", fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {checks.map((c, i) => {
            const color = statusColor(c.status);
            return (
              <tr key={i} style={{ borderBottom: "1px solid var(--border)", background: i % 2 ? "#fafaf8" : "transparent" }}>
                <td style={{ padding: "8px 12px", fontWeight: 500 }}>{c.service ?? c.name}</td>
                <td style={{ padding: "8px 12px" }}>
                  <span style={{ color, fontWeight: 700, fontSize: "0.8rem" }}>{(c.status ?? "—").toUpperCase()}</span>
                </td>
                <td style={{ padding: "8px 12px", fontVariantNumeric: "tabular-nums" }}>
                  {(c.latency_ms ?? c.response_time) !== undefined ? `${(c.latency_ms ?? c.response_time)?.toFixed(0)} ms` : "—"}
                </td>
                <td style={{ padding: "8px 12px", color: (c.consecutive_failures ?? 0) > 0 ? "var(--bad)" : undefined }}>
                  {c.consecutive_failures ?? 0}
                </td>
                <td style={{ padding: "8px 12px", color: "var(--muted)", fontSize: "0.78rem", maxWidth: "260px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {c.error ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
