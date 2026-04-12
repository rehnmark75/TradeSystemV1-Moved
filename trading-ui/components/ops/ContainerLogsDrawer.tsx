"use client";
import { useEffect, useRef, useState } from "react";

interface Props {
  containerName: string | null;
  onClose: () => void;
}

export default function ContainerLogsDrawer({ containerName, onClose }: Props) {
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lines, setLines] = useState(100);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerName) return;
    setLoading(true);
    setError(null);
    fetch(`/trading/api/infra/containers/${encodeURIComponent(containerName)}/logs?lines=${lines}`)
      .then(r => r.json())
      .then(d => {
        setLogs(d.logs ?? d.log_lines ?? (typeof d === "string" ? d.split("\n") : []));
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, [containerName, lines]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logs]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  if (!containerName) return null;

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 900, display: "flex" }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ flex: 1, background: "rgba(0,0,0,.3)" }} onClick={onClose} />
      <div style={{
        width: "min(700px, 90vw)",
        background: "#0d1117",
        color: "#c9d1d9",
        display: "flex",
        flexDirection: "column",
        boxShadow: "-4px 0 24px rgba(0,0,0,.4)",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 18px", borderBottom: "1px solid #30363d" }}>
          <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "0.95rem" }}>{containerName} logs</span>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <select value={lines} onChange={e => setLines(Number(e.target.value))}
              style={{ background: "#161b22", color: "#c9d1d9", border: "1px solid #30363d", borderRadius: "6px", padding: "4px 8px", fontSize: "0.8rem" }}>
              {[50, 100, 200, 500].map(n => <option key={n} value={n}>{n} lines</option>)}
            </select>
            <button onClick={onClose} style={{ background: "transparent", border: "none", color: "#c9d1d9", cursor: "pointer", fontSize: "1.2rem" }}>✕</button>
          </div>
        </div>
        <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px", fontFamily: "monospace", fontSize: "0.78rem", lineHeight: 1.6 }}>
          {loading && <div style={{ color: "#8b949e" }}>Loading…</div>}
          {error && <div style={{ color: "#f85149" }}>Error: {error}</div>}
          {!loading && !error && logs.length === 0 && <div style={{ color: "#8b949e" }}>No logs available.</div>}
          {logs.map((line, i) => {
            const isErr = /error|exception|critical/i.test(line);
            const isWarn = /warn/i.test(line);
            return (
              <div key={i} style={{ color: isErr ? "#f85149" : isWarn ? "#e3b341" : "#c9d1d9", whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
                {line}
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
}
