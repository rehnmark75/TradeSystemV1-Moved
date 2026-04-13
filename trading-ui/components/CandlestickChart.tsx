"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  SeriesMarker,
  Time,
} from "lightweight-charts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Trade {
  id: number;
  timestamp: string;
  entry_price: number;
  direction: string;
  profit_loss: number | null;
  status: string;
  pnl_currency: string | null;
  environment: string;
  strategy: string | null;
  signal_type: string | null;
  confidence_score: number | null;
}

interface RawCandle {
  open: number;
  high: number;
  low: number;
  close: number;
  start_time?: string;
  timestamp?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function epicToLabel(epic: string): string {
  const parts = epic.split(".");
  if (parts.length >= 3) {
    const raw = parts[2];
    if (raw.length === 6) return `${raw.slice(0, 3)}/${raw.slice(3)}`;
  }
  return epic;
}

function toUnixSeconds(ts: string): number {
  return Math.floor(new Date(ts).getTime() / 1000);
}

function pnlText(trade: Trade): string {
  if (trade.profit_loss == null) return trade.status;
  const sign = trade.profit_loss >= 0 ? "+" : "";
  return `${sign}${trade.profit_loss.toFixed(1)}`;
}

function buildMarkers(trades: Trade[]): SeriesMarker<Time>[] {
  return trades
    .filter((t) => t.entry_price != null && t.timestamp)
    .map((t) => {
      const isBuy = t.direction?.toUpperCase() === "BUY";
      const won = t.profit_loss != null && t.profit_loss > 0;
      const closed = t.status === "closed";
      return {
        time: toUnixSeconds(t.timestamp) as Time,
        position: isBuy ? "belowBar" : "aboveBar",
        shape: isBuy ? "arrowUp" : "arrowDown",
        color: closed ? (won ? "#22c55e" : "#ef4444") : "#f59e0b",
        text: pnlText(t),
        size: 1,
      } satisfies SeriesMarker<Time>;
    });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CandlestickChart() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  const [epics, setEpics] = useState<string[]>([]);
  const [selectedEpic, setSelectedEpic] = useState<string>("");
  const [timeframe, setTimeframe] = useState<"5" | "15">("5");
  const [configSet, setConfigSet] = useState<"live" | "demo">("demo");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tradeCount, setTradeCount] = useState(0);
  const [dateFrom, setDateFrom] = useState<string>("");
  const [dateTo, setDateTo] = useState<string>("");

  // Fetch enabled epics when configSet changes
  useEffect(() => {
    fetch(`/trading/api/chart/epics?config_set=${configSet}`)
      .then((r) => r.json())
      .then((d) => {
        const list: string[] = d.epics ?? [];
        setEpics(list);
        if (list.length > 0 && !list.includes(selectedEpic)) {
          setSelectedEpic(list[0]);
        }
      })
      .catch(() => setError("Failed to load epics"));
  }, [configSet]); // eslint-disable-line react-hooks/exhaustive-deps

  // Create chart once on mount
  useEffect(() => {
    if (!chartContainerRef.current) return;

    let chart: IChartApi;
    let canceled = false;

    import("lightweight-charts").then(({ createChart, ColorType }) => {
      if (canceled || !chartContainerRef.current) return;

      chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: "#0f172a" },
          textColor: "#94a3b8",
        },
        grid: {
          vertLines: { color: "#1e293b" },
          horzLines: { color: "#1e293b" },
        },
        crosshair: { mode: 1 },
        rightPriceScale: { borderColor: "#334155" },
        timeScale: {
          borderColor: "#334155",
          timeVisible: true,
          secondsVisible: false,
        },
        width: chartContainerRef.current.clientWidth,
        height: chartContainerRef.current.clientHeight,
      });

      seriesRef.current = chart.addCandlestickSeries({
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#22c55e",
        borderDownColor: "#ef4444",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
      });

      chartRef.current = chart;

      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          chart.applyOptions({ width, height });
        }
      });
      ro.observe(chartContainerRef.current);
      (chartRef.current as any).__ro = ro;
    });

    return () => {
      canceled = true;
      if (chart) {
        (chart as any).__ro?.disconnect();
        chart.remove();
      }
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Load candles + trades whenever epic, timeframe, or date range changes
  const loadData = useCallback(async () => {
    if (!selectedEpic || !seriesRef.current) return;
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        epic: selectedEpic,
        timeframe: timeframe,
      });
      if (dateFrom) params.set("from", new Date(dateFrom).toISOString());
      if (dateTo) params.set("to", new Date(dateTo + "T23:59:59").toISOString());
      if (!dateFrom && !dateTo) params.set("limit", "1000");

      const candleRes = await fetch(`/trading/api/chart/candles?${params}`);
      if (!candleRes.ok) throw new Error("Candles fetch failed");
      const candleData: RawCandle[] = await candleRes.json();

      if (!Array.isArray(candleData) || candleData.length === 0) {
        seriesRef.current.setData([]);
        seriesRef.current.setMarkers([]);
        setTradeCount(0);
        setLoading(false);
        return;
      }

      const bars: CandlestickData[] = candleData
        .map((c) => {
          const ts = c.start_time ?? c.timestamp ?? "";
          return {
            time: toUnixSeconds(ts) as Time,
            open: Number(c.open),
            high: Number(c.high),
            low: Number(c.low),
            close: Number(c.close),
          };
        })
        .filter((b) => (b.time as number) > 0)
        .sort((a, b) => (a.time as number) - (b.time as number));

      // Deduplicate by time
      const seen = new Set<number>();
      const uniqueBars = bars.filter((b) => {
        const t = b.time as number;
        if (seen.has(t)) return false;
        seen.add(t);
        return true;
      });

      seriesRef.current.setData(uniqueBars);
      chartRef.current?.timeScale().fitContent();

      const firstTs = candleData[0].start_time ?? candleData[0].timestamp ?? "";
      const lastTs =
        candleData[candleData.length - 1].start_time ??
        candleData[candleData.length - 1].timestamp ??
        "";

      const tradeRes = await fetch(
        `/trading/api/chart/trades?epic=${encodeURIComponent(selectedEpic)}&from=${firstTs}&to=${lastTs}&environment=${configSet}`
      );
      if (tradeRes.ok) {
        const tradeData = await tradeRes.json();
        const trades: Trade[] = tradeData.trades ?? [];
        setTradeCount(trades.length);
        seriesRef.current.setMarkers(buildMarkers(trades));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [selectedEpic, timeframe, dateFrom, dateTo]);

  useEffect(() => {
    const timer = setTimeout(() => loadData(), 50);
    return () => clearTimeout(timer);
  }, [loadData]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#0f172a", color: "#e2e8f0" }}>
      {/* Controls */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "8px 16px",
        borderBottom: "1px solid #1e293b",
        flexShrink: 0,
        flexWrap: "wrap",
      }}>
        {/* Config set toggle */}
        <div style={{ display: "flex", gap: "4px" }}>
          {(["live", "demo"] as const).map((cs) => (
            <button
              key={cs}
              onClick={() => setConfigSet(cs)}
              style={{
                padding: "4px 10px",
                borderRadius: "4px",
                border: "none",
                cursor: "pointer",
                fontSize: "12px",
                fontWeight: 600,
                background: configSet === cs ? "#3b82f6" : "#1e293b",
                color: configSet === cs ? "#fff" : "#64748b",
              }}
            >
              {cs.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Epic selector */}
        <select
          value={selectedEpic}
          onChange={(e) => setSelectedEpic(e.target.value)}
          style={{
            background: "#1e293b",
            color: "#f1f5f9",
            border: "1px solid #334155",
            borderRadius: "6px",
            padding: "5px 10px",
            fontSize: "13px",
            cursor: "pointer",
            minWidth: "140px",
          }}
        >
          {epics.map((e) => (
            <option key={e} value={e}>
              {epicToLabel(e)} ({e.includes("MINI") ? "Mini" : "CEEM"})
            </option>
          ))}
        </select>

        {/* Timeframe toggle */}
        <div style={{ display: "flex", gap: "4px" }}>
          {(["5", "15"] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              style={{
                padding: "4px 10px",
                borderRadius: "4px",
                border: "none",
                cursor: "pointer",
                fontSize: "12px",
                fontWeight: 600,
                background: timeframe === tf ? "#6366f1" : "#1e293b",
                color: timeframe === tf ? "#fff" : "#64748b",
              }}
            >
              {tf}m
            </button>
          ))}
        </div>

        {/* Date range */}
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            style={{
              background: "#1e293b",
              color: "#f1f5f9",
              border: "1px solid #334155",
              borderRadius: "4px",
              padding: "4px 6px",
              fontSize: "12px",
              colorScheme: "dark",
            }}
          />
          <span style={{ color: "#475569", fontSize: "12px" }}>—</span>
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            style={{
              background: "#1e293b",
              color: "#f1f5f9",
              border: "1px solid #334155",
              borderRadius: "4px",
              padding: "4px 6px",
              fontSize: "12px",
              colorScheme: "dark",
            }}
          />
          {(dateFrom || dateTo) && (
            <button
              onClick={() => { setDateFrom(""); setDateTo(""); }}
              style={{
                padding: "3px 7px",
                borderRadius: "4px",
                border: "1px solid #334155",
                background: "transparent",
                color: "#64748b",
                cursor: "pointer",
                fontSize: "11px",
              }}
              title="Clear date range"
            >
              ✕
            </button>
          )}
        </div>

        {/* Refresh */}
        <button
          onClick={loadData}
          disabled={loading}
          style={{
            padding: "4px 10px",
            borderRadius: "4px",
            border: "1px solid #334155",
            background: "#1e293b",
            color: "#94a3b8",
            cursor: loading ? "not-allowed" : "pointer",
            fontSize: "12px",
          }}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>

        {tradeCount > 0 && (
          <span style={{ fontSize: "12px", color: "#64748b" }}>
            {tradeCount} trade{tradeCount !== 1 ? "s" : ""}
          </span>
        )}

        {/* Legend */}
        <div style={{ marginLeft: "auto", display: "flex", gap: "12px", fontSize: "11px", color: "#64748b" }}>
          <span><span style={{ color: "#22c55e" }}>▲</span> BUY win</span>
          <span><span style={{ color: "#ef4444" }}>▲▼</span> loss</span>
          <span><span style={{ color: "#22c55e" }}>▼</span> SELL win</span>
          <span><span style={{ color: "#f59e0b" }}>●</span> open</span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: "6px 16px", background: "#450a0a", color: "#fca5a5", fontSize: "13px", flexShrink: 0 }}>
          {error}
        </div>
      )}

      {/* Chart canvas */}
      <div ref={chartContainerRef} style={{ flex: 1, minHeight: 0 }} />
    </div>
  );
}
