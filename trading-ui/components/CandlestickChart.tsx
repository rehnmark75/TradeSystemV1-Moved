"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  SeriesMarker,
  Time,
} from "lightweight-charts";
import RejectionStageFilter from "./RejectionStageFilter";
import {
  ALL_STAGES,
  CATEGORIES,
  CATEGORY_ORDER,
  colorForStage,
  describeStage,
  stageToCategory,
  type CategoryKey,
} from "../lib/rejectionStyles";

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

interface RejectionRow {
  id: number;
  scan_timestamp: string;
  rejection_stage: string;
  rejection_reason: string;
  attempted_direction: string | null;
  current_price: number | null;
  confidence_score: number | null;
  potential_stop_loss: number | null;
  potential_take_profit: number | null;
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

function isBull(dir: string | null | undefined): boolean {
  if (!dir) return true;
  const d = dir.toUpperCase();
  return d === "BULL" || d === "BUY" || d === "LONG";
}

function buildRejectionMarkers(
  rejections: RejectionRow[],
  selected: Set<string>,
  timeframeSec: number
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const r of rejections) {
    if (!selected.has(r.rejection_stage)) continue;
    if (!r.scan_timestamp) continue;
    const sec = toUnixSeconds(r.scan_timestamp);
    if (!(sec > 0)) continue;
    const bucket = Math.floor(sec / timeframeSec) * timeframeSec;
    out.push({
      time: bucket as Time,
      position: isBull(r.attempted_direction) ? "belowBar" : "aboveBar",
      shape: "circle",
      color: colorForStage(r.rejection_stage),
      size: 1,
    } satisfies SeriesMarker<Time>);
  }
  return out;
}

function countByCategory(
  rejections: RejectionRow[],
  selected: Set<string>
): Record<CategoryKey, number> {
  const out: Record<CategoryKey, number> = {
    structure: 0,
    risk: 0,
    confidence: 0,
    volume: 0,
    momentum: 0,
    sr: 0,
    time: 0,
    market: 0,
    scalp: 0,
    filters: 0,
  };
  for (const r of rejections) {
    if (!selected.has(r.rejection_stage)) continue;
    const cat = stageToCategory[r.rejection_stage];
    if (cat) out[cat] += 1;
  }
  return out;
}

function countByStage(rejections: RejectionRow[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of rejections) {
    out[r.rejection_stage] = (out[r.rejection_stage] ?? 0) + 1;
  }
  return out;
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

  const [trades, setTrades] = useState<Trade[]>([]);
  const [rejections, setRejections] = useState<RejectionRow[]>([]);
  const [rejectionsTruncated, setRejectionsTruncated] = useState(false);
  const [rejectionsTotal, setRejectionsTotal] = useState(0);
  const [selectedStages, setSelectedStages] = useState<Set<string>>(
    () => new Set(ALL_STAGES)
  );
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    rejections: RejectionRow[];
  } | null>(null);

  const timeframeSec = timeframe === "5" ? 300 : 900;

  // Bucketed index of rejections for crosshair lookup.
  const rejectionsByBucket = useMemo(() => {
    const map = new Map<number, RejectionRow[]>();
    for (const r of rejections) {
      if (!selectedStages.has(r.rejection_stage)) continue;
      if (!r.scan_timestamp) continue;
      const sec = Math.floor(new Date(r.scan_timestamp).getTime() / 1000);
      if (!(sec > 0)) continue;
      const bucket = Math.floor(sec / timeframeSec) * timeframeSec;
      const arr = map.get(bucket);
      if (arr) arr.push(r);
      else map.set(bucket, [r]);
    }
    return map;
  }, [rejections, selectedStages, timeframeSec]);

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

  // Load candles + trades + rejections whenever epic, timeframe, or date range changes
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
        setTrades([]);
        setRejections([]);
        setRejectionsTruncated(false);
        setRejectionsTotal(0);
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

      const [tradeRes, rejRes] = await Promise.all([
        fetch(
          `/trading/api/chart/trades?epic=${encodeURIComponent(selectedEpic)}&from=${firstTs}&to=${lastTs}&environment=${configSet}`
        ),
        fetch(
          `/trading/api/chart/smc-rejections?epic=${encodeURIComponent(selectedEpic)}&from=${firstTs}&to=${lastTs}`
        ),
      ]);

      let nextTrades: Trade[] = [];
      if (tradeRes.ok) {
        const tradeData = await tradeRes.json();
        nextTrades = tradeData.trades ?? [];
      }
      setTrades(nextTrades);
      setTradeCount(nextTrades.length);

      let nextRejections: RejectionRow[] = [];
      let nextTruncated = false;
      let nextTotal = 0;
      if (rejRes.ok) {
        const rejData = await rejRes.json();
        nextRejections = rejData.rejections ?? [];
        nextTruncated = Boolean(rejData.truncated);
        nextTotal = rejData.total ?? nextRejections.length;
      }
      setRejections(nextRejections);
      setRejectionsTruncated(nextTruncated);
      setRejectionsTotal(nextTotal);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [selectedEpic, timeframe, dateFrom, dateTo, configSet]);

  useEffect(() => {
    const timer = setTimeout(() => loadData(), 50);
    return () => clearTimeout(timer);
  }, [loadData]);

  // Merge + render markers whenever trades, rejections, selection, or timeframe change
  useEffect(() => {
    if (!seriesRef.current) return;
    const merged = [
      ...buildMarkers(trades),
      ...buildRejectionMarkers(rejections, selectedStages, timeframeSec),
    ].sort((a, b) => (a.time as number) - (b.time as number));
    seriesRef.current.setMarkers(merged);
  }, [trades, rejections, selectedStages, timeframeSec]);

  // Crosshair hover → show rejection details for the hovered bar bucket.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const handler = (param: {
      time?: unknown;
      point?: { x: number; y: number };
    }) => {
      if (!param || !param.time || !param.point) {
        setHoverInfo(null);
        return;
      }
      const bucket = Number(param.time);
      const matches = rejectionsByBucket.get(bucket);
      if (!matches || matches.length === 0) {
        setHoverInfo(null);
        return;
      }
      setHoverInfo({ x: param.point.x, y: param.point.y, rejections: matches });
    };

    chart.subscribeCrosshairMove(handler as any);
    return () => {
      chart.unsubscribeCrosshairMove(handler as any);
    };
  }, [rejectionsByBucket]);

  const stageCounts = useMemo(() => countByStage(rejections), [rejections]);
  const categoryCounts = useMemo(
    () => countByCategory(rejections, selectedStages),
    [rejections, selectedStages]
  );
  const visibleRejectionCount = useMemo(
    () => Object.values(categoryCounts).reduce((a, b) => a + b, 0),
    [categoryCounts]
  );

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

        {/* Rejection stage filter */}
        <RejectionStageFilter
          selected={selectedStages}
          onChange={setSelectedStages}
          counts={stageCounts}
        />

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

        {rejectionsTotal > 0 && (
          <span style={{ fontSize: "12px", color: "#64748b" }}>
            {visibleRejectionCount}/{rejectionsTotal} rejections
          </span>
        )}

        {rejectionsTruncated && (
          <span
            style={{
              fontSize: "11px",
              color: "#fbbf24",
              background: "#422006",
              border: "1px solid #78350f",
              borderRadius: "4px",
              padding: "2px 6px",
            }}
            title={`Only ${rejections.length} of ${rejectionsTotal} rejections loaded. Narrow the date range for complete view.`}
          >
            ⚠ truncated — narrow range
          </span>
        )}
      </div>

      {/* Legend row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "16px",
          padding: "6px 16px",
          borderBottom: "1px solid #1e293b",
          flexShrink: 0,
          flexWrap: "wrap",
          fontSize: "11px",
          color: "#64748b",
        }}
      >
        <div style={{ display: "flex", gap: "12px" }}>
          <span style={{ color: "#94a3b8", fontWeight: 600 }}>Trades:</span>
          <span><span style={{ color: "#22c55e" }}>▲</span> BUY win</span>
          <span><span style={{ color: "#ef4444" }}>▲▼</span> loss</span>
          <span><span style={{ color: "#22c55e" }}>▼</span> SELL win</span>
          <span><span style={{ color: "#f59e0b" }}>●</span> open</span>
        </div>
        <div
          style={{
            width: "1px",
            height: "16px",
            background: "#1e293b",
            alignSelf: "center",
          }}
        />
        <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
          <span style={{ color: "#94a3b8", fontWeight: 600 }}>Rejections:</span>
          {CATEGORY_ORDER.map((key) => {
            const cat = CATEGORIES[key];
            const count = categoryCounts[key];
            const dim = count === 0;
            return (
              <span
                key={key}
                style={{
                  opacity: dim ? 0.35 : 1,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px",
                }}
                title={`${cat.description}\n\nStages: ${cat.stages.join(", ")}`}
              >
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    background: cat.color,
                    display: "inline-block",
                  }}
                />
                {cat.label} ({count})
              </span>
            );
          })}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: "6px 16px", background: "#450a0a", color: "#fca5a5", fontSize: "13px", flexShrink: 0 }}>
          {error}
        </div>
      )}

      {/* Chart canvas */}
      <div ref={chartContainerRef} style={{ flex: 1, minHeight: 0, position: "relative" }}>
        {hoverInfo && hoverInfo.rejections.length > 0 && (
          <RejectionTooltip info={hoverInfo} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Rejection hover tooltip
// ---------------------------------------------------------------------------

function RejectionTooltip({
  info,
}: {
  info: { x: number; y: number; rejections: RejectionRow[] };
}) {
  const MAX_ROWS = 8;
  const rows = info.rejections.slice(0, MAX_ROWS);
  const overflow = info.rejections.length - rows.length;

  // Position to the right of crosshair, clamp so it stays on-screen
  const left = Math.max(8, info.x + 14);
  const top = Math.max(8, info.y + 14);

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        maxWidth: "420px",
        background: "#0f172a",
        border: "1px solid #334155",
        borderRadius: "6px",
        padding: "8px 10px",
        boxShadow: "0 8px 24px rgba(0,0,0,0.55)",
        fontSize: "11px",
        color: "#e2e8f0",
        pointerEvents: "none",
        zIndex: 10,
      }}
    >
      <div
        style={{
          fontWeight: 600,
          fontSize: "11px",
          color: "#94a3b8",
          marginBottom: "6px",
          borderBottom: "1px solid #1e293b",
          paddingBottom: "4px",
        }}
      >
        {info.rejections.length} rejection{info.rejections.length !== 1 ? "s" : ""} at this bar
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
        {rows.map((r) => {
          const color = colorForStage(r.rejection_stage);
          const dir = r.attempted_direction ?? "";
          return (
            <div key={r.id} style={{ display: "flex", gap: "6px", alignItems: "flex-start" }}>
              <span
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  background: color,
                  marginTop: "4px",
                  flexShrink: 0,
                }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", gap: "6px", alignItems: "baseline" }}>
                  <span style={{ fontFamily: "monospace", color: "#f1f5f9", fontWeight: 600 }}>
                    {r.rejection_stage}
                  </span>
                  {dir && (
                    <span
                      style={{
                        color: dir === "BULL" || dir === "BUY" ? "#22c55e" : "#ef4444",
                        fontSize: "10px",
                        fontWeight: 600,
                      }}
                    >
                      {dir}
                    </span>
                  )}
                  <span style={{ color: "#64748b", fontSize: "10px", marginLeft: "auto" }}>
                    {new Date(r.scan_timestamp).toLocaleTimeString(undefined, {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
                <div style={{ color: "#94a3b8", fontSize: "10px", marginTop: "1px" }}>
                  {describeStage(r.rejection_stage)}
                </div>
                {r.rejection_reason && (
                  <div
                    style={{
                      color: "#cbd5e1",
                      fontSize: "11px",
                      marginTop: "2px",
                      wordBreak: "break-word",
                    }}
                  >
                    {r.rejection_reason}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {overflow > 0 && (
          <div style={{ color: "#64748b", fontSize: "10px", fontStyle: "italic" }}>
            +{overflow} more…
          </div>
        )}
      </div>
    </div>
  );
}
