"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";
import TradeFilters, { type FilterState } from "./_components/TradeFilters";
import TradeTable, { type TradeRow } from "./_components/TradeTable";
import TradeChart from "./_components/TradeChart";
import TrailingTimeline from "./_components/TrailingTimeline";
import CounterfactualBadge, { type CounterfactualVerdict } from "./_components/CounterfactualBadge";
import PostmortemPanel, { type PostmortemData } from "./_components/PostmortemPanel";

type ChartData = {
  trade_id: number;
  symbol: string;
  direction: string;
  entry_price: number;
  initial_sl: number | null;
  current_sl: number | null;
  tp: number | null;
  open_time: number;
  close_time: number;
  profit_loss: number | null;
  pnl_currency: string | null;
  early_be_executed: boolean;
  stop_limit_changes_count: number;
  candles: { time: number; open: number; high: number; low: number; close: number }[];
  sl_history: { time_iso: string; sl: number; event: string }[];
  mfe: { price: number | null; pips: number; time: number | null };
  mae: { price: number | null; pips: number; time: number | null };
  counterfactual: { verdict: CounterfactualVerdict; delta_pips: number | null; would_have_hit: string | null } | null;
};

const PIP_MULT: Record<string, number> = {};
function getPipMult(symbol: string): number {
  if (symbol.includes("JPY")) return 100;
  if (symbol.includes("CEEM")) return 1;
  return 10000;
}

const fmtPrice = (v: number | null | undefined, digits = 5) =>
  v == null ? "-" : v.toFixed(digits);

const fmtPips = (v: number | null) =>
  v == null ? "-" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;

const fmtDuration = (mins: number | null) => {
  if (mins == null) return "-";
  if (mins < 60) return `${mins}m`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
};

export default function ForexTradeAnalysisPage() {
  const { environment } = useEnvironment();
  const [filters, setFilters] = useState<FilterState>({ from: "", to: "", epic: "", outcome: "" });
  const [epics, setEpics] = useState<string[]>([]);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [tradesLoading, setTradesLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState<string | null>(null);
  const [postmortem, setPostmortem] = useState<PostmortemData | null>(null);
  const [postmortemStatus, setPostmortemStatus] = useState<"idle" | "loading" | "ready" | "generating" | "error">("idle");
  const detailRef = useRef<HTMLDivElement>(null);

  const fetchTrades = useCallback(() => {
    setTradesLoading(true);
    const params = new URLSearchParams({ env: environment });
    if (filters.from) params.set("from", filters.from);
    if (filters.to) params.set("to", filters.to);
    if (filters.epic) params.set("epic", filters.epic);
    if (filters.outcome) params.set("outcome", filters.outcome);
    fetch(`/trading/api/forex/trade-analysis/trades/?${params}`)
      .then((r) => r.json())
      .then((d) => {
        const rows: TradeRow[] = d.trades ?? [];
        setTrades(rows);
        if (!epics.length) {
          const seen = new Set<string>();
          rows.forEach((t) => t.symbol && seen.add(t.symbol));
          setEpics([...seen].sort());
        }
      })
      .catch(() => setTrades([]))
      .finally(() => setTradesLoading(false));
  }, [environment, filters, epics.length]);

  useEffect(() => {
    fetchTrades();
  }, [fetchTrades]);

  const fetchPostmortem = (id: number, env: string, attempt = 0) => {
    setPostmortemStatus("loading");
    fetch(`/trading/api/forex/trade-analysis/postmortem/?tradeId=${id}&env=${env}`)
      .then(async (r) => {
        const d = await r.json();
        if (r.status === 202) {
          // Still generating — retry once after 4 seconds
          setPostmortemStatus("generating");
          if (attempt < 3) {
            setTimeout(() => fetchPostmortem(id, env, attempt + 1), 4000);
          } else {
            setPostmortemStatus("error");
          }
          return;
        }
        if (!r.ok) { setPostmortemStatus("error"); return; }
        setPostmortem(d.postmortem ?? null);
        setPostmortemStatus(d.postmortem ? "ready" : "error");
      })
      .catch(() => setPostmortemStatus("error"));
  };

  const handleSelect = (id: number) => {
    setSelectedId(id);
    setChartData(null);
    setChartError(null);
    setChartLoading(true);
    setPostmortem(null);
    setPostmortemStatus("idle");
    setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    fetch(`/trading/api/forex/trade-analysis/chart-data/?tradeId=${id}&env=${environment}`)
      .then((r) => r.json())
      .then((d) => {
        setChartData(d);
        // Fetch postmortem in parallel (only for closed trades)
        fetchPostmortem(id, environment);
      })
      .catch(() => setChartError("Failed to load chart data."))
      .finally(() => setChartLoading(false));
  };

  const selectedTrade = trades.find((t) => t.id === selectedId) ?? null;
  const pipMult = chartData ? getPipMult(chartData.symbol) : 10000;

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">K.L.I.R.R</Link>
        <EnvironmentToggle />
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Trade Analysis</h1>
          <p>Inspect trailing-stop history, price action, and counterfactual outcomes.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/trade-analysis" />

      <div className="panel">
        <TradeFilters filters={filters} onChange={setFilters} epics={epics} />

        {tradesLoading ? (
          <div className="chart-placeholder">Loading trades…</div>
        ) : (
          <TradeTable trades={trades} selectedId={selectedId} onSelect={handleSelect} />
        )}
      </div>

      {selectedId !== null && (
        <div ref={detailRef} className="panel ta-detail-panel">
          {chartLoading && <div className="chart-placeholder">Loading chart data…</div>}
          {chartError && <div className="error">{chartError}</div>}
          {chartData && (
            <>
              {/* Summary cards */}
              <div className="ta-summary-cards">
                <div className="summary-card">
                  Pair
                  <strong>{chartData.symbol.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", "")}</strong>
                </div>
                <div className="summary-card">
                  Direction
                  <strong className={chartData.direction === "BUY" ? "dir-buy" : "dir-sell"}>
                    {chartData.direction}
                  </strong>
                </div>
                <div className="summary-card">
                  Entry
                  <strong>{fmtPrice(chartData.entry_price)}</strong>
                </div>
                <div className="summary-card">
                  Orig SL
                  <strong>{fmtPrice(chartData.initial_sl)}</strong>
                </div>
                <div className="summary-card">
                  TP
                  <strong>{fmtPrice(chartData.tp)}</strong>
                </div>
                <div className="summary-card">
                  MFE
                  <strong className="pips-pos">{fmtPips(chartData.mfe?.pips)}</strong>
                </div>
                <div className="summary-card">
                  MAE
                  <strong className="pips-neg">
                    {chartData.mae?.pips > 0 ? `-${chartData.mae.pips.toFixed(1)}` : "0.0"}
                  </strong>
                </div>
                <div className="summary-card">
                  Duration
                  <strong>{fmtDuration(selectedTrade?.lifecycle_duration_minutes ?? null)}</strong>
                </div>
                <div className="summary-card">
                  P&amp;L
                  <strong>
                    {chartData.profit_loss != null
                      ? `${chartData.profit_loss >= 0 ? "+" : ""}${chartData.profit_loss.toFixed(2)} ${chartData.pnl_currency ?? ""}`
                      : "-"}
                  </strong>
                </div>
                {chartData.counterfactual && (
                  <div className="summary-card">
                    Counterfactual
                    <CounterfactualBadge
                      verdict={chartData.counterfactual.verdict}
                      deltaP={chartData.counterfactual.delta_pips}
                      wouldHaveHit={chartData.counterfactual.would_have_hit}
                    />
                  </div>
                )}
              </div>

              {/* Chart */}
              <div className="ta-chart-wrap">
                <TradeChart
                  candles={chartData.candles}
                  entry={chartData.entry_price}
                  initialSl={chartData.initial_sl}
                  tp={chartData.tp}
                  slHistory={chartData.sl_history}
                  mfe={chartData.mfe}
                  mae={chartData.mae}
                  openTime={chartData.open_time}
                  closeTime={chartData.close_time}
                  direction={chartData.direction}
                />
              </div>

              {/* Trailing timeline */}
              {chartData.sl_history.length > 0 && (
                <div className="ta-timeline-wrap">
                  <div className="chart-title">Trailing Stop Timeline</div>
                  <TrailingTimeline
                    slHistory={chartData.sl_history}
                    openTime={chartData.open_time}
                    closeTime={chartData.close_time}
                    entry={chartData.entry_price}
                    direction={chartData.direction}
                    pipMult={pipMult}
                  />
                </div>
              )}

              {/* AI Post-Mortem */}
              <div className="ta-postmortem-wrap">
                <div className="chart-title">AI Post-Mortem</div>
                {postmortemStatus === "loading" && (
                  <div className="chart-placeholder">Generating post-mortem…</div>
                )}
                {postmortemStatus === "generating" && (
                  <div className="chart-placeholder">Generating post-mortem… retrying</div>
                )}
                {postmortemStatus === "error" && (
                  <div className="chart-placeholder" style={{ color: "#94a3b8" }}>
                    Post-mortem not available for this trade.
                  </div>
                )}
                {postmortemStatus === "ready" && postmortem && (
                  <PostmortemPanel data={postmortem} />
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
