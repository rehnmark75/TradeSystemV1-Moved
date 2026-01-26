/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type TradeItem = {
  id: number;
  symbol: string;
  direction: string;
  timestamp: string;
  status: string;
  profit_loss: number | null;
  pnl_currency: string | null;
  pnl_display: string;
};

type TradeListPayload = {
  trades: TradeItem[];
};

type AnalysisState = {
  loading: boolean;
  error: string | null;
  data: Record<string, any> | null;
};

const formatDateTime = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit"
  });
};

const formatValue = (value: unknown) => {
  if (value == null) return "-";
  if (typeof value === "number") {
    if (Number.isFinite(value)) {
      return value.toFixed(2);
    }
    return "-";
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value;
  return "View raw";
};

const filterEntries = (data: Record<string, any> | null, keys: string[]) =>
  keys.map((key) => ({ key, value: data?.[key] }));

const hasMeaningfulValue = (value: unknown) => {
  if (value == null) return false;
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value === "string") return value.trim().length > 0;
  if (typeof value === "boolean") return true;
  if (typeof value === "object") return Object.keys(value as Record<string, unknown>).length > 0;
  return false;
};

const summarizeStage = (stage: any) => {
  if (!stage) return "-";
  if (typeof stage !== "object") return formatValue(stage);
  const activated = stage.activated === true ? "Activated" : "Not reached";
  const trigger =
    stage.trigger_threshold ??
    stage.trigger_pts ??
    stage.trigger_points ??
    stage.trigger;
  const lock =
    stage.lock_amount ??
    stage.lock_points ??
    stage.final_lock ??
    stage.actual_lock ??
    stage.lock;
  const time = stage.activation_time ?? stage.time;
  const maxProfit = stage.max_profit_reached ?? stage.max_profit;
  const parts = [
    activated,
    trigger != null ? `trigger ${trigger}` : null,
    lock != null ? `lock ${lock}` : null,
    maxProfit != null ? `max ${maxProfit}` : null,
    time != null ? `time ${time}` : null
  ].filter(Boolean);
  return parts.length ? parts.join(" | ") : "View raw";
};

const summarizeMarketStructure = (structure: any) => {
  if (!structure) return "-";
  if (typeof structure !== "object") return formatValue(structure);
  const current =
    structure.current_structure ?? structure.structure_type ?? structure.trend ?? structure.type;
  const trendStrength = structure.trend_strength ?? structure.strength;
  const swingHigh = structure.swing_high ?? structure.high;
  const swingLow = structure.swing_low ?? structure.low;
  const breaks = Array.isArray(structure.structure_breaks)
    ? structure.structure_breaks.length
    : structure.breaks;
  const parts = [
    current ? `structure ${current}` : null,
    trendStrength != null ? `strength ${formatValue(trendStrength)}` : null,
    swingHigh != null ? `swing high ${formatValue(swingHigh)}` : null,
    swingLow != null ? `swing low ${formatValue(swingLow)}` : null,
    breaks != null ? `breaks ${breaks}` : null
  ].filter(Boolean);
  return parts.length ? parts.join(" | ") : "View raw";
};

const renderKeyGrid = (items: Array<{ key: string; value: unknown }>) => (
  <div className="kv-grid">
    {items.map((item) => (
      <div key={item.key} className="kv-card">
        <span className="kv-label">{item.key}</span>
        <strong className="kv-value">{formatValue(item.value)}</strong>
      </div>
    ))}
  </div>
);

export default function ForexTradeAnalysisPage() {
  const [tradeList, setTradeList] = useState<TradeItem[]>([]);
  const [selectedTrade, setSelectedTrade] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"trailing" | "signal" | "outcome">("trailing");
  const [trailingState, setTrailingState] = useState<AnalysisState>({
    loading: false,
    error: null,
    data: null
  });
  const [signalState, setSignalState] = useState<AnalysisState>({
    loading: false,
    error: null,
    data: null
  });
  const [outcomeState, setOutcomeState] = useState<AnalysisState>({
    loading: false,
    error: null,
    data: null
  });

  useEffect(() => {
    fetch("/stock-scanner/api/forex/trade-analysis/trades/")
      .then((res) => res.json())
      .then((data: TradeListPayload) => {
        setTradeList(data.trades ?? []);
        if (data.trades?.length) {
          setSelectedTrade(data.trades[0].id);
        }
      })
      .catch(() => setTradeList([]));
  }, []);

  const loadAnalysis = (tradeId: number) => {
    setTrailingState({ loading: true, error: null, data: null });
    setSignalState({ loading: true, error: null, data: null });
    setOutcomeState({ loading: true, error: null, data: null });

    fetch(`/stock-scanner/api/forex/trade-analysis/trade/?tradeId=${tradeId}`)
      .then((res) => res.json())
      .then((data) => setTrailingState({ loading: false, error: null, data }))
      .catch(() =>
        setTrailingState({ loading: false, error: "Failed to load trailing analysis.", data: null })
      );

    fetch(`/stock-scanner/api/forex/trade-analysis/signal/?tradeId=${tradeId}`)
      .then((res) => res.json())
      .then((data) => setSignalState({ loading: false, error: null, data }))
      .catch(() =>
        setSignalState({ loading: false, error: "Failed to load signal analysis.", data: null })
      );

    fetch(`/stock-scanner/api/forex/trade-analysis/outcome/?tradeId=${tradeId}`)
      .then((res) => res.json())
      .then((data) => setOutcomeState({ loading: false, error: null, data }))
      .catch(() =>
        setOutcomeState({ loading: false, error: "Failed to load outcome analysis.", data: null })
      );
  };

  const activeState =
    activeTab === "trailing" ? trailingState : activeTab === "signal" ? signalState : outcomeState;

  const tradeDetails = trailingState.data?.trade_details ?? signalState.data?.trade_details;
  const trailingSummary = trailingState.data?.summary ?? {};
  const stageAnalysis = trailingState.data?.stage_analysis ?? {};
  const signalOverview = signalState.data?.signal_overview ?? {};
  const smartMoney = signalState.data?.smart_money_analysis ?? {};
  const confluence = signalState.data?.confluence_factors ?? {};
  const entryTiming = signalState.data?.entry_timing_analysis ?? {};
  const strategyIndicators = signalState.data?.raw_data?.strategy_indicators ?? {};
  const tier3Entry = strategyIndicators?.tier3_entry ?? {};
  const outcomeSummary = outcomeState.data?.outcome_summary ?? {};
  const priceAction = outcomeState.data?.price_action_analysis ?? {};
  const maeMfe = outcomeState.data?.mae_mfe_analysis ?? priceAction ?? {};
  const exitQuality = outcomeState.data?.exit_quality ?? outcomeState.data?.exit_quality_assessment ?? {};
  const learningInsights = outcomeState.data?.learning_insights ?? {};
  const entryQuality = outcomeState.data?.entry_quality_assessment ?? {};

  const confluenceTags = useMemo(() => {
    if (!confluence?.factors || !Array.isArray(confluence.factors)) return [];
    return confluence.factors
      .filter((item: any) => item?.present)
      .map((item: any) => item?.label || item?.factor || item?.name || "Factor");
  }, [confluence]);

  const confluenceSummary =
    confluence?.factors_present != null && confluence?.factors_total != null
      ? `${confluence.factors_present}/${confluence.factors_total} factors`
      : null;

  const entryTimingItems = filterEntries(entryTiming, [
    "entry_window",
    "pullback_depth",
    "optimal_entry"
  ]);
  const hasEntryTimingValues = entryTimingItems.some((item) => hasMeaningfulValue(item.value));
  const fallbackEntryItems = filterEntries(tier3Entry, [
    "entry_price",
    "pullback_depth",
    "fib_zone",
    "in_optimal_zone"
  ]);

  const summaryItems = filterEntries(trailingSummary, [
    "trade_outcome",
    "exit_reason",
    "max_profit_points",
    "max_profit_price",
    "max_profit_time",
    "final_exit_price"
  ]);
  const hasSummaryValues = summaryItems.some((item) => hasMeaningfulValue(item.value));

  const stageItems = [
    { key: "breakeven", value: summarizeStage(stageAnalysis?.breakeven) },
    { key: "stage1", value: summarizeStage(stageAnalysis?.stage1) },
    { key: "stage2", value: summarizeStage(stageAnalysis?.stage2) },
    { key: "stage3", value: summarizeStage(stageAnalysis?.stage3) }
  ];

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          Trading Hub
        </Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Trade Analysis</h1>
          <p>Deep dive into trailing logic, entry signal, and outcome diagnostics.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <div className="forex-nav">
        <Link href="/forex" className="forex-pill">
          Overview
        </Link>
        <Link href="/forex/strategy" className="forex-pill">
          Strategy Performance
        </Link>
        <Link href="/forex/trade-performance" className="forex-pill">
          Trade Performance
        </Link>
        <Link href="/forex/performance-snapshot" className="forex-pill">
          Performance Snapshot
        </Link>
        <Link href="/forex/market-intelligence" className="forex-pill">
          Market Intelligence
        </Link>
        <Link href="/forex/smc-rejections" className="forex-pill">
          SMC Rejections
        </Link>
      </div>

      <div className="panel">
        <div className="trade-analysis-controls">
          <div>
            <label>Select Trade</label>
            <select
              value={selectedTrade ?? ""}
              onChange={(event) => setSelectedTrade(Number(event.target.value))}
            >
              {tradeList.map((trade) => (
                <option key={trade.id} value={trade.id}>
                  #{trade.id} | {trade.symbol} | {trade.direction} |{" "}
                  {formatDateTime(trade.timestamp)} | {trade.pnl_display}
                </option>
              ))}
            </select>
          </div>
          <button
            className="section-tab active"
            onClick={() => {
              if (selectedTrade) {
                loadAnalysis(selectedTrade);
              }
            }}
          >
            Analyze Trade
          </button>
        </div>

        {tradeDetails ? (
          <div className="metrics-grid">
            <div className="summary-card">
              Symbol
              <strong>{tradeDetails.symbol ?? "-"}</strong>
            </div>
            <div className="summary-card">
              Direction
              <strong>{tradeDetails.direction ?? "-"}</strong>
            </div>
            <div className="summary-card">
              Entry
              <strong>
                {tradeDetails.entry_price != null ? tradeDetails.entry_price.toFixed(5) : "-"}
              </strong>
            </div>
            <div className="summary-card">
              Status
              <strong>{tradeDetails.status ?? "-"}</strong>
            </div>
            <div className="summary-card">
              P&L
              <strong>{tradeDetails.profit_loss ?? "-"}</strong>
            </div>
          </div>
        ) : null}

        <div className="section-tabs">
          <button
            className={`section-tab ${activeTab === "trailing" ? "active" : ""}`}
            onClick={() => setActiveTab("trailing")}
          >
            Trailing Stop
          </button>
          <button
            className={`section-tab ${activeTab === "signal" ? "active" : ""}`}
            onClick={() => setActiveTab("signal")}
          >
            Signal Analysis
          </button>
          <button
            className={`section-tab ${activeTab === "outcome" ? "active" : ""}`}
            onClick={() => setActiveTab("outcome")}
          >
            Outcome Analysis
          </button>
        </div>

        {activeState.error ? <div className="error">{activeState.error}</div> : null}

        {activeState.loading ? (
          <div className="chart-placeholder">Loading analysis...</div>
        ) : activeTab === "trailing" ? (
          <div className="trade-analysis-grid">
            <div className="panel">
              <div className="chart-title">Summary</div>
              {hasSummaryValues ? (
                renderKeyGrid(summaryItems)
              ) : (
                <div className="muted">No summary data available for this trade.</div>
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Trade Details</div>
              {renderKeyGrid(
                filterEntries(trailingState.data?.trade_details ?? {}, [
                  "symbol",
                  "direction",
                  "entry_price",
                  "sl_price",
                  "tp_price",
                  "status"
                ])
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Pair Configuration</div>
              {renderKeyGrid(
                filterEntries(trailingState.data?.pair_configuration ?? {}, [
                  "break_even_trigger_points",
                  "stage1_trigger_points",
                  "stage1_lock_points",
                  "stage2_trigger_points",
                  "stage2_lock_points",
                  "stage3_trigger_points",
                  "stage3_atr_multiplier"
                ])
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Stage Analysis</div>
              {renderKeyGrid(stageItems)}
            </div>
          </div>
        ) : activeTab === "signal" ? (
          <div className="trade-analysis-grid">
            <div className="panel">
              <div className="chart-title">Signal Overview</div>
              {renderKeyGrid(
                filterEntries(signalOverview, [
                  "pair",
                  "epic",
                  "direction",
                  "strategy",
                  "timeframe",
                  "confidence_score",
                  "price_at_signal",
                  "spread_pips",
                  "market_session"
                ])
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Smart Money / Structure</div>
              {renderKeyGrid(
                [
                  ...filterEntries(smartMoney, ["validated", "type", "score"]),
                  { key: "market_structure", value: summarizeMarketStructure(smartMoney?.market_structure) }
                ]
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Confluence & Timing</div>
              {hasEntryTimingValues ? (
                renderKeyGrid(entryTimingItems)
              ) : (
                renderKeyGrid(fallbackEntryItems)
              )}
              {confluenceSummary ? (
                <div className="muted">Confluence: {confluenceSummary}</div>
              ) : null}
              {confluenceTags.length ? (
                <div className="tag-list">
                  {confluenceTags.map((tag: string) => (
                    <span key={tag} className="tag">
                      {tag}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="muted">No confluence tags available.</div>
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Raw Signal Data</div>
              <pre className="analysis-json">
                {JSON.stringify(signalState.data?.raw_data ?? {}, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <div className="trade-analysis-grid">
            <div className="panel">
              <div className="chart-title">Outcome Summary</div>
              {renderKeyGrid([
                {
                  key: "outcome",
                  value: outcomeSummary.outcome ?? outcomeSummary.result
                },
                {
                  key: "pnl",
                  value: outcomeSummary.pnl ?? outcomeSummary.profit_loss ?? tradeDetails?.profit_loss
                },
                {
                  key: "pnl_pips",
                  value: outcomeSummary.pnl_pips ?? outcomeSummary.pips_gained
                },
                {
                  key: "exit_type",
                  value: outcomeSummary.exit_type ?? exitQuality.exit_type
                },
                {
                  key: "time_in_trade",
                  value: outcomeSummary.time_in_trade ?? outcomeSummary.duration_display
                }
              ])}
            </div>
            <div className="panel">
              <div className="chart-title">MAE / MFE</div>
              {renderKeyGrid(
                [
                  {
                    key: "mae",
                    value: maeMfe.mae_pips ?? priceAction?.mae?.pips ?? maeMfe.mae
                  },
                  {
                    key: "mfe",
                    value: maeMfe.mfe_pips ?? priceAction?.mfe?.pips ?? maeMfe.mfe
                  },
                  {
                    key: "mae_price",
                    value: maeMfe.mae_price ?? priceAction?.mae?.price
                  },
                  {
                    key: "mfe_price",
                    value: maeMfe.mfe_price ?? priceAction?.mfe?.price
                  },
                  {
                    key: "mfe_mae_ratio",
                    value: maeMfe.mfe_mae_ratio ?? priceAction?.mfe_mae_ratio
                  }
                ]
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Insights</div>
              {renderKeyGrid([
                { key: "entry_quality", value: entryQuality.verdict ?? entryQuality.percentage },
                { key: "exit_quality", value: exitQuality.verdict ?? exitQuality.exit_efficiency_pct },
                { key: "notes", value: learningInsights.key_takeaway }
              ])}
              {renderKeyGrid(
                filterEntries(learningInsights, ["lesson", "suggestion", "mistake"])
              )}
            </div>
            <div className="panel">
              <div className="chart-title">Raw Outcome Data</div>
              <pre className="analysis-json">
                {JSON.stringify(outcomeState.data?.raw_data ?? {}, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
